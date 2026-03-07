from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Union, List

from nonebot import logger, on_message, on_command, on_notice
from nonebot.adapters.onebot.v11 import (
    Bot, MessageEvent, GroupMessageEvent, PrivateMessageEvent, 
    Message, MessageSegment, GroupRecallNoticeEvent, FriendRecallNoticeEvent
)
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from .config import config
from .v2.core.nlp import handle_user_message
from .v2.core.skills import skill_manager
from .v2.core.memory import memory_core, RecalledMessage

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-skills",
    description="基于 Gemini V2 架构的智能技能插件，支持长短期记忆与自动化工具调用",
    usage="直接对话 / 提到机器人对话 / 使用关键字对话",
    type="application",
    homepage="https://github.com/yourname/nonebot-plugin-skills",
    supported_adapters={"~onebot.v11"},
)

# --- 初始化 V2 技能引擎 ---
V2_SKILLS_DIR = Path(__file__).parent / "v2" / "skills"
if V2_SKILLS_DIR.exists():
    logger.info(f"Loading V2 skills from {V2_SKILLS_DIR}")
    skill_manager.load_from_directory(str(V2_SKILLS_DIR), "nonebot_plugin_skills.v2.skills")

# --- 核心消息处理器 ---
nlp_handler = on_message(priority=50, block=False)
_MESSAGE_CACHE = {} 
_MESSAGE_ID_REPLY_RE = re.compile(r"\[ID:\s*(\d+)\]")

def _should_trigger(event: MessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return True
    if event.is_tome():
        return True
    text = event.get_plaintext().lower()
    for kw in config.bot_keywords:
        if kw.lower() in text:
            return True
    return False

async def _send_reply(bot: Bot, event: MessageEvent, reply: Union[str, Message]):
    """
    处理消息分段发送。支持字符串或 Message 对象。
    """
    if not reply:
        return

    # 1. 统一转为 Message 处理
    if isinstance(reply, str):
        full_msg = Message(reply)
    else:
        full_msg = reply

    # 2. 检查是否包含图片 (用于禁用合并转发)
    has_image = any(seg.type == "image" for seg in full_msg)

    # 3. 如果包含多条独立回复 (通过 <botbr> 分隔)，需要拆分发送
    # 如果是 Message 对象，先转为字符串寻找分隔符，再重新构建每段的 Message
    msg_str = str(full_msg)
    parts_texts = [p.strip() for p in msg_str.split("<botbr>") if p.strip()]
    
    for i, p_text in enumerate(parts_texts):
        try:
            reply_seg = None
            reply_match = _MESSAGE_ID_REPLY_RE.search(p_text)
            if reply_match:
                reply_seg = MessageSegment.reply(int(reply_match.group(1)))
                p_text = _MESSAGE_ID_REPLY_RE.sub("", p_text, count=1).strip()

            # 这里的核心逻辑：将字符串还原为 Message 对象，以保留二进制图片段
            # 我们根据 p_text 在原始 full_msg 中寻找对应的 segments
            # 简单实现：将 p_text 中的 CQ 码和文字拆开连续发送
            sub_segments = re.split(r"(\[CQ:image,[^\]]+\])", p_text)
            
            pending_prefix = Message(reply_seg) if reply_seg else Message()
            for seg_text in sub_segments:
                seg_text = seg_text.strip()
                if not seg_text: continue
                
                # 如果是图片 CQ 码，转为二进制 MessageSegment 发送以提高成功率
                if "[CQ:image" in seg_text:
                    # 尝试从 CQ 码中提取 file (可能是 base64 或 url)
                    file_match = re.search(r"file=([^,\]]+)", seg_text)
                    if file_match:
                        file_val = file_match.group(1)
                        image_msg = pending_prefix + Message(MessageSegment.image(file_val))
                        await nlp_handler.send(image_msg)
                        pending_prefix = Message()
                else:
                    await nlp_handler.send(pending_prefix + Message(seg_text))
                    pending_prefix = Message()

            if pending_prefix:
                await nlp_handler.send(pending_prefix)
                
                if len(sub_segments) > 1:
                    await asyncio.sleep(0.3)

            if i < len(parts_texts) - 1:
                await asyncio.sleep(0.8)
        except Exception as e:
            logger.error(f"Failed to send message part: {e}")

async def _persistent_message_content(message: Message) -> str:
    """
    将消息中的不稳定内容（如图片 URL）持久化。
    如果是图片，立即下载并转为 Base64。
    """
    new_msg = Message()
    from nonebot_plugin_skills.v2.core.nlp import _get_image_data
    from nonebot_plugin_skills.v2.core.utils import save_image_to_local

    for seg in message:
        if seg.type == "image":
            url = seg.data.get("url")
            if url:
                # 立即下载图片数据
                data = await _get_image_data(url)
                if data:
                    # 压缩并转为 Base64 代码
                    b64_cq = save_image_to_local(data)
                    # 从字符串 [CQ:image,file=...] 还原回消息段
                    # 简单处理：直接作为文本加入，因为 RecalledMessage.text 就是存储字符串的
                    new_msg.append(b64_cq)
                    continue
        new_msg.append(seg)
    return str(new_msg)

@nlp_handler.handle()
async def _(bot: Bot, event: MessageEvent):
    msg_id = getattr(event, "message_id", None)
    # 原始文本用于 AI 触发判断
    raw_text = str(event.get_message()).strip()
    
    sender = getattr(event, "sender", None)
    nickname = getattr(sender, "nickname", "用户") if sender else "用户"
    
    if msg_id:
        # 异步进行图片持久化并存入缓存
        # 强制转为字符串以匹配撤回事件中的查找逻辑
        asyncio.create_task(_cache_message_task(str(msg_id), nickname, event.get_message()))

    session_id = f"group_{event.group_id}" if isinstance(event, GroupMessageEvent) else f"private_{event.user_id}"
    user_id = str(event.user_id)

    if raw_text:
        # 在存入记忆时显式传入 message_id
        if user_id == str(bot.self_id):
            memory_core.add_message(session_id, "model", str(raw_text), message_id=msg_id)
        else:
            memory_core.add_message(session_id, "user", f"{nickname}: {raw_text}", message_id=msg_id)

    if user_id == str(bot.self_id):
        return

    if not config.nlp_enable or not _should_trigger(event):
        return

    try:
        # 新逻辑：不再等待 handle_user_message 返回，它内部会进行流式分段发送
        await handle_user_message(bot=bot, event=event, session_id=session_id, user_id=user_id, text=raw_text, already_added=True)
    except Exception as e:
        from nonebot.exception import FinishedException
        if isinstance(e, FinishedException): raise e
        logger.error(f"NLP V2 Error: {e}")

async def _cache_message_task(msg_id: Any, nickname: str, message: Message):
    """异步持久化任务"""
    content = await _persistent_message_content(message)
    # 强制将 msg_id 转为字符串，确保各种环境下的 ID 匹配稳定性
    sid = str(msg_id)
    _MESSAGE_CACHE[sid] = (nickname, content)
    if len(_MESSAGE_CACHE) > 5000: # 稍微扩大一点缓存
        _MESSAGE_CACHE.pop(next(iter(_MESSAGE_CACHE)))
    logger.debug(f"Cached message {sid} from {nickname}")

# --- 撤回监控 ---
recall_notice = on_notice(priority=10)
@recall_notice.handle()
async def _(bot: Bot, event: Union[GroupRecallNoticeEvent, FriendRecallNoticeEvent]):
    # 强制将 msg_id 转为字符串进行查找
    msg_id = str(event.message_id)
    
    # 获取身份：user_id 是原作者，operator_id 是撤回的人（管理员或本人）
    sender_id = str(event.user_id)
    operator_id = str(getattr(event, "operator_id", sender_id))
    
    session_id = f"group_{event.group_id}" if isinstance(event, GroupRecallNoticeEvent) else f"private_{event.user_id}"
    
    logger.info(f"Recall detected: msg_id={msg_id}, sender={sender_id}, operator={operator_id} in {session_id}")

    if msg_id in _MESSAGE_CACHE:
        nickname, content = _MESSAGE_CACHE[msg_id]
        
        # 备注撤回者身份
        recall_note = ""
        if operator_id != sender_id:
            recall_note = f" (被管理员 {operator_id} 撤回)"
            
        # 记录撤回。这里存储的文本会包含“原作者”和“撤回者”信息
        recall_obj = RecalledMessage(
            message_id=int(msg_id) if msg_id.isdigit() else 0, 
            user_id=sender_id, 
            nickname=nickname, 
            text=f"{content}{recall_note}"
        )
        memory_core.add_recall(session_id, recall_obj)
        logger.info(f"Recorded recall: 原作者={nickname}, 撤回者={operator_id}")
    else:
        logger.warning(f"Recall event ignored - msg_id {msg_id} not in cache.")

# --- 管理指令 ---
reload_handler = on_command("重载技能", priority=10, block=True)
@reload_handler.handle()
async def _():
    try:
        skill_manager.load_from_directory(str(V2_SKILLS_DIR), "nonebot_plugin_skills.v2.skills")
        await reload_handler.finish(f"已成功重载 V2 技能引擎，当前加载技能数: {len(skill_manager.skills)}")
    except Exception as e:
        await reload_handler.finish(f"重载失败: {e}")
