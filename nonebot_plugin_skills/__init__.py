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
    description="基于 Gemini V2 架构的智能技能插件，支持 long-term memory 与自动化工具调用",
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

def _normalize_msg_id(msg_id: Any) -> str:
    try:
        val = int(msg_id)
        if val < 0: val = val & 0xFFFFFFFF
        return str(val)
    except: return str(msg_id).strip()

def _should_trigger(event: MessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent): return True
    if event.is_tome(): return True
    text = event.get_plaintext().lower()
    for kw in config.bot_keywords:
        if kw.lower() in text: return True
    return False


def _extract_recall_query_count(text: str) -> int:
    if any(kw in text for kw in ("全部", "所有", "all")):
        return 100

    match = re.search(r"(\d+)\s*条?", text)
    if match:
        try:
            return max(1, int(match.group(1)))
        except Exception:
            pass
    return 100


def _should_handle_recall_directly(text: str) -> bool:
    if "撤回" not in text:
        return False

    direct_patterns = (
        "最近撤回",
        "撤回了什么",
        "撤回消息",
        "查看撤回",
        "哪些撤回",
        "撤回记录",
    )
    return any(pattern in text for pattern in direct_patterns)


async def _handle_direct_recall_query(bot: Bot, event: MessageEvent, session_id: str, user_id: str, raw_text: str) -> bool:
    if not _should_handle_recall_directly(raw_text):
        return False

    from nonebot_plugin_skills.v2.core.context import SkillContext
    from nonebot_plugin_skills.v2.skills.recall import build_recall_forward_messages, recall_view

    skill_context = SkillContext(
        bot=bot,
        event=event,
        session_id=session_id,
        user_id=user_id,
        raw_text=raw_text,
    )

    if isinstance(event, GroupMessageEvent):
        message_parts = await build_recall_forward_messages(
            count=_extract_recall_query_count(raw_text),
            context=skill_context,
        )
        if message_parts:
            sent = await _send_forward_message_parts(bot, event, message_parts)
            if sent:
                memory_core.add_message(session_id, "model", f"[系统：已转发 {len(message_parts)} 条撤回记录]")
                return True

    result = await recall_view(
        count=_extract_recall_query_count(raw_text),
        context=skill_context,
    )
    should_force_forward = (
        isinstance(event, GroupMessageEvent)
        and config.combine_message_threshold > 0
        and len(result.strip()) >= config.combine_message_threshold
    )
    await _send_reply(bot, event, Message(result), force_forward=should_force_forward)
    memory_core.add_message(session_id, "model", str(result))
    return True

async def _persistent_message_content(message: Message) -> str:
    """持久化图片内容为 Base64"""
    new_msg = Message()
    from nonebot_plugin_skills.v2.core.nlp import _get_image_data
    from nonebot_plugin_skills.v2.core.utils import build_image_segment
    for seg in message:
        if seg.type == "image":
            url = seg.data.get("url")
            if url:
                data = await _get_image_data(url)
                if data:
                    img_seg = await build_image_segment(data)
                    new_msg.append(img_seg)
                    continue
        new_msg.append(seg)
    return str(new_msg)

async def _cache_message_sync(msg_id: Any, nickname: str, message: Message):
    """关键逻辑：先完成持久化下载，再存入缓存，确保图片永不失效"""
    content = await _persistent_message_content(message)
    sid = _normalize_msg_id(msg_id)
    _MESSAGE_CACHE[sid] = (nickname, content)
    if len(_MESSAGE_CACHE) > 5000:
        _MESSAGE_CACHE.pop(next(iter(_MESSAGE_CACHE)))
    logger.debug(f"Fully persisted & cached message {sid} from {nickname}")


def _should_use_forward_reply(event: MessageEvent, full_msg: Message) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    if config.combine_message_threshold <= 0:
        return False

    msg_str = str(full_msg).strip()
    msg_str = _MESSAGE_ID_REPLY_RE.sub("", msg_str, count=1).strip()
    visible_length = max(len(Message(msg_str).extract_plain_text().strip()), len(msg_str))
    return visible_length >= config.combine_message_threshold


def _message_to_forward_content(message: Message) -> list[dict[str, Any]]:
    return [{"type": seg.type, "data": dict(seg.data)} for seg in message]


def _split_forward_nodes(full_msg: Message) -> list[Message]:
    raw_text = str(full_msg)
    parts = [p.strip() for p in raw_text.split("<botbr>") if p.strip()]
    if not parts:
        return []

    nodes: list[Message] = []
    max_len = max(config.combine_message_threshold, 120)

    for part in parts:
        subparts = [s.strip() for s in re.split(r"\n{2,}", part) if s.strip()]
        if not subparts:
            subparts = [part]

        for subpart in subparts:
            if len(subpart) <= max_len:
                nodes.append(Message(subpart))
                continue

            lines = [line.strip() for line in subpart.splitlines() if line.strip()]
            if lines:
                current = ""
                for line in lines:
                    candidate = f"{current}\n{line}".strip() if current else line
                    if current and len(candidate) > max_len:
                        nodes.append(Message(current))
                        current = line
                    else:
                        current = candidate
                if current:
                    nodes.append(Message(current))
                continue

            for start in range(0, len(subpart), max_len):
                chunk = subpart[start : start + max_len].strip()
                if chunk:
                    nodes.append(Message(chunk))

    return nodes


async def _send_forward_message_parts(bot: Bot, event: GroupMessageEvent, message_parts: list[Message], cache_message: Message | None = None) -> bool:
    if not message_parts:
        return False

    attempts = [
        [
            MessageSegment.node_custom(
                user_id=int(bot.self_id),
                nickname="嘉然",
                content=message_part,
            )
            for message_part in message_parts
        ],
        [
            {
                "type": "node",
                "data": {
                    "user_id": str(bot.self_id),
                    "nickname": "嘉然",
                    "content": _message_to_forward_content(message_part),
                },
            }
            for message_part in message_parts
        ],
        [
            {
                "type": "node",
                "data": {
                    "uin": str(bot.self_id),
                    "name": "嘉然",
                    "content": _message_to_forward_content(message_part),
                },
            }
            for message_part in message_parts
        ],
    ]

    errors = []
    for idx, nodes in enumerate(attempts, 1):
        try:
            result = await bot.call_api(
                "send_group_forward_msg",
                group_id=event.group_id,
                messages=nodes,
            )
            if result and isinstance(result, dict) and "message_id" in result and cache_message:
                asyncio.create_task(_cache_message_sync(result["message_id"], "嘉然", cache_message))
            logger.info(f"Forward send succeeded with strategy #{idx}")
            return True
        except Exception as e:
            errors.append(f"#{idx}: {e}")

    logger.warning(f"Forward send failed, fallback to normal send: {' | '.join(errors)}")
    return False


async def _send_forward_reply(bot: Bot, event: GroupMessageEvent, full_msg: Message) -> bool:
    message_parts = _split_forward_nodes(full_msg)
    return await _send_forward_message_parts(bot, event, message_parts, cache_message=full_msg)

async def _send_reply(bot: Bot, event: MessageEvent, reply: Union[str, Message], force_forward: bool = False):
    if not reply: return
    full_msg = Message(reply) if isinstance(reply, str) else reply
    if force_forward and isinstance(event, GroupMessageEvent):
        sent = await _send_forward_reply(bot, event, full_msg)
        if sent:
            return

    if not force_forward and _should_use_forward_reply(event, full_msg):
        sent = await _send_forward_reply(bot, event, full_msg)
        if sent:
            return

    msg_str = str(full_msg)
    parts_texts = [p.strip() for p in msg_str.split("<botbr>") if p.strip()]
    
    for i, p_text in enumerate(parts_texts):
        try:
            reply_seg = None
            reply_match = _MESSAGE_ID_REPLY_RE.search(p_text)
            if reply_match:
                reply_seg = MessageSegment.reply(int(reply_match.group(1)))
                p_text = _MESSAGE_ID_REPLY_RE.sub("", p_text, count=1).strip()

            sub_segments = re.split(r"(\[CQ:image,[^\]]+\])", p_text)
            
            for seg_text in sub_segments:
                seg_text = seg_text.strip()
                if not seg_text: continue
                
                result = None
                pending_prefix = Message(reply_seg) if reply_seg else Message()
                if "[CQ:image" in seg_text:
                    file_match = re.search(r"file=([^,\]]+)", seg_text)
                    if file_match:
                        file_val = file_match.group(1)
                        result = await nlp_handler.send(pending_prefix + Message(MessageSegment.image(file_val)))
                else:
                    result = await nlp_handler.send(pending_prefix + Message(seg_text))
                
                # 机器人自己发出的消息也执行同步持久化缓存
                if result and isinstance(result, dict) and "message_id" in result:
                    # 机器人发出的通常已经是 base64 或正在发，也要存入缓存以便管理员撤回时能查到
                    asyncio.create_task(_cache_message_sync(result["message_id"], "嘉然", Message(seg_text)))

            if i < len(parts_texts) - 1: await asyncio.sleep(0.8)
        except Exception as e: logger.error(f"Failed to send message part: {e}")

@nlp_handler.handle()
async def _(bot: Bot, event: MessageEvent):
    msg_id = getattr(event, "message_id", None)
    raw_text = str(event.get_message()).strip()
    sender = getattr(event, "sender", None)
    nickname = getattr(sender, "nickname", "用户") if sender else "用户"
    
    if msg_id:
        # 这里必须先启动持久化下载，确保图片被转为 Base64
        asyncio.create_task(_cache_message_sync(msg_id, nickname, event.get_message()))

    session_id = f"group_{event.group_id}" if isinstance(event, GroupMessageEvent) else f"private_{event.user_id}"
    user_id = str(event.user_id)

    if raw_text:
        if user_id == str(bot.self_id): memory_core.add_message(session_id, "model", str(raw_text), message_id=msg_id)
        else: memory_core.add_message(session_id, "user", f"{nickname}: {raw_text}", message_id=msg_id)

    if user_id == str(bot.self_id) or not config.nlp_enable or not _should_trigger(event): return

    if await _handle_direct_recall_query(bot, event, session_id, user_id, raw_text):
        return

    try:
        await handle_user_message(bot=bot, event=event, session_id=session_id, user_id=user_id, text=raw_text, already_added=True)
    except Exception as e:
        from nonebot.exception import FinishedException
        if not isinstance(e, FinishedException): logger.error(f"NLP V2 Error: {e}")

# --- 撤回监控 ---
recall_notice = on_notice(priority=10)
@recall_notice.handle()
async def _(bot: Bot, event: Union[GroupRecallNoticeEvent, FriendRecallNoticeEvent]):
    raw_mid = getattr(event, "message_id", 0)
    msg_id = _normalize_msg_id(raw_mid)
    sender_id = str(event.user_id)
    operator_id = str(getattr(event, "operator_id", sender_id))
    session_id = f"group_{event.group_id}" if isinstance(event, GroupRecallNoticeEvent) else f"private_{event.user_id}"
    
    logger.info(f"Recall detected: msg_id={msg_id}, operator={operator_id} in {session_id}")
    if msg_id in _MESSAGE_CACHE:
        nickname, content = _MESSAGE_CACHE[msg_id]
        note = f" (被管理员 {operator_id} 撤回)" if operator_id != sender_id else ""
        recall_obj = RecalledMessage(message_id=int(msg_id) if msg_id.isdigit() else 0, user_id=sender_id, nickname=nickname, text=f"{content}{note}")
        memory_core.add_recall(session_id, recall_obj)
        recall_role = "model" if sender_id == str(bot.self_id) else "user"
        recall_content = f"{nickname} 撤回了消息：{content}{note}"
        memory_core.add_message(
            session_id,
            recall_role,
            recall_content,
            recalled_message_id=msg_id,
        )
        logger.info(f"Recorded recall: 原作者={nickname}")
    else: logger.warning(f"Recall event ignored - msg_id {msg_id} not in cache.")

# --- 管理指令 ---
reload_handler = on_command("重载技能", priority=10, block=True)
@reload_handler.handle()
async def _():
    try:
        skill_manager.load_from_directory(str(V2_SKILLS_DIR), "nonebot_plugin_skills.v2.skills")
        await reload_handler.finish(f"已成功重载 V2 技能引擎，当前加载技能数: {len(skill_manager.skills)}")
    except Exception as e: await reload_handler.finish(f"重载失败: {e}")
