import re
import base64
import hashlib
from typing import Optional, Any, Dict, Union, List
from pathlib import Path
from nonebot import logger
from google import genai
from google.genai import types
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.memory import memory_core
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.http import get_http_client
from nonebot_plugin_skills.v2.core.utils import (
    build_image_segment,
    extract_image_sources,
    local_path_from_file_uri,
)

from nonebot_plugin_skills.config import config

# 图片描述缓存，防止重复请求视觉模型分析同一张图
_IMAGE_DESCRIPTION_CACHE: Dict[str, str] = {}

# --- 核心人设与回复规则 ---
SYSTEM_PROMPT = (
    "你是嘉然(Diana)，A-SOUL成员。性格温柔、体贴、元气满满。\n"
    "【核心回复规则】\n"
    "1. 严禁瞎编！系统在构造上下文时，可能会把历史图片转换成简短的视觉描述；如果你需要对历史图进行修改、重绘或极其精细的细节分析，请务必调用对应的【工具】去获取原图像素。\n"
    "2. 当需要处理图片任务时，优先调用工具。\n"
    "3. 在群聊中如需艾特用户，请使用 OneBot v11 CQ 码格式：[CQ:at,qq=目标QQ号]；在私聊中不要使用@或CQ艾特码。\n"
    "4. 你的历史记录带 [消息ID: xxx] 标记；只有在确实有必要强调上下文或避免歧义时，才使用 [CQ:reply,id=xxx] 引用回复，平时不要每条都引用。\n"
    "5. 不要输出 Markdown 语法标记，例如 **粗体**、__下划线__、`代码`、# 标题，直接用普通聊天文本。"
)


def _build_system_prompt(session_id: str) -> str:
    if session_id.startswith("private_"):
        return (
            f"{SYSTEM_PROMPT}\n"
            "6. 当前是私聊场景，回复中不要使用@、[CQ:at,...]，直接自然回复即可。"
        )
    return SYSTEM_PROMPT

def _get_client() -> Optional[genai.Client]:
    if not config.google_api_key: return None
    return genai.Client(api_key=config.google_api_key)

async def _get_image_data(url: str) -> Optional[bytes]:
    client = get_http_client()
    try:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.content
    except: return None


async def _load_image_bytes(source: str) -> Optional[bytes]:
    if source.startswith("base64://"):
        try:
            return base64.b64decode(source.replace("base64://", "", 1))
        except Exception:
            return None

    if source.startswith("file://"):
        path = local_path_from_file_uri(source)
        if path and path.exists():
            try:
                return path.read_bytes()
            except Exception:
                return None
        return None

    if Path(source).exists():
        try:
            return Path(source).read_bytes()
        except Exception:
            return None

    return await _get_image_data(source)


async def _render_history_text_with_image_descriptions(content: str) -> tuple[str, Dict[str, bytes]]:
    descriptions: Dict[str, str] = {}
    raw_images: Dict[str, bytes] = {}

    for source in extract_image_sources(content):
        if source in descriptions:
            continue
        img_data = await _load_image_bytes(source)
        if not img_data:
            continue
        raw_images[source] = img_data
        descriptions[source] = await _get_image_description(img_data)

    rendered_parts = []
    for seg in Message(content):
        if seg.type == "image":
            source = seg.data.get("file") or seg.data.get("url")
            rendered_parts.append(descriptions.get(source or "", "[图片]"))
        elif seg.type == "text":
            rendered_parts.append(seg.data.get("text", ""))
        else:
            rendered_parts.append(str(seg))

    return "".join(rendered_parts).strip(), raw_images


async def _describe_image_result(result: Union[str, bytes, MessageSegment]) -> str:
    if isinstance(result, bytes):
        return await _get_image_description(result)

    if isinstance(result, MessageSegment):
        source = result.data.get("file") or result.data.get("url")
        if source:
            img_data = await _load_image_bytes(source)
            if img_data:
                return await _get_image_description(img_data)
        return "[图片]"

    for source in extract_image_sources(str(result)):
        img_data = await _load_image_bytes(source)
        if img_data:
            return await _get_image_description(img_data)

    return "[图片]"

async def _get_image_description(image_data: bytes) -> str:
    """获取图片的视觉索引描述 (带缓存)"""
    img_hash = hashlib.md5(image_data).hexdigest()
    if img_hash in _IMAGE_DESCRIPTION_CACHE:
        return _IMAGE_DESCRIPTION_CACHE[img_hash]
    
    client = _get_client()
    if not client: return "[图片内容]"
    try:
        resp = await client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(text="请用一句话简短但全面地描述这张图片的内容（包括主体、颜色、风格）。"),
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
                ])
            ]
        )
        desc = f"[视觉记忆: {resp.text.strip()}]" if resp.text else "[一张图片]"
        _IMAGE_DESCRIPTION_CACHE[img_hash] = desc
        return desc
    except: return "[图片]"

async def _extract_response_to_message(response: Any) -> Message:
    msg = Message()
    if not response.candidates or not response.candidates[0].content.parts:
        return msg
    for part in response.candidates[0].content.parts:
        if part.text:
            clean_text = re.sub(r"\[CQ:image,[^\]]+\]", "", part.text).strip()
            if clean_text: msg.extend(Message(clean_text))
        elif part.inline_data:
            img_seg = await build_image_segment(part.inline_data.data)
            if img_seg: msg.append(img_seg)
    return msg

def _clean_text_for_memory(msg: Union[str, Message]) -> str:
    if isinstance(msg, Message): return msg.extract_plain_text().strip()
    return re.sub(r"\[CQ:image,[^\]]+\]", "", str(msg)).strip()

async def _summarize_history(session_id: str, history: List[Any]) -> str:
    """对当前历史记录进行摘要压缩"""
    client = _get_client()
    if not client: return ""
    
    formatted_history = []
    for m in history:
        content = str(m.content or "")
        # 简单过滤，不重复生成图片描述以节省 token
        content = re.sub(r"\[视觉记忆: [^\]]+\]", "[图片]", content)
        formatted_history.append(f"{m.role}: {content}")
    
    history_text = "\n".join(formatted_history)
    old_summary = memory_core.get_history_summary(session_id) or "暂无旧摘要"
    
    prompt = (
        f"请根据以下历史对话记录和旧的摘要，生成一段简洁的【对话摘要】。\n"
        f"要求：保留重要的事实、用户的偏好、以及对话的当前进度。字数控制在 500 字以内。\n\n"
        f"【旧摘要】: {old_summary}\n\n"
        f"【最新对话记录】:\n{history_text}\n\n"
        f"请直接输出新的摘要内容："
    )
    
    try:
        resp = await client.aio.models.generate_content(
            model=config.gemini_text_model,
            contents=prompt
        )
        return resp.text.strip() if resp.text else ""
    except Exception as e:
        logger.error(f"摘要生成失败: {e}")
        return ""

async def handle_user_message(bot: Any, event: Any, session_id: str, user_id: str, text: str, already_added: bool = False):
    from nonebot_plugin_skills import _send_reply
    client = _get_client()
    if not client: return

    # --- 1. 处理当前输入 ---
    current_image_parts = []
    message = getattr(event, "message", Message())
    for seg in message:
        if seg.type == "image":
            img_url = seg.data.get("url")
            if img_url:
                img_data = await _get_image_data(img_url)
                if img_data: current_image_parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))

    input_text = text if text.strip() else "(召唤嘉然)"
    reply_obj = getattr(event, "reply", None)
    quoted_id = str(reply_obj.message_id) if reply_obj else None
    if quoted_id: input_text = f"（回复了消息 ID: {quoted_id}）{input_text}"

    if not already_added: 
        memory_core.add_message(session_id, "user", str(event.get_message()), message_id=getattr(event, "message_id", None))

    active_model = config.gemini_image_model if current_image_parts else config.gemini_text_model
    context_prompt = memory_core.get_context_prompt(session_id, user_id)
    history_summary = memory_core.get_history_summary(session_id)
    
    summary_part = f"\n--- 历史对话摘要 ---\n{history_summary}\n" if history_summary else ""
    system_instruction = f"{_build_system_prompt(session_id)}\n--- 记忆 ---\n{context_prompt}{summary_part}"

    # --- 2. 视觉索引历史构建 ---
    history = memory_core.get_history(session_id)
    contents = []
    
    for i, m in enumerate(history):
        text_content = str(m.content or "(空)")
        prefix_parts = []
        if m.message_id:
            prefix_parts.append(f"[消息ID: {m.message_id}]")
        if getattr(m, "recalled_message_id", None):
            prefix_parts.append(f"[撤回消息ID: {m.recalled_message_id}]")
        prefix = f"{' '.join(prefix_parts)} " if prefix_parts else ""

        text_content, history_images = await _render_history_text_with_image_descriptions(text_content)
        parts = [types.Part.from_text(text=f"{prefix}{text_content[:1000]}")]

        # 只有在【当前新图】或【被引用消息的原图】时，才注入原始像素
        if i == len(history) - 1 and m.role == "user" and current_image_parts:
            parts.extend(current_image_parts)
        elif quoted_id and m.message_id == quoted_id:
            for file_val in extract_image_sources(m.content)[:1]:
                img_data = history_images.get(file_val) or await _load_image_bytes(file_val)
                if img_data:
                    parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
                    active_model = config.gemini_image_model

        contents.append(types.Content(role=m.role, parts=parts))

    if not contents or contents[-1].role != "user":
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=str(input_text))] + current_image_parts))

    tools = skill_manager.get_llm_tools()
    count_config = types.CountTokensConfig()
    config_obj = types.GenerateContentConfig(system_instruction=system_instruction, tools=tools, temperature=config.chat_style_temperature)

    # --- 3. 主动 Token 检查与预防性压缩 ---
    try:
        # 预估当前上下文 Token 数
        token_count_resp = await client.aio.models.count_tokens(
            model=active_model,
            contents=contents,
            config=count_config,
        )
        current_tokens = token_count_resp.total_tokens
        
        # 设定主动压缩阈值 (例如 30,000 tokens，对于 Flash 模型这已经包含相当多历史了)
        # 您可以根据需求调整这个值，或者从 config 中读取
        PREACTIVE_COMPRESSION_THRESHOLD = 30000
        
        if current_tokens is not None and current_tokens > PREACTIVE_COMPRESSION_THRESHOLD:
            logger.info(f"Session {session_id} tokens ({current_tokens}) exceed threshold. Proactively compressing...")
            new_summary = await _summarize_history(session_id, history)
            if new_summary:
                memory_core.set_history_summary(session_id, new_summary)
                memory_core.clear_history(session_id)
                # 递归调用一次以使用新摘要重新构造 contents (只会触发一次，因为 clear 后 tokens 必降)
                return await handle_user_message(bot, event, session_id, user_id, text, already_added=True)
    except Exception as token_e:
        logger.warning(f"Failed to count tokens: {token_e}")

    try:
        response = await client.aio.models.generate_content(model=active_model, contents=contents, config=config_obj)
        full_reply = Message()
        while response.function_calls:
            interim_msg = await _extract_response_to_message(response)
            if interim_msg:
                await _send_reply(bot, event, interim_msg)
                full_reply.extend(interim_msg)

            candidate_content = None
            if response.candidates and response.candidates[0].content:
                raw_content = response.candidates[0].content
                if raw_content.parts:
                    candidate_content = types.Content(
                        role=raw_content.role or "model",
                        parts=[types.Part(part) for part in raw_content.parts],
                    )
            if candidate_content:
                contents.append(candidate_content)

            skill_ctx = SkillContext(bot=bot, event=event, session_id=session_id, user_id=user_id, raw_text=text)
            response_parts = []
            for call in response.function_calls:
                skill_name = str(call.name or "").strip()
                raw_skill_args = call.args or {}
                skill_args: Dict[str, Any] = (
                    raw_skill_args if isinstance(raw_skill_args, dict) else dict(raw_skill_args)
                )
                skill_result = await skill_manager.execute(skill_name, context=skill_ctx, **skill_args)
                if isinstance(skill_result, (bytes, MessageSegment)) or "[CQ:image" in str(skill_result):
                    img_seg = await build_image_segment(skill_result)
                    if img_seg:
                        await _send_reply(bot, event, Message(img_seg))
                        full_reply.append(img_seg)
                    clean_res = await _describe_image_result(skill_result)
                else: clean_res = str(skill_result)

                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call.id,
                            name=skill_name,
                            response={"result": clean_res},
                        )
                    )
                )
            if response_parts:
                contents.append(types.Content(role="user", parts=response_parts))
            response = await client.aio.models.generate_content(model=active_model, contents=contents, config=config_obj)

        final_msg = await _extract_response_to_message(response)
        if final_msg:
            await _send_reply(bot, event, final_msg)
            full_reply.extend(final_msg)
        memory_core.add_message(session_id, "model", str(full_reply))
    except Exception as e:
        if "token count exceeds" in str(e):
            logger.warning(f"Session {session_id} token count exceeded. Compressing history...")
            
            # 1. 生成摘要
            new_summary = await _summarize_history(session_id, history)
            if new_summary:
                memory_core.set_history_summary(session_id, new_summary)
            
            # 2. 清空历史记录 (摘要已存入 memory_core)
            memory_core.clear_history(session_id)
            
            # 3. 自动重试一次 (使用新摘要)
            try:
                await handle_user_message(bot, event, session_id, user_id, text, already_added=True)
            except Exception as retry_e:
                logger.error(f"Retry after compression failed: {retry_e}")
                await _send_reply(bot, event, Message("嘉然记忆压缩失败了... 咱们清空记忆重新开始吧~"))
        else:
            logger.error(f"NLP 处理错误: {e}")
            await _send_reply(bot, event, Message(f"嘉然刚才遇到了一点小状况呢... {str(e)}"))
