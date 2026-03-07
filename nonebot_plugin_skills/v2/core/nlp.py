import asyncio
import re
import base64
import io
import hashlib
from typing import List, Optional, Any, Tuple, Dict, Union
from nonebot import logger
from google import genai
from google.genai import types
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.memory import memory_core
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.http import get_http_client
from nonebot_plugin_skills.v2.core.utils import build_image_segment

from nonebot_plugin_skills.config import config

# 图片描述缓存，防止重复请求视觉模型分析同一张图
_IMAGE_DESCRIPTION_CACHE: Dict[str, str] = {}

# --- 核心人设与回复规则 ---
SYSTEM_PROMPT = (
    "你是嘉然(Diana)，A-SOUL成员。性格温柔、体贴、元气满满。\n"
    "【核心回复规则】\n"
    "1. 严禁瞎编！历史记录中包含图片描述。如果你需要对历史图进行修改、重绘或极其精细的细节分析，请务必调用对应的【工具】去获取原图像素。\n"
    "2. 当需要处理图片任务时，优先调用工具。\n"
    "3. 你具有艾特(@)他人的能力：[CQ:at,qq={user_id}]。\n"
    "4. 你的历史记录带 [ID: xxx] 标记；如果你要引用其中某条消息，直接输出对应的 [ID: xxx]，系统会自动转成真正的引用回复。"
)

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
    sys_prompt = SYSTEM_PROMPT.replace('{user_id}', str(user_id))
    system_instruction = f"{sys_prompt}\n--- 记忆 ---\n{context_prompt}"

    # --- 2. 视觉索引历史构建 ---
    history = memory_core.get_history(session_id)
    contents = []
    
    for i, m in enumerate(history):
        text_content = str(m.content or "(空)")
        prefix = f"[ID: {m.message_id}] " if m.message_id else ""
        
        # 识别历史中的图片并转换为描述
        img_matches = re.findall(r"\[CQ:image,file=([^,\]]+)\]", text_content)
        for file_val in img_matches:
            img_data = None
            if file_val.startswith("base64://"):
                try: img_data = base64.b64decode(file_val.replace("base64://", ""))
                except: pass
            else: img_data = await _get_image_data(file_val)
            
            if img_data:
                # 核心逻辑：普通历史记录只传【文字描述】节省 Token
                desc = await _get_image_description(img_data)
                text_content = text_content.replace(f"[CQ:image,file={file_val}]", desc)
        
        parts = [types.Part.from_text(text=f"{prefix}{text_content[:1000]}")]
        
        # 只有在【当前新图】或【被引用消息的原图】时，才注入原始像素
        if i == len(history) - 1 and m.role == "user" and current_image_parts:
            parts.extend(current_image_parts)
        elif quoted_id and m.message_id == quoted_id:
            for file_val in img_matches[:1]:
                img_data = await _get_image_data(file_val) if not file_val.startswith("base64://") else base64.b64decode(file_val.replace("base64://", ""))
                if img_data:
                    parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
                    active_model = config.gemini_image_model

        contents.append(types.Content(role=m.role, parts=parts))

    if not contents or contents[-1].role != "user":
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=str(input_text))] + current_image_parts))

    tools = skill_manager.get_llm_tools()
    config_obj = types.GenerateContentConfig(system_instruction=system_instruction, tools=tools, temperature=config.chat_style_temperature)

    try:
        response = await client.aio.models.generate_content(model=active_model, contents=contents, config=config_obj)
        full_reply = Message()
        while response.function_calls:
            interim_msg = await _extract_response_to_message(response)
            if interim_msg:
                await _send_reply(bot, event, interim_msg)
                full_reply.extend(interim_msg)

            skill_ctx = SkillContext(bot=bot, event=event, session_id=session_id, user_id=user_id, raw_text=text)
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
                    clean_res = "[系统：图已发出]"
                else: clean_res = str(skill_result)
                
                contents.append(types.Content(role="model", parts=[types.Part.from_function_call(name=skill_name, args=skill_args)]))
                contents.append(types.Content(role="user", parts=[types.Part.from_function_response(name=skill_name, response={"result": clean_res})]))
            response = await client.aio.models.generate_content(model=active_model, contents=contents, config=config_obj)

        final_msg = await _extract_response_to_message(response)
        if final_msg:
            await _send_reply(bot, event, final_msg)
            full_reply.extend(final_msg)
        memory_core.add_message(session_id, "model", str(full_reply))
    except Exception as e:
        if "token count exceeds" in str(e):
            memory_core.clear_history(session_id)
            await _send_reply(bot, event, Message("嘉然脑袋过载了呜呜... 记忆已清空，咱们重新聊吧~"))
        else:
            logger.error(f"NLP 处理错误: {e}")
            await _send_reply(bot, event, Message(f"嘉然刚才遇到了一点小状况呢... {str(e)}"))
