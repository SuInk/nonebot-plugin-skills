import asyncio
from typing import Optional, List, Union, Tuple
from google import genai
from google.genai import types
from nonebot import logger
from nonebot.adapters.onebot.v11 import Message
import re
import random
import base64

from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.http import get_http_client
from nonebot_plugin_skills.v2.core.utils import build_image_segment
from nonebot_plugin_skills.config import config

image_schema = {
    "properties": {
        "prompt": {"type": "STRING", "description": "详细的描述词、修改指令或你想问的问题。"},
        "target_qq": {"type": "STRING", "description": "可选：操作目标的 QQ 号。"},
        "is_group": {"type": "BOOLEAN", "description": "是否涉及群头像。"}
    },
    "required": ["prompt"]
}


def _extract_first_inline_image_bytes(response: object) -> Optional[bytes]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None

    first_candidate = candidates[0]
    content = getattr(first_candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None)
        if data:
            return data

    return None

def _get_client() -> Optional[genai.Client]:
    if not config.google_api_key: return None
    return genai.Client(api_key=config.google_api_key)

async def _get_image_bytes(url: str) -> Optional[bytes]:
    client = get_http_client()
    try:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

async def _find_relevant_image(context: SkillContext, is_group: bool = False, target_qq: Optional[str] = None) -> Tuple[Optional[bytes], str]:
    """智能搜寻最相关的图片源"""
    # 1. 引用
    reply = getattr(context.event, "reply", None)
    if reply:
        for seg in reply.message:
            if seg.type == "image":
                url = seg.data.get("url")
                if url:
                    data = await _get_image_bytes(url)
                    if data: return data, "被引用消息里的图片"

    # 2. 当前消息
    message = getattr(context.event, "message", Message())
    for seg in message:
        if seg.type == "image":
            url = seg.data.get("url")
            if url:
                data = await _get_image_bytes(url)
                if data: return data, "这张图片"

    # 3. 历史追溯 (20条)
    from nonebot_plugin_skills.v2.core.memory import memory_core
    history = memory_core.get_history(context.session_id)
    for m in reversed(history[-20:]):
        match = re.search(r"\[CQ:image,file=([^,\]]+)\]", m.content)
        if match:
            file_val = match.group(1)
            from nonebot_plugin_skills.v2.core.nlp import _get_image_data
            data = None
            if file_val.startswith("base64://"):
                try: data = base64.b64decode(file_val.replace("base64://", ""))
                except: pass
            else: data = await _get_image_data(file_val)
            if data: return data, "刚才看到的那张图片"

    # 4. 头像
    if is_group:
        group_id = getattr(context.event, "group_id", None)
        if group_id:
            url = f"https://p.qlogo.cn/gh/{group_id}/{group_id}/640"
            data = await _get_image_bytes(url)
            if data: return data, "本群的头像"
    
    if target_qq:
        url = f"https://q1.qlogo.cn/g?b=qq&nk={target_qq}&s=640"
        data = await _get_image_bytes(url)
        if data: return data, f"用户 {target_qq} 的头像"

    return None, ""

@skill_manager.register("modify_existing_image", "【图片修改】使用 AI 修改现有的图片、QQ头像或群头像。可以直接改变内容或风格。", image_schema)
async def modify_existing_image(prompt: str, target_qq: Optional[str] = None, is_group: bool = False, context: Optional[SkillContext] = None) -> Union[str, bytes]:
    client = _get_client()
    if not client or not context: return "无法连接 AI。"

    source_bytes, source_name = await _find_relevant_image(context, is_group, target_qq)
    if not source_bytes: return "唔... 嘉然不知道要改哪张图呢。你可以先发一张图给我哦~"

    try:
        response = await client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_bytes(data=source_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=f"Please modify this image according to: {prompt}. Output the modified image data directly.")
                ])
            ]
        )
        image_bytes = _extract_first_inline_image_bytes(response)
        if image_bytes:
            return image_bytes
        return "改图失败了呢..."
    except Exception as e:
        return f"出错了: {e}"

@skill_manager.register("chat_about_image", "分析某张图片，回答细节问题或进行闲聊。", image_schema)
async def chat_about_image(prompt: str, target_qq: Optional[str] = None, is_group: bool = False, context: Optional[SkillContext] = None) -> str:
    client = _get_client()
    if not client or not context: return "引擎未就绪。"

    source_bytes, source_name = await _find_relevant_image(context, is_group, target_qq)
    if not source_bytes: return "嘉然没看到你想聊哪张图呢。"

    try:
        response = await client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_bytes(data=source_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt)
                ])
            ]
        )
        return response.text or "嘉然看了一眼，但不知道怎么说..."
    except Exception as e:
        return f"分析失败: {e}"

@skill_manager.register("create_new_image", "【图片生成】根据您的描述生成一张全新的高清图片。", image_schema)
async def create_new_image(prompt: str, context: Optional[SkillContext] = None) -> Union[str, bytes]:
    client = _get_client()
    if not client: return "未配置 API。"
    try:
        # 直接使用 3.1 生成图片内容
        response = await client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=f"Generate a high quality image: {prompt}")])]
        )
        image_bytes = _extract_first_inline_image_bytes(response)
        if image_bytes:
            return image_bytes
        return "嘉然画不出来呢..."
    except Exception as e:
        return f"画图失败: {e}"
