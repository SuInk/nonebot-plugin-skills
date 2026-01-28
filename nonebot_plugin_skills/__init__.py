from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from google import genai
from google.genai import types
from nonebot import get_driver, get_plugin_config, logger, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent, MessageSegment
from nonebot.exception import FinishedException
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata
from pydantic import BaseModel

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-skills",
    description="基于 Gemini 的头像/图片处理与聊天插件，支持上下文缓存与群/私聊隔离",
    usage="指令：处理头像 <指令> 或 技能/聊天 <内容>",
    type="application",
    homepage="https://github.com/yourname/nonebot-plugin-skills",
    supported_adapters={"~onebot.v11"},
)


class Config(BaseModel):
    google_api_key: str = ""
    gemini_text_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-2.5-flash-image"
    request_timeout: float = 30.0
    image_timeout: float = 120.0
    history_ttl_sec: int = 600
    history_max_messages: int = 20
    gemini_log_response: bool = False
    nlp_enable: bool = True
    bot_keywords: List[str] = []


config = get_plugin_config(Config)
SKILLS_PATH = Path(__file__).with_name("SKILLS.md")


def _mask_api_key(text: str) -> str:
    if not config.google_api_key:
        return text
    return text.replace(config.google_api_key, "***")


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _safe_error_message(exc: Exception) -> str:
    detail = str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        response_text = _truncate(exc.response.text)
        detail = f"{detail} | response: {response_text}"
    detail = _mask_api_key(detail)
    if detail:
        return detail
    return f"{type(exc).__name__}: 未知错误"


def _redact_large_data(value: object, depth: int = 0) -> object:
    if depth > 4:
        return "..."
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, val in value.items():
            if key == "data" and isinstance(val, (bytes, str)):
                size = len(val)
                unit = "bytes" if isinstance(val, bytes) else "chars"
                result[key] = f"<{size} {unit}>"
            else:
                result[key] = _redact_large_data(val, depth + 1)
        return result
    if isinstance(value, list):
        trimmed = value[:20]
        result_list = [_redact_large_data(item, depth + 1) for item in trimmed]
        if len(value) > 20:
            result_list.append("...")
        return result_list
    return value


def _dump_response(response: object) -> str:
    for attr in ("model_dump", "to_dict"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                data = method()
                redacted = _redact_large_data(data)
                return json.dumps(redacted, ensure_ascii=True)
            except Exception:
                pass
    try:
        text = str(response)
    except Exception:
        text = repr(response)
    return _truncate(_mask_api_key(text), 1200)


def _log_response_text(prefix: str, response: object) -> None:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        logger.info("{}: {}", prefix, _truncate(_mask_api_key(text), 1200))


def _load_skills_text() -> str:
    if SKILLS_PATH.exists():
        try:
            return SKILLS_PATH.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read SKILLS.md: %s", exc)
    return (
        "[chat]\n"
        "- 使用中文简洁回复，必要时追问澄清\n"
        "- 结合上下文，不重复罗嗦\n"
        "[image]\n"
        "- 将用户需求改写为清晰的图像编辑指令\n"
        "- 保持主体身份与结构，除非用户明确要求变更\n"
        "- 只输出图像编辑指令，不要解释\n"
    )


SKILLS_TEXT = _load_skills_text()


def _split_skills(text: str) -> Tuple[str, str]:
    chat_lines: List[str] = []
    image_lines: List[str] = []
    mode = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "[chat]":
            mode = "chat"
            continue
        if line.lower() == "[image]":
            mode = "image"
            continue
        if mode == "chat":
            chat_lines.append(line.lstrip("- ").strip())
        elif mode == "image":
            image_lines.append(line.lstrip("- ").strip())
    return "\n".join(chat_lines).strip(), "\n".join(image_lines).strip()


CHAT_SKILLS, IMAGE_SKILLS = _split_skills(SKILLS_TEXT)


@dataclass
class HistoryItem:
    role: str
    text: str
    ts: float


@dataclass
class SessionState:
    history: List[HistoryItem]
    last_image_url: Optional[str]


_SESSIONS: dict[str, SessionState] = {}
_CLIENT: Optional[genai.Client] = None


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group:{event.group_id}"
    return f"private:{event.get_user_id()}"


def _now() -> float:
    return time.time()


def _get_state(session_id: str) -> SessionState:
    state = _SESSIONS.get(session_id)
    if state is None:
        state = SessionState(history=[], last_image_url=None)
        _SESSIONS[session_id] = state
    return state


def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        if not config.google_api_key:
            raise RuntimeError("未配置 GOOGLE_API_KEY")
        _CLIENT = genai.Client(api_key=config.google_api_key)
    return _CLIENT


def _prune_state(state: SessionState) -> None:
    ttl = max(30, int(config.history_ttl_sec))
    cutoff = _now() - ttl
    state.history = [item for item in state.history if item.ts >= cutoff]
    if len(state.history) > config.history_max_messages:
        state.history = state.history[-config.history_max_messages :]


def _extract_first_image_url(message: Message) -> Optional[str]:
    for seg in message:
        if seg.type == "image":
            url = seg.data.get("url") or seg.data.get("file")
            if url:
                return url
    return None


def _extract_at_user(message: Message) -> Optional[str]:
    for seg in message:
        if seg.type == "at":
            qq = seg.data.get("qq")
            if qq and qq != "all":
                return str(qq)
    return None


def _avatar_url(qq: str) -> str:
    return f"http://q.qlogo.cn/headimg_dl?dst_uin={qq}&spec=640"


def _history_to_gemini(state: SessionState) -> List[types.Content]:
    contents: List[types.Content] = []
    for item in state.history:
        contents.append(
            types.Content(
                role=item.role,
                parts=[types.Part.from_text(text=item.text)],
            )
        )
    return contents


def _iter_response_parts(response: object) -> List[object]:
    parts: List[object] = []
    candidates = getattr(response, "candidates", None)
    if candidates:
        for cand in candidates:
            content = getattr(cand, "content", None)
            cand_parts = getattr(content, "parts", None) if content else None
            if cand_parts:
                parts.extend(cand_parts)
    if not parts:
        direct_parts = getattr(response, "parts", None)
        if direct_parts:
            parts.extend(direct_parts)
    return parts


def _extract_inline_data(part: object) -> Optional[object]:
    if isinstance(part, dict):
        return part.get("inline_data") or part.get("inlineData")
    return getattr(part, "inline_data", None)


def _extract_text_value(part: object) -> Optional[str]:
    if isinstance(part, dict):
        value = part.get("text")
        return value if isinstance(value, str) else None
    value = getattr(part, "text", None)
    return value if isinstance(value, str) else None


async def _call_gemini_text(prompt: str, state: SessionState) -> str:
    client = _get_client()
    contents = _history_to_gemini(state)
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    )

    config_obj = (
        types.GenerateContentConfig(system_instruction=CHAT_SKILLS) if CHAT_SKILLS else None
    )
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_text_model,
            contents=contents,
            config=config_obj,
        ),
        timeout=config.request_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini text response: {}", _dump_response(response))
        _log_response_text("Gemini text content", response)
    if response.text:
        return response.text.strip()
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        if getattr(part, "text", None):
            text_parts.append(getattr(part, "text"))
    return "\n".join(text_parts).strip() or "（没有生成到有效回复）"


async def _download_image_bytes(url: str) -> Tuple[str, bytes]:
    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg")
        data = resp.content
    return content_type, data


async def _call_gemini_image(prompt: str, image_url: str, state: SessionState) -> Tuple[bool, str]:
    client = _get_client()
    content_type, image_bytes = await _download_image_bytes(image_url)
    contents = _history_to_gemini(state)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"{IMAGE_SKILLS}\n用户需求：{prompt}"),
                types.Part.from_bytes(data=image_bytes, mime_type=content_type),
            ],
        )
    )

    config_obj = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=contents,
            config=config_obj,
        ),
        timeout=config.image_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini image response: {}", _dump_response(response))
        _log_response_text("Gemini image content", response)

    for part in _iter_response_parts(response):
        inline_data = _extract_inline_data(part)
        text_value = _extract_text_value(part)
        if inline_data:
            if isinstance(inline_data, dict):
                data = inline_data.get("data")
            else:
                data = getattr(inline_data, "data", None)
            if isinstance(data, bytes):
                return True, base64.b64encode(data).decode("ascii")
            if isinstance(data, str):
                return True, data
        if text_value:
            return False, text_value
    if getattr(response, "text", None):
        return False, getattr(response, "text")
    raise RuntimeError("未获取到有效图片结果")


def _image_segment_from_result(result: str) -> MessageSegment:
    if not result:
        raise RuntimeError("图片结果为空")
    if result.startswith("http://") or result.startswith("https://"):
        return MessageSegment.image(result)
    if result.startswith("base64://"):
        return MessageSegment.image(result)
    if result.startswith("data:image"):
        return MessageSegment.image(result)
    return MessageSegment.image(f"base64://{result}")


def _append_history(state: SessionState, role: str, text: str) -> None:
    state.history.append(HistoryItem(role=role, text=text, ts=_now()))
    _prune_state(state)


history_collector = on_message(priority=99, block=False)
nlp_handler = on_message(priority=15, block=False)
avatar_handler = on_command("处理头像", priority=5)
chat_handler = on_command("技能", aliases={"聊天", "对话"}, priority=5)


@history_collector.handle()
async def _collect_history(event: MessageEvent):
    session_id = _session_id(event)
    state = _get_state(session_id)

    text = event.get_plaintext().strip()
    image_url = _extract_first_image_url(event.get_message())
    if image_url:
        state.last_image_url = image_url

    if text:
        _append_history(state, "user", text)


def _is_command_message(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    try:
        starts = list(get_driver().config.command_start or [])
    except Exception:
        starts = ["/"]
    if not starts:
        return False
    command_words = ["处理头像", "技能", "聊天", "对话"]
    for prefix in starts:
        if not prefix:
            continue
        for word in command_words:
            if text.startswith(prefix + word):
                return True
    return False


def _match_keyword(text: str) -> Optional[str]:
    for kw in config.bot_keywords:
        if kw and kw in text:
            return kw
    return None


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


async def _classify_intent(
    text: str,
    state: SessionState,
    has_image: bool,
    at_user: Optional[str],
) -> Optional[dict]:
    if not config.google_api_key:
        return None
    client = _get_client()
    system = (
        "你是消息意图解析器，只输出 JSON，不要解释或补充说明。"
        "不要输出拒绝/免责声明/权限说明（例如“我无法访问账号”）。"
        "严格输出如下 JSON："
        "{"
        "\"action\": \"image|chat|ignore\","
        "\"target\": \"message_image|at_user|last_image|sender_avatar\","
        "\"instruction\": \"string\""
        "}"
        "说明："
        "- action=image 表示要处理图片/头像；instruction 为图片编辑指令（风格、效果等）。"
        "- action=chat 表示普通聊天；instruction 为要回复的文本。"
        "- action=ignore 表示不处理；instruction 为空字符串。"
        "- target 仅在 action=image 时使用："
        "  message_image=本消息里的图片；"
        "  at_user=@用户头像；"
        "  last_image=聊天记录里最近图片；"
        "  sender_avatar=发送者头像。"
    )
    user_prompt = (
        f"文本: {text}\n"
        f"消息包含图片: {has_image}\n"
        f"是否@用户: {bool(at_user)}\n"
        f"是否有最近图片: {bool(state.last_image_url)}\n"
    )
    config_obj = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
    )
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_text_model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
            config=config_obj,
        ),
        timeout=config.request_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini intent response: {}", _dump_response(response))
        _log_response_text("Gemini intent content", response)
    payload = _extract_json(response.text or "")
    return payload


@nlp_handler.handle()
async def _handle_natural_language(bot: Bot, event: MessageEvent):
    if not config.nlp_enable:
        return
    text = event.get_plaintext().strip()
    if not text:
        return
    if _is_command_message(text):
        return
    if str(event.get_user_id()) == str(event.self_id):
        return

    keyword = _match_keyword(text)
    if not keyword:
        return

    session_id = _session_id(event)
    state = _get_state(session_id)
    image_url = _extract_first_image_url(event.get_message())
    at_user = _extract_at_user(event.get_message())
    has_image = image_url is not None

    try:
        intent = await _classify_intent(text, state, has_image, at_user)
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        return

    if not intent:
        return

    action = str(intent.get("action", "ignore")).lower()
    if action == "ignore":
        return

    if action == "chat":
        prompt = intent.get("instruction") or text
        try:
            reply = await _call_gemini_text(str(prompt), state)
            _append_history(state, "user", str(prompt))
            _append_history(state, "model", reply)
            await nlp_handler.send(reply)
        except Exception as exc:
            logger.error("NLP chat failed: {}", _safe_error_message(exc))
        return

    if action != "image":
        return

    prompt = str(intent.get("instruction") or text)
    target = str(intent.get("target") or "").lower()
    if not image_url:
        if target == "message_image":
            image_url = None
        elif target == "at_user" and at_user:
            image_url = _avatar_url(at_user)
        elif target == "last_image" and state.last_image_url:
            image_url = state.last_image_url
        else:
            image_url = _avatar_url(event.get_user_id())

    if not image_url:
        await nlp_handler.send("未找到可处理的图片或头像。")
        return

    await nlp_handler.send("正在生成图片，请稍候...")
    try:
        is_image, result = await _call_gemini_image(prompt, image_url, state)
        _append_history(state, "user", f"处理头像：{prompt}")
        if is_image:
            _append_history(state, "model", "[已生成图片]")
            await nlp_handler.send(_image_segment_from_result(result))
        else:
            _append_history(state, "model", result)
            await nlp_handler.send(f"模型返回了文本结果：\n{result}")
    except Exception as exc:
        logger.error("NLP image failed: {}", _safe_error_message(exc))
        await nlp_handler.send(f"出错了：{_safe_error_message(exc)}")


@avatar_handler.handle()
async def handle_avatar(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    session_id = _session_id(event)
    state = _get_state(session_id)

    prompt = args.extract_plain_text().strip()
    if not prompt:
        await avatar_handler.finish("请告诉我你想怎么处理头像，例如：处理头像 变成赛博朋克风")

    image_url = _extract_first_image_url(event.get_message())
    if not image_url:
        at_user = _extract_at_user(event.get_message())
        if at_user:
            image_url = _avatar_url(at_user)
        elif state.last_image_url:
            image_url = state.last_image_url
        else:
            image_url = _avatar_url(event.get_user_id())

    await avatar_handler.send("正在生成图片，请稍候...")
    try:
        is_image, result = await _call_gemini_image(prompt, image_url, state)
        _append_history(state, "user", f"处理头像：{prompt}")
        if is_image:
            _append_history(state, "model", "[已生成图片]")
            await avatar_handler.finish(_image_segment_from_result(result))
        else:
            _append_history(state, "model", result)
            await avatar_handler.finish(f"模型返回了文本结果：\n{result}")
    except FinishedException:
        raise
    except Exception as exc:
        logger.error("Avatar handler failed: {}", _safe_error_message(exc))
        await avatar_handler.finish(f"出错了：{_safe_error_message(exc)}")


@chat_handler.handle()
async def handle_chat(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    session_id = _session_id(event)
    state = _get_state(session_id)
    prompt = args.extract_plain_text().strip()
    if not prompt:
        await chat_handler.finish("请发送要聊天的内容，例如：聊天 你好")

    try:
        reply = await _call_gemini_text(prompt, state)
        _append_history(state, "user", prompt)
        _append_history(state, "model", reply)
        await chat_handler.finish(reply)
    except FinishedException:
        raise
    except Exception as exc:
        logger.error("Chat handler failed: {}", _safe_error_message(exc))
        await chat_handler.finish(f"出错了：{_safe_error_message(exc)}")
