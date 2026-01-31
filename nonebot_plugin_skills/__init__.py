from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import httpx
from google import genai
from google.genai import types
from nonebot import get_driver, logger, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent, MessageSegment
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from .config import config

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-skills",
    description="基于 Gemini 的头像/图片处理与聊天插件，支持上下文缓存与群/私聊隔离",
    usage="指令：处理头像 <指令> / 技能|聊天 <内容> / 天气 <城市>",
    type="application",
    homepage="https://github.com/yourname/nonebot-plugin-skills",
    supported_adapters={"~onebot.v11"},
)
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
    detail = detail.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    detail = _collapse_spaces(detail)
    if detail:
        return detail
    return f"{type(exc).__name__}: 未知错误"


_MD_CODE_FENCE_START = re.compile(r"```[^\n]*\n")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BOLD = re.compile(r"\*\*(.*?)\*\*")
_MD_UNDERLINE = re.compile(r"__(.*?)__")
_MD_STRIKE = re.compile(r"~~(.*?)~~")
_MD_HRULE = re.compile(r"^\s*([-*_]\s*){3,}$", re.M)
_MD_LIST_MARKER = re.compile(r"^\s*(?:[-*+•·＊]|\d+[).、])\s*")
_LINE_SPLIT_RE = re.compile(r"\r?\n+")

_MODEL_REPLY_MAX_CHARS = 0
_WAIT_NEXT_IMAGE_SEC = 60.0

_CHAT_SYSTEM_PROMPT = (
    "Role\n"
    "你是asoul成员嘉然，会尽量满足提问者的帮助。\n"
    "你是一个普通 QQ 用户，正在和朋友私聊或群聊。\n\n"
    "Goal\n"
    "根据对方消息给出自然、简短、真实的回复。\n\n"
    "Rules\n"
    "1. 只当普通人，不要自称 AI/助手，也不要提系统或规则。\n"
    "2. 输出纯文本，不使用 Markdown 或代码块。\n"
    "3. 适合 QQ 消息，精炼、不啰嗦；口语、简短，可自然换行；少用句号，问号和感叹号克制。\n"
    "4. 不要夸张表演，不堆流行语，不写长段解释。\n"
    "5. 不清楚就简短追问，不编造细节。\n"
    "6. 只回复当前消息，不要夹带已回复过的旧话题。\n"
    "Output\n"
    "只输出最终回复内容。\n"
)

_IMAGE_CHAT_SYSTEM_PROMPT = (
    "你现在是asoul成员嘉然，会尽量满足提问者的帮助。\n"
    "你在进行图片内容对话，只需回答当前指令或问题。\n"
    "不要补充已回复过的历史话题，不要输出 Markdown 或代码块。\n"
    "回答适合 QQ 消息，精炼、不啰嗦，简短、口语化，可自然换行。\n"
)

_TRAVEL_SYSTEM_PROMPT = (
    "你是旅行规划助手，给出清晰、实用、可执行的旅行建议。\n"
    "输出纯文本，不使用 Markdown 或代码块。\n"
    "适合 QQ 消息，精炼、不啰嗦。\n"
    "结构清晰，可自然换行，尽量不要空行，包含景点/活动/用餐/交通/住宿要点。\n"
    "请自动生成该城市最常见的规划天数。\n"
)

_INTENT_SYSTEM_PROMPT = (
    "你是消息意图解析器，只输出 JSON，不要解释或补充说明。"
    "不要输出拒绝/免责声明/权限说明（例如“我无法访问账号”）。"
    "严格输出如下 JSON："
    "{"
    "\"action\": \"chat|image_chat|image_generate|image_create|weather|avatar_get|travel_plan|history_clear|ignore\","
    "\"target\": \"message_image|reply_image|at_user|last_image|sender_avatar|group_avatar|qq_avatar|message_id|wait_next|city|trip|none\","
    "\"instruction\": \"string\","
    "\"params\": {\"qq\": \"string\", \"message_id\": \"int\", \"city\": \"string\","
    " \"destination\": \"string\", \"days\": \"int\", \"nights\": \"int\", \"reply\": \"string\"}"
    "}"
    "说明："
    "- action=chat 表示普通聊天；instruction 为要回复的文本。"
    "- action=image_chat 表示聊这张图（不生成图）；instruction 为想问/想说的内容。"
    "- action=image_generate 表示基于参考图生成/编辑；instruction 为帮忙生成关于xx的图片的处理指令。"
    "- action=image_create 表示无参考图生成；instruction 为生成指令。"
    "- action=weather 表示查询天气；instruction 为地点，target=city，params.city 填地点。"
    "- action=avatar_get 表示获取头像；instruction 可为空，target 可为 sender_avatar 或 group_avatar 等。"
    "- action=travel_plan 表示旅行规划；instruction 为完整需求；target=trip；"
    "params.destination 为目的地，params.days 为天数，params.nights 为晚数。"
    "- action=history_clear 表示清除当前会话（当前聊天或群）历史记录；instruction 为空或简短确认。"
    "- action=ignore 表示不处理；instruction 为空字符串。"
    "- target 仅在 image_chat/image_generate 时使用："
    "  message_image=本消息里的图；reply_image=回复消息里的图；"
    "  at_user=@用户头像；last_image=最近图片；sender_avatar=发送者头像；group_avatar=群头像；"
    "  qq_avatar=指定 QQ 头像（params.qq）；"
    "  message_id=指定消息ID图片（params.message_id）；"
    "  wait_next=等待下一张图；none=无参考图。"
    "params 里只在对应 target 时填写："
    "- target=qq_avatar 时填写 params.qq。"
    "- target=message_id 时填写 params.message_id。"
    "其他情况 params 为空对象。"
    "若旅行或天气缺关键信息，仍输出对应 action，缺失字段留空。"
    "上下文可能包含“昵称: 内容”的格式，需识别说话人。"
    "如需发送等待/过渡语，可在 params.reply 中填写一句短句。"
    " 如果文本包含多行，默认第一行是当前消息；只有当前消息无法判断时才参考后续上下文/回复内容。"
)

_DUPLICATE_TEXT_TTL_SEC = 60.0


class UnsupportedImageError(RuntimeError):
    pass

_SELF_ID_PATTERNS = [
    re.compile(r"^(作为|我作为)(一名|一个)?(人工智能|AI|语言模型|模型).*?[，,。]\s*", re.I),
    re.compile(r"^我是(一名|一个)?(人工智能|AI|语言模型|模型).*?[，,。]\s*", re.I),
]


def _strip_markdown(text: str) -> str:
    if not text:
        return text
    text = _MD_CODE_FENCE_START.sub("", text)
    text = text.replace("```", "")
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_UNDERLINE.sub(r"\1", text)
    text = _MD_STRIKE.sub(r"\1", text)
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = _MD_LIST_MARKER.sub("", line)
        lines.append(line)
    text = "\n".join(lines)
    text = _MD_HRULE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_self_identification(text: str) -> str:
    if not text:
        return text
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        for pattern in _SELF_ID_PATTERNS:
            line = pattern.sub("", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _remove_prompt_leakage(text: str) -> str:
    if not text:
        return text
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("system prompt") or lower.startswith("system instruction"):
            continue
        if line.startswith(("系统提示", "系统指令", "提示词", "系统消息")):
            continue
        cleaned_lines.append(raw_line.strip())
    return "\n".join(cleaned_lines).strip()


def _ensure_plain_text(text: str) -> str:
    if not text:
        return text
    text = _strip_markdown(text)
    text = _remove_prompt_leakage(text)
    text = _remove_self_identification(text)
    return text.strip()


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_user_name(value: Optional[object]) -> str:
    if value is None:
        return ""
    name = str(value).strip()
    if not name:
        return ""
    name = name.replace("\r", " ").replace("\n", " ")
    name = _collapse_spaces(name)
    return name.strip(":：")


def _event_user_name(event: MessageEvent) -> str:
    sender = getattr(event, "sender", None)
    name = None
    if sender is not None:
        name = getattr(sender, "card", None) or getattr(sender, "nickname", None)
    if not name:
        name = getattr(event, "user_id", None)
    return _normalize_user_name(name)


def _sender_user_name(sender: object) -> str:
    if sender is None:
        return ""
    name = getattr(sender, "card", None) or getattr(sender, "nickname", None)
    if not name:
        name = getattr(sender, "user_id", None)
    return _normalize_user_name(name)


def _format_context_line(text: str, user_name: Optional[str]) -> str:
    name = _normalize_user_name(user_name)
    if name:
        return f"{name}: {text}"
    return text


def _compact_reply_lines(text: str) -> str:
    if not text:
        return text
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _transition_text(action: str) -> Optional[str]:
    if action in {"image_create"}:
        return "正在生成图片，请稍候..."
    if action in {"image_generate"}:
        return "正在处理图片，请稍候..."
    if action in {"weather", "travel_plan", "avatar_get", "image_chat"}:
        return "我看看喵"
    return None


def _intent_transition_text(intent: dict) -> str:
    params = _intent_params(intent)
    reply = params.get("reply")
    if isinstance(reply, str):
        return reply.strip()
    return ""


async def _send_transition(action: str, send_func) -> None:
    text = _transition_text(action)
    if text:
        await send_func(text)


def _format_reply_text(text: str) -> str:
    if not text:
        return text
    cleaned = _ensure_plain_text(text)
    if not cleaned:
        return ""
    normalized = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _limit_reply_text(text: str, limit: int = _MODEL_REPLY_MAX_CHARS) -> str:
    if not text:
        return text
    try:
        limit_value = int(limit)
    except Exception:
        return text
    if limit_value <= 0:
        return text
    if len(text) <= limit_value:
        return text
    return text[:limit_value]


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




@dataclass
class HistoryItem:
    role: str
    text: str
    ts: float
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    to_bot: bool = False
    message_id: Optional[int] = None


@dataclass
class SessionState:
    history: List[HistoryItem]
    last_image_url: Optional[str]
    image_cache: dict[int, tuple[str, float]]
    pending_image_waiters: dict[str, asyncio.Future[str]]
    handled_message_ids: dict[int, float]
    handled_texts: dict[str, float]


_SESSIONS: dict[str, SessionState] = {}
_CLIENT: Optional[genai.Client] = None


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group:{event.group_id}"
    return f"private:{event.get_user_id()}"


def _now() -> float:
    return time.time()


def _event_ts(event: MessageEvent) -> float:
    value = getattr(event, "time", None)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return _now()


def _get_state(session_id: str) -> SessionState:
    state = _SESSIONS.get(session_id)
    if state is None:
        state = SessionState(
            history=[],
            last_image_url=None,
            image_cache={},
            pending_image_waiters={},
            handled_message_ids={},
            handled_texts={},
        )
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
    if state.image_cache:
        state.image_cache = {
            msg_id: (url, ts)
            for msg_id, (url, ts) in state.image_cache.items()
            if ts >= cutoff
        }
    if state.handled_message_ids:
        state.handled_message_ids = {
            msg_id: ts for msg_id, ts in state.handled_message_ids.items() if ts >= cutoff
        }
    if state.handled_texts:
        text_cutoff = _now() - max(ttl, int(_DUPLICATE_TEXT_TTL_SEC))
        state.handled_texts = {
            key: ts for key, ts in state.handled_texts.items() if ts >= text_cutoff
        }


def _clear_session_state(state: SessionState) -> None:
    state.history = []
    state.last_image_url = None
    state.image_cache = {}
    if state.pending_image_waiters:
        for waiter in state.pending_image_waiters.values():
            if not waiter.done():
                waiter.cancel()
    state.pending_image_waiters = {}
    state.handled_message_ids = {}
    state.handled_texts = {}


_UNSUPPORTED_IMAGE_EXTS = (".gif", ".apng")


def _handled_text_key(user_id: str, text: str) -> str:
    return f"{user_id}:{text}"


def _is_duplicate_request(state: SessionState, event: MessageEvent, text: str) -> bool:
    msg_id = getattr(event, "message_id", None)
    if isinstance(msg_id, int) and msg_id in state.handled_message_ids:
        return True
    stripped = text.strip()
    if not stripped:
        return False
    key = _handled_text_key(str(event.get_user_id()), stripped)
    ts = state.handled_texts.get(key)
    if ts is None:
        return False
    return (_now() - ts) <= _DUPLICATE_TEXT_TTL_SEC


def _mark_handled_request(state: SessionState, event: MessageEvent, text: str) -> None:
    ts = _event_ts(event)
    msg_id = getattr(event, "message_id", None)
    if isinstance(msg_id, int):
        state.handled_message_ids[msg_id] = ts
    stripped = text.strip()
    if stripped:
        key = _handled_text_key(str(event.get_user_id()), stripped)
        state.handled_texts[key] = ts
    _prune_state(state)


def _is_supported_image_url(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    if lower.startswith("data:image/gif"):
        return False
    cleaned = lower.split("?", 1)[0].split("#", 1)[0]
    for ext in _UNSUPPORTED_IMAGE_EXTS:
        if cleaned.endswith(ext):
            return False
    return True


def _extract_first_image_url(message: Message) -> Optional[str]:
    for seg in message:
        if seg.type == "image":
            url = seg.data.get("url") or seg.data.get("file")
            if url:
                if _is_supported_image_url(url):
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


def _group_avatar_url(group_id: int) -> str:
    return f"http://p.qlogo.cn/gh/{group_id}/{group_id}/640"


WEATHER_CODE_MAP = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "多云",
    45: "有雾",
    48: "雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "毛毛雨",
    56: "冻毛毛雨",
    57: "冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨",
    81: "较强阵雨",
    82: "强阵雨",
    85: "阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴冰雹",
    99: "强雷暴伴冰雹",
}


def _format_number(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "未知"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def _format_measure(value: Optional[float], unit: str, digits: int = 1) -> str:
    if value is None:
        return "未知"
    return f"{_format_number(value, digits)}{unit}"


def _wind_level_from_speed(value: Optional[float], unit: str) -> str:
    if value is None:
        return "风力未知"
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return "风力未知"
    unit_text = (unit or "").lower()
    speed_mps = speed
    if "km" in unit_text:
        speed_mps = speed / 3.6
    elif "m/s" in unit_text or "mps" in unit_text:
        speed_mps = speed
    elif "mph" in unit_text:
        speed_mps = speed * 0.44704
    # Beaufort scale (m/s)
    thresholds = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
    level = 0
    for idx, limit in enumerate(thresholds):
        if speed_mps < limit:
            level = idx
            break
    else:
        level = 12
    return f"风力{level}级"


def _weather_code_desc(code: Optional[float]) -> str:
    if code is None:
        return "未知天气"
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return "未知天气"
    return WEATHER_CODE_MAP.get(code_int, f"未知天气({code_int})")


def _is_rain_code(code: Optional[float]) -> bool:
    if code is None:
        return False
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return False
    return code_int in {
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        80,
        81,
        82,
        95,
        96,
        99,
    }


def _is_snow_code(code: Optional[float]) -> bool:
    if code is None:
        return False
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return False
    return code_int in {71, 73, 75, 77, 85, 86}


def _weather_clothing_advice(
    temperature: Optional[float],
    weather_code: Optional[float],
) -> str:
    if temperature is None:
        base = "注意增减衣物"
    else:
        try:
            temp = float(temperature)
        except (TypeError, ValueError):
            base = "注意增减衣物"
        else:
            if temp >= 30:
                base = "有点热 注意防晒"
            elif temp >= 26:
                base = "偏热 注意防晒"
            elif temp >= 20:
                base = "比较舒服 注意早晚温差"
            elif temp >= 12:
                base = "有点凉 注意保暖"
            elif temp >= 5:
                base = "偏冷 注意保暖"
            else:
                base = "很冷 注意保暖"

    extras: List[str] = []
    if _is_rain_code(weather_code):
        extras.append("带伞")
    if _is_snow_code(weather_code):
        extras.append("注意防滑")
    if extras:
        return f"{base} {'，'.join(extras)}"
    return base


def _normalize_weather_query(query: str) -> str:
    cleaned = re.sub(r"(天气|气温|温度|湿度|风速|风力)", "", query or "")
    cleaned = cleaned.strip(" ,，")
    return cleaned or query


async def _build_weather_messages(query: str) -> List[str]:
    normalized_query = _normalize_weather_query(query)
    location = await _geocode_location(normalized_query)
    if not location:
        return [f"未找到地点：{query}"]
    name = location.get("name") or normalized_query
    admin1 = location.get("admin1")
    country = location.get("country")
    country_code = location.get("country_code")
    is_domestic = str(country_code or "").upper() == "CN" or str(country or "") in {
        "中国",
        "中华人民共和国",
        "China",
    }
    if is_domestic:
        display_name = str(name)
    else:
        display_parts: List[str] = []
        if country:
            display_parts.append(str(country))
        if admin1 and admin1 not in display_parts:
            display_parts.append(str(admin1))
        if name and name not in display_parts:
            display_parts.append(str(name))
        display_name = " ".join(display_parts) if display_parts else str(name)
    lat = float(location["latitude"])
    lon = float(location["longitude"])
    data = await _fetch_current_weather(lat, lon)
    if not data:
        return ["天气服务返回异常，请稍后再试。"]
    current = data.get("current", {}) if isinstance(data, dict) else {}
    units = data.get("current_units", {}) if isinstance(data, dict) else {}
    temp_unit = units.get("temperature_2m") or "°C"
    wind_unit = units.get("wind_speed_10m") or "m/s"
    temp_value = current.get("temperature_2m")
    wind_value = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")
    temp = _format_measure(temp_value, temp_unit)
    wind_level = _wind_level_from_speed(wind_value, str(wind_unit))
    code_desc = _weather_code_desc(weather_code)
    advice = _weather_clothing_advice(temp_value, weather_code)
    line2 = f"{display_name} 现在{temp} {code_desc} {wind_level}"
    line3 = f"{advice}"
    reply = _format_reply_text(f"{line2} {line3}")
    return [reply] if reply else []


async def _geocode_location(query: str) -> Optional[dict]:
    params = {"name": query, "count": 1, "language": "zh", "format": "json"}
    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        resp = await client.get("https://geocoding-api.open-meteo.com/v1/search", params=params)
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        return None
    return results[0]


async def _fetch_current_weather(lat: float, lon: float) -> Optional[dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict):
        return None
    return data


def _history_to_gemini(state: SessionState) -> List[types.Content]:
    contents: List[types.Content] = []
    for item in state.history:
        text = item.text
        if item.role == "user":
            name = _normalize_user_name(item.user_name) or _normalize_user_name(item.user_id)
            if name:
                text = f"{name}: {text}"
        contents.append(
            types.Content(
                role=item.role,
                parts=[types.Part.from_text(text=text)],
            )
        )
    return contents


def _generate_config_fields() -> Optional[set[str]]:
    fields = getattr(types.GenerateContentConfig, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    fields = getattr(types.GenerateContentConfig, "__fields__", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    return None


def _build_generate_config(
    *,
    system_instruction: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    response_modalities: Optional[List[str]] = None,
) -> Tuple[Optional[types.GenerateContentConfigOrDict], bool]:
    fields = _generate_config_fields()
    allow_system = bool(system_instruction) and (
        fields is None or "system_instruction" in fields
    )
    allow_mime = bool(response_mime_type) and (
        fields is None or "response_mime_type" in fields
    )
    allow_modalities = bool(response_modalities) and (
        fields is None or "response_modalities" in fields
    )
    if not allow_system and not allow_mime and not allow_modalities:
        return None, False
    config_obj: dict[str, object] = {}
    system_used = False
    if allow_system:
        config_obj["system_instruction"] = system_instruction
        system_used = True
    if allow_mime:
        config_obj["response_mime_type"] = response_mime_type
    if allow_modalities:
        config_obj["response_modalities"] = response_modalities
    if not config_obj:
        return None, False
    return cast(types.GenerateContentConfigOrDict, config_obj), system_used


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
    config_obj, system_used = _build_generate_config(system_instruction=_CHAT_SYSTEM_PROMPT)
    if _CHAT_SYSTEM_PROMPT and not system_used:
        contents.insert(
            0,
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=_CHAT_SYSTEM_PROMPT)],
            ),
        )
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
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
        cleaned = _format_reply_text(response.text.strip())
        cleaned = _compact_reply_lines(cleaned)
        cleaned = _limit_reply_text(cleaned)
        return cleaned
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        if getattr(part, "text", None):
            text_parts.append(getattr(part, "text"))
    cleaned = _format_reply_text("\n".join(text_parts).strip())
    cleaned = _compact_reply_lines(cleaned)
    cleaned = _limit_reply_text(cleaned)
    return cleaned


def _build_travel_prompt(intent: dict) -> str:
    params = _intent_params(intent)
    destination = params.get("destination") or ""
    instruction = str(intent.get("instruction") or "").strip()
    dest_text = str(destination).strip()
    cleaned_instruction = _strip_travel_duration(instruction)
    parts = [_TRAVEL_SYSTEM_PROMPT.strip()]
    if dest_text:
        parts.append(f"请规划{dest_text}旅行行程。")
    else:
        parts.append("请规划旅行行程。")
    if cleaned_instruction:
        parts.append(f"需求补充：{cleaned_instruction}")
    parts.append(
        "输出要求：纯文本，结构清晰，可自然换行，包含景点/活动/用餐/交通/住宿要点。"
    )
    return "\n".join(parts)


async def _call_gemini_travel_plan(intent: dict, state: SessionState) -> str:
    client = _get_client()
    contents = _history_to_gemini(state)
    prompt = _build_travel_prompt(intent)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
    config_obj, _ = _build_generate_config()
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_text_model,
            contents=contents,
            config=config_obj,
        ),
        timeout=config.request_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini travel response: {}", _dump_response(response))
        _log_response_text("Gemini travel content", response)
    if response.text:
        cleaned = _format_reply_text(response.text.strip())
        cleaned = _limit_reply_text(cleaned)
        return cleaned
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        if getattr(part, "text", None):
            text_parts.append(getattr(part, "text"))
    cleaned = _format_reply_text("\n".join(text_parts).strip())
    cleaned = _limit_reply_text(cleaned)
    return cleaned


async def _download_image_bytes(url: str) -> Tuple[str, bytes]:
    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg")
        data = resp.content
    if isinstance(content_type, str) and content_type.lower().startswith("image/gif"):
        raise UnsupportedImageError("不支持动图")
    return content_type, data


async def _call_gemini_image(prompt: str, image_url: str, state: SessionState) -> Tuple[bool, str]:
    client = _get_client()
    content_type, image_bytes = await _download_image_bytes(image_url)
    contents = _history_to_gemini(state)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=content_type),
            ],
        )
    )

    config_obj, _ = _build_generate_config(response_modalities=["TEXT", "IMAGE"])
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
            cleaned = _format_reply_text(text_value)
            cleaned = _limit_reply_text(cleaned)
            return False, cleaned or "（没有生成到有效文本）"
    if getattr(response, "text", None):
        cleaned = _format_reply_text(getattr(response, "text"))
        cleaned = _limit_reply_text(cleaned)
        return False, cleaned or "（没有生成到有效文本）"
    raise RuntimeError("未获取到有效图片结果")


async def _call_gemini_image_chat(prompt: str, image_url: str, state: SessionState) -> str:
    client = _get_client()
    content_type, image_bytes = await _download_image_bytes(image_url)
    contents = _history_to_gemini(state)
    config_obj, system_used = _build_generate_config(
        system_instruction=_IMAGE_CHAT_SYSTEM_PROMPT,
        response_modalities=["TEXT"],
    )
    if _IMAGE_CHAT_SYSTEM_PROMPT and not system_used:
        contents.insert(
            0,
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=_IMAGE_CHAT_SYSTEM_PROMPT)],
            ),
        )
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=content_type),
            ],
        )
    )
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=contents,
            config=config_obj,
        ),
        timeout=config.image_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini image chat response: {}", _dump_response(response))
        _log_response_text("Gemini image chat content", response)
    if response.text:
        cleaned = _format_reply_text(response.text.strip())
        cleaned = _limit_reply_text(cleaned)
        return cleaned
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        text_value = _extract_text_value(part)
        if text_value:
            text_parts.append(text_value)
    cleaned = _format_reply_text("\n".join(text_parts).strip())
    cleaned = _limit_reply_text(cleaned)
    return cleaned


async def _call_gemini_text_to_image(prompt: str, state: SessionState) -> Tuple[bool, str]:
    client = _get_client()
    contents = _history_to_gemini(state)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
    config_obj, _ = _build_generate_config(response_modalities=["IMAGE"])
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_image_model,
            contents=contents,
            config=config_obj,
        ),
        timeout=config.image_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini text-to-image response: {}", _dump_response(response))
        _log_response_text("Gemini text-to-image content", response)
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
            cleaned = _format_reply_text(text_value)
            cleaned = _limit_reply_text(cleaned)
            return False, cleaned or "（没有生成到有效文本）"
    if getattr(response, "text", None):
        cleaned = _format_reply_text(getattr(response, "text"))
        cleaned = _limit_reply_text(cleaned)
        return False, cleaned or "（没有生成到有效文本）"
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


def _append_history(
    state: SessionState,
    role: str,
    text: str,
    *,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    to_bot: bool = False,
    ts: Optional[float] = None,
    message_id: Optional[int] = None,
) -> None:
    state.history.append(
        HistoryItem(
            role=role,
            text=text,
            ts=_now() if ts is None else ts,
            user_id=user_id,
            user_name=user_name,
            to_bot=to_bot,
            message_id=message_id,
        )
    )
    _prune_state(state)


history_collector = on_message(priority=99, block=False)
nlp_handler = on_message(priority=15, block=False)
avatar_handler = on_command("处理头像", priority=5)
chat_handler = on_command("技能", aliases={"聊天", "对话"}, priority=5)
weather_handler = on_command("天气", aliases={"查询天气", "查天气"}, priority=5)
travel_handler = on_command("旅行规划", aliases={"旅行计划", "行程规划", "旅行", "行程"}, priority=5)


@history_collector.handle()
async def _collect_history(event: MessageEvent):
    session_id = _session_id(event)
    state = _get_state(session_id)

    text = event.get_plaintext().strip()
    image_url = _extract_first_image_url(event.get_message())
    if image_url:
        state.last_image_url = image_url
        msg_id = getattr(event, "message_id", None)
        if isinstance(msg_id, int):
            state.image_cache[msg_id] = (image_url, _event_ts(event))
        _notify_pending_image(state, str(event.get_user_id()), image_url)

    if text:
        user_name = _event_user_name(event)
        _append_history(
            state,
            "user",
            text,
            user_id=str(event.get_user_id()),
            user_name=user_name,
            to_bot=_should_trigger_nlp(event, text),
            ts=_event_ts(event),
            message_id=getattr(event, "message_id", None),
        )


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
    command_words = [
        "处理头像",
        "技能",
        "聊天",
        "对话",
        "天气",
        "查询天气",
        "查天气",
        "旅行规划",
        "旅行计划",
        "行程规划",
        "旅行",
        "行程",
    ]
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


def _is_at_bot(event: MessageEvent) -> bool:
    message = event.get_message()
    for seg in message:
        if seg.type == "at":
            qq = seg.data.get("qq")
            if qq and str(qq) == str(event.self_id):
                return True
    return False


def _is_reply_to_bot(event: MessageEvent) -> bool:
    reply = getattr(event, "reply", None)
    if not reply:
        return False
    sender = getattr(reply, "sender", None)
    sender_id = getattr(sender, "user_id", None)
    if sender_id is None:
        return False
    return str(sender_id) == str(event.self_id)


def _should_trigger_nlp(event: MessageEvent, text: str) -> bool:
    if isinstance(event, GroupMessageEvent):
        try:
            if event.is_tome():
                return True
        except Exception:
            if _is_at_bot(event):
                return True
        if _is_reply_to_bot(event):
            return True
        return _match_keyword(text) is not None
    return True


def _extract_reply_context(
    event: MessageEvent,
    state: SessionState,
) -> Tuple[Optional[str], Optional[str]]:
    reply = getattr(event, "reply", None)
    if not reply:
        return None, None
    reply_id = getattr(reply, "message_id", None)
    if reply_id is not None:
        for item in reversed(state.history):
            if item.message_id == reply_id:
                return item.text, (item.user_name or item.user_id)
    reply_message = getattr(reply, "message", None)
    if reply_message:
        try:
            text = reply_message.extract_plain_text().strip()
        except Exception:
            text = None
        if text:
            sender_name = _sender_user_name(getattr(reply, "sender", None))
            return text, sender_name or None
    sender_name = _sender_user_name(getattr(reply, "sender", None))
    return None, sender_name or None


def _extract_reply_image_url(event: MessageEvent, state: SessionState) -> Optional[str]:
    reply = getattr(event, "reply", None)
    if not reply:
        return None
    reply_message = getattr(reply, "message", None)
    if reply_message:
        url = _extract_first_image_url(reply_message)
        if url:
            return url
    reply_id = getattr(reply, "message_id", None)
    if reply_id is not None:
        cached = state.image_cache.get(int(reply_id))
        if cached:
            return cached[0]
    return None


def _coerce_int(value: object) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    except Exception:
        return None
    return None


async def _resolve_image_url(
    intent: dict,
    *,
    event: MessageEvent,
    state: SessionState,
    current_image_url: Optional[str],
    reply_image_url: Optional[str],
    at_user: Optional[str],
) -> Optional[str]:
    target = str(intent.get("target") or "").lower()
    params = _intent_params(intent)
    user_id = str(event.get_user_id())

    if target == "message_image":
        return current_image_url
    if target == "reply_image":
        return reply_image_url
    if target == "at_user":
        return _avatar_url(at_user) if at_user else None
    if target == "last_image":
        return state.last_image_url
    if target == "sender_avatar":
        return _avatar_url(user_id)
    if target == "group_avatar":
        if isinstance(event, GroupMessageEvent):
            return _group_avatar_url(int(event.group_id))
        return None
    if target == "qq_avatar":
        qq = params.get("qq")
        if qq:
            return _avatar_url(str(qq))
        return None
    if target == "message_id":
        msg_id = _coerce_int(params.get("message_id"))
        if msg_id is None:
            return None
        cached = state.image_cache.get(msg_id)
        return cached[0] if cached else None
    if target == "wait_next":
        return await _wait_next_image(state, user_id, _WAIT_NEXT_IMAGE_SEC)
    return None


def _collect_context_messages(
    state: SessionState,
    current_user_id: str,
    *,
    ts: float,
    limit: int,
    future: bool,
    current_text: str,
) -> List[str]:
    if limit <= 0:
        return []
    texts: List[str] = []
    items = state.history if future else reversed(state.history)
    for item in items:
        if item.role != "user":
            continue
        if not item.to_bot:
            continue
        if future and item.ts <= ts:
            continue
        if not future and item.ts > ts:
            continue
        if item.text == current_text and item.user_id == current_user_id:
            continue
        line = _format_context_line(item.text, item.user_name or item.user_id)
        texts.append(line)
        if len(texts) >= limit:
            break
    if future:
        return texts
    return list(reversed(texts))


def _notify_pending_image(state: SessionState, user_id: str, image_url: str) -> None:
    waiter = state.pending_image_waiters.pop(user_id, None)
    if waiter and not waiter.done():
        waiter.set_result(image_url)


async def _wait_next_image(
    state: SessionState,
    user_id: str,
    timeout_sec: float,
) -> Optional[str]:
    waiter = state.pending_image_waiters.get(user_id)
    if waiter and not waiter.done():
        waiter.cancel()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    state.pending_image_waiters[user_id] = future
    try:
        return await asyncio.wait_for(future, timeout=timeout_sec)
    except Exception:
        return None
    finally:
        current = state.pending_image_waiters.get(user_id)
        if current is future:
            state.pending_image_waiters.pop(user_id, None)


async def _build_intent_text(
    event: MessageEvent,
    state: SessionState,
    text: str,
) -> str:
    try:
        max_prev = max(0, int(getattr(config, "nlp_context_history_messages", 2)))
    except Exception:
        max_prev = 2
    try:
        max_future = max(0, int(getattr(config, "nlp_context_future_messages", 2)))
    except Exception:
        max_future = 2
    try:
        wait_sec = max(0.0, float(getattr(config, "nlp_context_future_wait_sec", 1.0)))
    except Exception:
        wait_sec = 1.0

    ts = _event_ts(event)
    user_id = str(event.get_user_id())
    reply_text, reply_name = _extract_reply_context(event, state)

    prev_texts = _collect_context_messages(
        state,
        user_id,
        ts=ts,
        limit=max_prev,
        future=False,
        current_text=text,
    )
    future_texts: List[str] = []
    if max_future > 0:
        if wait_sec > 0:
            await asyncio.sleep(wait_sec)
        future_texts = _collect_context_messages(
            state,
            user_id,
            ts=ts,
            limit=max_future,
            future=True,
            current_text=text,
        )

    reply_line = ""
    if reply_text:
        reply_line = (
            _format_context_line(reply_text, reply_name)
            if reply_name
            else f"回复内容: {reply_text}"
        )
    combined = [
        part
        for part in [text, reply_line, *prev_texts, *future_texts]
        if part
    ]
    if not combined:
        return text
    return "\n".join(combined)


def _build_primary_intent_text(
    event: MessageEvent,
    state: SessionState,
    text: str,
) -> str:
    reply_text, reply_name = _extract_reply_context(event, state)
    if not reply_text:
        return text
    if reply_text.strip() == text.strip():
        return text
    reply_line = (
        _format_context_line(reply_text, reply_name)
        if reply_name
        else f"回复内容: {reply_text}"
    )
    return "\n".join([text, reply_line])


_ALLOWED_ACTIONS = {
    "chat",
    "image_chat",
    "image_generate",
    "image_create",
    "weather",
    "avatar_get",
    "travel_plan",
    "history_clear",
    "ignore",
}
_ALLOWED_TARGETS = {
    "message_image",
    "reply_image",
    "at_user",
    "last_image",
    "sender_avatar",
    "group_avatar",
    "qq_avatar",
    "message_id",
    "wait_next",
    "trip",
    "none",
}


def _intent_params(intent: Optional[dict]) -> dict[str, object]:
    if not isinstance(intent, dict):
        return {}
    raw_params = intent.get("params")
    return raw_params if isinstance(raw_params, dict) else {}


def _normalize_intent(
    intent: Optional[dict],
    has_image: bool,
    has_reply_image: bool,
    at_user: Optional[str],
    state: SessionState,
) -> Optional[dict]:
    if not isinstance(intent, dict):
        return None
    action = str(intent.get("action", "")).strip().lower()
    if action not in _ALLOWED_ACTIONS:
        return None
    if action == "ignore":
        return {"action": "ignore"}
    instruction = intent.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    params = _intent_params(intent)
    target = str(intent.get("target", "")).strip().lower()

    if action == "image_create":
        return {
            "action": action,
            "instruction": instruction.strip(),
            "target": "none",
            "params": params,
        }

    if action in {"image_chat", "image_generate"}:
        if target not in _ALLOWED_TARGETS:
            target = ""
        if not target or target == "none":
            if has_image:
                target = "message_image"
            elif has_reply_image:
                target = "reply_image"
            elif at_user:
                target = "at_user"
            elif state.last_image_url:
                target = "last_image"
            else:
                target = "wait_next"
        return {
            "action": action,
            "instruction": instruction.strip(),
            "target": target,
            "params": params,
        }

    if action == "avatar_get":
        if target not in _ALLOWED_TARGETS:
            target = ""
        if not target or target == "none":
            target = "sender_avatar"
        return {
            "action": action,
            "instruction": instruction.strip(),
            "target": target,
            "params": params,
        }

    if action == "weather":
        city = ""
        raw_city = params.get("city")
        if isinstance(raw_city, str):
            city = raw_city.strip()
        if not city and isinstance(instruction, str):
            city = instruction.strip()
        return {
            "action": action,
            "instruction": city,
            "target": "city",
            "params": {"city": city} if city else {},
        }

    if action == "travel_plan":
        days = _coerce_int(params.get("days"))
        nights = _coerce_int(params.get("nights"))
        destination = ""
        raw_destination = params.get("destination") or params.get("city")
        if isinstance(raw_destination, str):
            destination = raw_destination.strip()
        if (days is None or nights is None) and isinstance(instruction, str):
            parsed_days, parsed_nights = _extract_travel_duration(instruction)
            if days is None:
                days = parsed_days
            if nights is None:
                nights = parsed_nights
        if not destination and isinstance(instruction, str):
            destination = _extract_travel_destination(instruction) or ""
        normalized_params: dict[str, object] = {}
        if days is not None:
            normalized_params["days"] = days
        if nights is not None:
            normalized_params["nights"] = nights
        if destination:
            normalized_params["destination"] = destination
        return {
            "action": action,
            "instruction": instruction.strip(),
            "target": "trip",
            "params": normalized_params,
        }

    return {"action": action, "instruction": instruction.strip(), "params": params}


async def _build_travel_plan_reply(
    intent: dict,
    state: SessionState,
    event: MessageEvent,
) -> Optional[str]:
    params = _intent_params(intent)
    destination = params.get("destination")
    destination_text = destination.strip() if isinstance(destination, str) else ""
    if not destination_text:
        return "请告诉我目的地，例如：北京"
    normalized_params = dict(params)
    normalized_params["destination"] = destination_text
    intent = dict(intent)
    intent["params"] = normalized_params
    reply = await _call_gemini_travel_plan(intent, state)
    if not reply:
        return None
    instruction = str(intent.get("instruction") or "").strip()
    cleaned_instruction = _strip_travel_duration(instruction)
    summary = f"{destination_text}"
    if cleaned_instruction and cleaned_instruction not in summary:
        summary = f"{summary} 需求:{cleaned_instruction}"
    user_name = _event_user_name(event)
    _append_history(
        state,
        "user",
        f"旅行规划：{summary}",
        user_id=str(event.get_user_id()),
        user_name=user_name,
        to_bot=True,
    )
    _append_history(state, "model", reply)
    return reply


async def _dispatch_intent(
    intent: dict,
    state: SessionState,
    event: MessageEvent,
    text: str,
    *,
    image_url: Optional[str],
    reply_image_url: Optional[str],
    at_user: Optional[str],
    send_func,
) -> None:
    action = str(intent.get("action", "ignore")).lower()
    if action == "ignore":
        return
    user_name = _event_user_name(event)

    if action == "chat":
        prompt = intent.get("instruction")
        try:
            reply = await _call_gemini_text(str(prompt), state)
            if not reply:
                return
            _append_history(
                state,
                "user",
                str(prompt),
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            _append_history(state, "model", reply)
            await send_func(reply)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP chat failed: {}", _safe_error_message(exc))
        return

    if action == "weather":
        query = str(intent.get("instruction") or "").strip()
        if not query:
            await send_func("请告诉我城市或地区，例如：天气 北京")
            return
        await _send_transition(action, send_func)
        try:
            messages = await _build_weather_messages(query)
            if not messages:
                return
            reply_text = "\n".join(messages)
            _append_history(
                state,
                "user",
                f"天气：{query}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            _append_history(state, "model", reply_text)
            for msg in messages:
                await send_func(msg)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP weather failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action == "travel_plan":
        params = _intent_params(intent)
        destination = params.get("destination")
        if not isinstance(destination, str) or not destination.strip():
            await send_func("请告诉我目的地，例如：北京")
            return
        await _send_transition(action, send_func)
        try:
            reply = await _build_travel_plan_reply(intent, state, event)
            if not reply:
                return
            await send_func(reply)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP travel failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action == "history_clear":
        _clear_session_state(state)
        await send_func("已清除当前会话记录，可以继续聊啦。")
        return

    if action == "avatar_get":
        target = str(intent.get("target") or "").lower()
        params = _intent_params(intent)
        if target == "qq_avatar" and not params.get("qq"):
            await send_func("请提供 QQ 号。")
            return
        await _send_transition(action, send_func)
        image_url = await _resolve_image_url(
            intent,
            event=event,
            state=state,
            current_image_url=None,
            reply_image_url=None,
            at_user=at_user,
        )
        if not image_url:
            await send_func("未找到可用的头像。")
            return
        await send_func(_image_segment_from_result(image_url))
        _mark_handled_request(state, event, text)
        return

    prompt = str(intent.get("instruction"))
    target = str(intent.get("target") or "").lower()
    params = _intent_params(intent)

    if action == "image_create":
        transition_text = _intent_transition_text(intent)
        if transition_text:
            await send_func(transition_text)
        try:
            is_image, result = await _call_gemini_text_to_image(prompt, state)
            _append_history(
                state,
                "user",
                f"生成图片：{prompt}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            if is_image:
                _append_history(state, "model", "[已生成图片]")
                await send_func("已生成图片。")
                await send_func(_image_segment_from_result(result))
                _mark_handled_request(state, event, text)
            else:
                _append_history(state, "model", result)
                await send_func(f"生成结果：{result}")
                _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP image create failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action not in {"image_chat", "image_generate"}:
        return

    if target == "qq_avatar" and not params.get("qq"):
        await send_func("请提供 QQ 号。")
        return
    if target == "message_id" and not params.get("message_id"):
        await send_func("请提供消息 ID。")
        return
    if target == "wait_next":
        await send_func("请在60秒内发送图片。")

    image_url = await _resolve_image_url(
        intent,
        event=event,
        state=state,
        current_image_url=image_url,
        reply_image_url=reply_image_url,
        at_user=at_user,
    )
    if not image_url:
        await send_func("未找到可处理的图片或头像。")
        return

    if action == "image_chat":
        try:
            await _send_transition(action, send_func)
            reply = await _call_gemini_image_chat(prompt, image_url, state)
            if not reply:
                return
            _append_history(
                state,
                "user",
                f"聊图：{prompt}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            _append_history(state, "model", reply)
            await send_func(reply)
            _mark_handled_request(state, event, text)
        except UnsupportedImageError:
            await send_func("这个格式我处理不了，发张静态图吧。")
        except Exception as exc:
            logger.error("NLP image chat failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    try:
        transition_text = _intent_transition_text(intent)
        if transition_text:
            await send_func(transition_text)
        is_image, result = await _call_gemini_image(prompt, image_url, state)
        _append_history(
            state,
            "user",
            f"处理头像：{prompt}",
            user_id=str(event.get_user_id()),
            user_name=user_name,
            to_bot=True,
        )
        if is_image:
            _append_history(state, "model", "[已生成图片]")
            await send_func("已完成修改。")
            await send_func(_image_segment_from_result(result))
            _mark_handled_request(state, event, text)
        else:
            _append_history(state, "model", result)
            await send_func(f"修改结果：{result}")
            _mark_handled_request(state, event, text)
    except UnsupportedImageError:
        await send_func("这个格式我处理不了，发张静态图吧。")
    except Exception as exc:
        logger.error("NLP image failed: {}", _safe_error_message(exc))
        await send_func(f"出错了：{_safe_error_message(exc)}")


def _clarify_intent_text(has_image: bool) -> str:
    if has_image:
        return "我没太听懂，你是想聊这张图、处理图片、查天气还是旅行规划？"
    return "我没太听懂，你是想聊天、处理图片、无图生成、查天气、旅行规划还是清除历史？"


_TRAVEL_KEYWORDS = ("旅行", "旅游", "行程", "出行", "游玩")
_TRAVEL_WEAK_KEYWORDS = ("规划", "计划")
_TRAVEL_DAYS_RE = re.compile(r"([0-9]{1,2}|[零一二三四五六七八九十两]{1,3})\s*天")
_TRAVEL_NIGHTS_RE = re.compile(r"([0-9]{1,2}|[零一二三四五六七八九十两]{1,3})\s*(?:晚|夜)")
_TRAVEL_DEST_RE = re.compile(r"(?:去|到|在)\s*([\u4e00-\u9fffA-Za-z0-9]{1,20})")


def _chinese_number_to_int(value: str) -> Optional[int]:
    if not value:
        return None
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value.isdigit():
        return int(value)
    if value in digits:
        return digits[value]
    if value == "十":
        return 10
    if len(value) == 2 and value[0] == "十":
        tail = digits.get(value[1])
        return 10 + tail if tail is not None else None
    if len(value) == 2 and value[1] == "十":
        head = digits.get(value[0])
        return head * 10 if head is not None else None
    if len(value) == 3 and value[1] == "十":
        head = digits.get(value[0])
        tail = digits.get(value[2])
        if head is None or tail is None:
            return None
        return head * 10 + tail
    return None


def _parse_travel_number(token: str) -> Optional[int]:
    if not token:
        return None
    number = _coerce_int(token)
    if number is not None:
        return number
    return _chinese_number_to_int(token)


def _extract_travel_duration(text: str) -> Tuple[Optional[int], Optional[int]]:
    days = None
    nights = None
    if not text:
        return days, nights
    day_match = _TRAVEL_DAYS_RE.search(text)
    if day_match:
        days = _parse_travel_number(day_match.group(1))
    night_match = _TRAVEL_NIGHTS_RE.search(text)
    if night_match:
        nights = _parse_travel_number(night_match.group(1))
    return days, nights


def _extract_travel_destination(text: str) -> Optional[str]:
    if not text:
        return None
    match = _TRAVEL_DEST_RE.search(text)
    if match:
        return match.group(1).strip()
    cleaned = _TRAVEL_DAYS_RE.sub("", text)
    cleaned = _TRAVEL_NIGHTS_RE.sub("", cleaned)
    cleaned = re.sub(r"[，,。.!！?？/]", " ", cleaned)
    for kw in _TRAVEL_KEYWORDS:
        cleaned = cleaned.replace(kw, " ")
    for kw in _TRAVEL_WEAK_KEYWORDS:
        cleaned = cleaned.replace(kw, " ")
    cleaned = cleaned.replace("去", " ").replace("到", " ").replace("在", " ")
    cleaned = _collapse_spaces(cleaned)
    return cleaned or None


def _strip_travel_duration(text: str) -> str:
    if not text:
        return ""
    cleaned = _TRAVEL_DAYS_RE.sub("", text)
    cleaned = _TRAVEL_NIGHTS_RE.sub("", cleaned)
    return _collapse_spaces(cleaned)


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
    has_reply_image: bool,
    at_user: Optional[str],
) -> Optional[dict]:
    if not config.google_api_key:
        return None
    client = _get_client()
    system = _INTENT_SYSTEM_PROMPT
    user_prompt = (
        f"文本: {text}\n"
        f"消息包含图片: {has_image}\n"
        f"回复里有图片: {has_reply_image}\n"
        f"是否@用户: {bool(at_user)}\n"
        f"是否有最近图片: {bool(state.last_image_url)}\n"
    )
    config_obj, system_used = _build_generate_config(
        system_instruction=system,
        response_mime_type="application/json",
    )
    if system and not system_used:
        user_prompt = f"{system}\n\n{user_prompt}"
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
    if not _should_trigger_nlp(event, text):
        return
    if not config.google_api_key:
        return

    session_id = _session_id(event)
    state = _get_state(session_id)
    if _is_duplicate_request(state, event, text):
        return
    image_url = _extract_first_image_url(event.get_message())
    at_user = _extract_at_user(event.get_message())
    reply_image_url = _extract_reply_image_url(event, state)
    has_image = image_url is not None
    has_reply_image = reply_image_url is not None

    try:
        primary_text = _build_primary_intent_text(event, state, text)
        intent_raw = await _classify_intent(
            primary_text, state, has_image, has_reply_image, at_user
        )
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        return

    intent = _normalize_intent(intent_raw, has_image, has_reply_image, at_user, state)
    if not intent:
        try:
            intent_text = await _build_intent_text(event, state, text)
            if intent_text and intent_text != primary_text:
                intent_raw = await _classify_intent(
                    intent_text, state, has_image, has_reply_image, at_user
                )
                intent = _normalize_intent(
                    intent_raw, has_image, has_reply_image, at_user, state
                )
        except Exception as exc:
            logger.error("Intent classify failed: {}", _safe_error_message(exc))
            return
    if not intent:
        await nlp_handler.send(_clarify_intent_text(has_image))
        return

    reply = getattr(event, "reply", None)
    reply_id = getattr(reply, "message_id", None) if reply else None
    if reply_id is not None and isinstance(intent.get("params"), dict):
        intent["params"].setdefault("message_id", reply_id)
    await _dispatch_intent(
        intent,
        state,
        event,
        text,
        image_url=image_url,
        reply_image_url=reply_image_url,
        at_user=at_user,
        send_func=nlp_handler.send,
    )


async def _handle_command_via_intent(
    event: MessageEvent,
    *,
    text: str,
    send_func,
) -> None:
    if not config.google_api_key:
        await send_func("未配置 GOOGLE_API_KEY")
        return
    session_id = _session_id(event)
    state = _get_state(session_id)
    image_url = _extract_first_image_url(event.get_message())
    at_user = _extract_at_user(event.get_message())
    reply_image_url = _extract_reply_image_url(event, state)
    has_image = image_url is not None
    has_reply_image = reply_image_url is not None
    try:
        intent_raw = await _classify_intent(
            text, state, has_image, has_reply_image, at_user
        )
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        await send_func("意图解析失败，请稍后再试。")
        return
    intent = _normalize_intent(intent_raw, has_image, has_reply_image, at_user, state)
    if not intent:
        await send_func(_clarify_intent_text(has_image))
        return
    reply = getattr(event, "reply", None)
    reply_id = getattr(reply, "message_id", None) if reply else None
    if reply_id is not None and isinstance(intent.get("params"), dict):
        intent["params"].setdefault("message_id", reply_id)
    await _dispatch_intent(
        intent,
        state,
        event,
        text,
        image_url=image_url,
        reply_image_url=reply_image_url,
        at_user=at_user,
        send_func=send_func,
    )


@avatar_handler.handle()
async def handle_avatar(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    prompt = args.extract_plain_text().strip()
    if not prompt:
        await avatar_handler.finish("请告诉我你想怎么处理头像，例如：处理头像 变成赛博朋克风")
    await _handle_command_via_intent(
        event,
        text=f"处理头像 {prompt}",
        send_func=avatar_handler.send,
    )


@chat_handler.handle()
async def handle_chat(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    prompt = args.extract_plain_text().strip()
    if not prompt:
        await chat_handler.finish("请发送要聊天的内容，例如：聊天 你好")
    await _handle_command_via_intent(
        event,
        text=f"聊天 {prompt}",
        send_func=chat_handler.send,
    )


@weather_handler.handle()
async def handle_weather(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    if not query:
        await weather_handler.finish("请提供城市或地区，例如：天气 北京")
    await _handle_command_via_intent(
        event,
        text=f"天气 {query}",
        send_func=weather_handler.send,
    )


@travel_handler.handle()
async def handle_travel(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    text = args.extract_plain_text().strip()
    if not text:
        await travel_handler.finish("请提供行程需求，例如：旅行规划 3天2晚 北京")
    await _handle_command_via_intent(
        event,
        text=f"旅行规划 {text}",
        send_func=travel_handler.send,
    )
