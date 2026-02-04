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
_PARA_SPLIT_RE = re.compile(r"\n{2,}")

_MODEL_REPLY_MAX_CHARS = 0
_WAIT_NEXT_IMAGE_SEC = 60.0
_IMAGE_CACHE_REF_PREFIX = "cache:"
_IMAGE_CACHE_MAX_BYTES = 8 * 1024 * 1024

_CHAT_SYSTEM_PROMPT = (
    "Role\n"
    "你是asoul成员嘉然，会尽量满足提问者的帮助，正在和朋友私聊或群聊。\n\n"
    "Goal\n"
    "根据对方消息给出自然、真实、适合 QQ 消息的回复。\n\n"
    "Rules\n"
    "1. 只当普通人，不要自称 AI/助手，也不要提系统或规则。\n"
    "2. 输出纯文本，不使用 Markdown 或代码块。\n"
    "3. 只回复当前消息，不要夹带已回复过的旧话题。\n"
    "Output\n"
    "只输出最终回复内容，可自然换行。\n"
)

_IMAGE_CHAT_SYSTEM_PROMPT = (
    "你现在是asoul成员嘉然，会尽量满足提问者的帮助。\n"
    "你在进行图片内容对话，只需回答当前指令或问题。\n"
    "不要补充已回复过的历史话题，不要输出 Markdown 或代码块。\n"
    "回答适合 QQ 消息，精炼、不啰嗦，简短、口语化，可自然换行。\n"
)

_TRAVEL_SYSTEM_PROMPT = (
    "Role\n"
    "你是旅行规划助手，给出清晰、实用、可执行的旅行建议。\n"
    "Goal\n"
    "根据对方消息给出自然、真实、适合 QQ 消息的回复。\n\n"
    "Rules\n"
    "输出纯文本，不使用 Markdown 或代码块。\n"
    "适合 QQ 消息，精炼、不啰嗦。\n"
    "结构清晰，可自然换行，尽量不要空行，包含景点/活动/用餐/交通/住宿/注意事项等要点。\n"
    "请自动生成该城市最常见的规划天数。\n"
    "Output\n"
    "只输出最终回复内容。\n"
)

_INTENT_SYSTEM_PROMPT = (
    "你是消息意图解析器，只输出 JSON，不要解释或补充说明。"
    "不要输出拒绝/免责声明/权限说明（例如“我无法访问账号”）。"
    "只输出单一 JSON 对象，格式如下："
    "{"
    "\"action\": \"chat|image_chat|image_generate|image_create|weather|avatar_get|travel_plan|history_clear|ignore\","
    "\"target\": \"message_image|reply_image|at_user|last_image|sender_avatar|group_avatar|qq_avatar|message_id|wait_next|city|trip|none\","
    "\"params\": {\"qq\": \"string\", \"message_id\": \"int\", \"city\": \"string\","
    " \"destination\": \"string\", \"days\": \"int\", \"nights\": \"int\", \"reply\": \"string\"}"
    "}"
    "规则："
    "- action=ignore：target=none，params={}。"
    "- action=chat：普通聊天；target=none。"
    "- action=image_chat：聊这张图（不生成图）；target 用于选图：message_image/reply_image/at_user/last_image/sender_avatar/group_avatar/qq_avatar/message_id/wait_next。"
    "- action=image_generate：基于参考图生成/编辑；target 用于选图：同上。"
    "- action=image_create：无参考图生成；target=none。"
    "- action=weather：查询天气；target=city；params.city 为地点（没有就留空）。"
    "- action=avatar_get：获取头像；target 可为 sender_avatar/group_avatar/qq_avatar/at_user；target=qq_avatar 时填 params.qq。"
    "- action=travel_plan：旅行规划；target=trip；params.destination/days/nights 可填则填。"
    "- action=history_clear：清除当前会话历史；target=none。"
    "- target=message_id 时填写 params.message_id。"
    "- params 仅在对应 target/场景需要时填写，其余为空对象。"
    "- 若旅行或天气缺关键信息，仍输出对应 action，缺失字段留空"
    "- 当需要调用第三方工具且可能耗时（如 weather、image_create、image_generate、image_chat、avatar_get、travel_plan）时，可在 params.reply 中给等待/过渡语。"
    "- 若消息里 @ 多人，仍输出 target=at_user，系统会按顺序处理多个头像。"
    "- 上下文可能包含“昵称: 内容”的格式，需识别说话人。"
)

_DUPLICATE_TEXT_TTL_SEC = 60.0
_HISTORY_SUMMARY_ITEM_MAX_CHARS = 400

_HISTORY_SUMMARY_SYSTEM_PROMPT = (
    "你是对话摘要器，请将对话压缩成简短摘要。"
    "保留关键信息、用户偏好、需求、结论与待办。"
    "输出纯文本，不使用 Markdown、编号或引号。"
)


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
    name = name.strip(":：")
    # Avoid leaking QQ numbers to external APIs when we don't have a nickname.
    if name.isdigit() and len(name) >= 5:
        return ""
    return name


def _event_user_name(event: MessageEvent) -> str:
    sender = getattr(event, "sender", None)
    name = None
    if sender is not None:
        name = getattr(sender, "card", None) or getattr(sender, "nickname", None)
    normalized = _normalize_user_name(name)
    return normalized or "用户"


def _sender_user_name(sender: object) -> str:
    if sender is None:
        return ""
    name = getattr(sender, "card", None) or getattr(sender, "nickname", None)
    normalized = _normalize_user_name(name)
    return normalized or "用户"


def _format_context_line(text: str, user_name: Optional[str]) -> str:
    name = _normalize_user_name(user_name)
    if name:
        return f"{name}: {text}"
    return text


def _model_user_name() -> str:
    return "我"


def _compact_reply_lines(text: str) -> str:
    if not text:
        return text
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    last_blank = False
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            # Keep a single blank line as paragraph separator.
            if lines and not last_blank:
                lines.append("")
                last_blank = True
            continue
        lines.append(line)
        last_blank = False
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def _message_send_delay_sec() -> float:
    try:
        value = float(getattr(config, "message_send_delay_sec", 0.0))
    except Exception:
        value = 0.0
    return max(0.0, value)


def _split_by_double_newline(text: str) -> List[str]:
    if not text:
        return []
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    blocks = [block.strip() for block in _PARA_SPLIT_RE.split(normalized) if block.strip()]
    return blocks


def _forward_line_threshold() -> int:
    try:
        threshold = int(getattr(config, "forward_line_threshold", 0))
    except Exception:
        threshold = 0
    if threshold <= 0:
        return 8
    return threshold


def _forward_char_threshold() -> int:
    try:
        threshold = int(getattr(config, "forward_char_threshold", 0))
    except Exception:
        threshold = 0
    if threshold <= 0:
        return 100
    return threshold


def _bot_display_name(bot: Bot) -> str:
    nick = getattr(bot, "config", None)
    if nick is not None:
        nickname = getattr(nick, "nickname", None)
        if isinstance(nickname, (list, tuple)) and nickname:
            return str(nickname[0])
        if isinstance(nickname, str) and nickname.strip():
            return nickname.strip()
    return "嘉然"


def _strip_leading_command(text: str, words: Tuple[str, ...]) -> str:
    """Strip leading nonebot command prefix + command word (when it looks like a command).

    This is used to recover the raw user intent text for business logic after intent JSON routing.
    """
    value = str(text or "").strip()
    if not value:
        return ""
    try:
        starts = list(get_driver().config.command_start or [])
    except Exception:
        starts = ["/"]
    if "" not in starts:
        starts.append("")
    seps = set(" \t\r\n:：,，。.!！?？;；")
    for prefix in starts:
        if prefix is None:
            continue
        for word in words:
            token = f"{prefix}{word}"
            if not value.startswith(token):
                continue
            rest = value[len(token) :]
            if not rest:
                return ""
            # Only strip when it is a command token boundary, to avoid breaking normal text like "聊天好无聊".
            if rest[0] not in seps:
                continue
            return rest.lstrip("".join(seps)).strip()
    return value


_CQ_AT_TOKEN_RE = re.compile(r"\[CQ:at,qq=(all|\d+)(?:,[^\]]*)?\]")
_CQ_TOKEN_RE = re.compile(r"\[CQ:[^\]]+\]")


def _sanitize_cq_tokens(text: str) -> str:
    """Best-effort sanitize CQ-like tokens in plain text.

    We avoid leaking QQ numbers to external LLM APIs. Also keep the semantic of mentions.
    """
    if not text:
        return text

    def _replace_at(match: re.Match[str]) -> str:
        qq = match.group(1)
        if qq == "all":
            return "@全体成员"
        return "@用户"

    cleaned = _CQ_AT_TOKEN_RE.sub(_replace_at, str(text))
    cleaned = _CQ_TOKEN_RE.sub("", cleaned)
    return cleaned


def _normalize_prompt_text(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    ).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


async def _resolve_at_display_name(bot: Bot, event: MessageEvent, qq: str) -> str:
    qq = str(qq or "").strip()
    if not qq:
        return ""
    if qq == "all":
        return "全体成员"
    if str(getattr(event, "self_id", "")) and qq == str(getattr(event, "self_id")):
        return _bot_display_name(bot)
    if not isinstance(event, GroupMessageEvent):
        return ""
    try:
        user_id = int(qq)
    except Exception:
        return ""
    try:
        info = await bot.get_group_member_info(group_id=event.group_id, user_id=user_id)
    except Exception:
        return ""
    if not isinstance(info, dict):
        return ""
    name = info.get("card") or info.get("nickname") or ""
    return _normalize_user_name(name)


async def _event_message_text(bot: Bot, event: MessageEvent) -> str:
    """Build a safe text representation of the current message for LLM/tool prompts.

    - Preserve @ mentions as @昵称 (best-effort), but never leak QQ numbers.
    - Strip CQ tokens if they appear as literal text.
    - Ignore non-text segments (images, files, etc.) to avoid leaking URLs/IDs.
    """
    message = event.get_message()
    parts: List[str] = []
    at_list: List[str] = []
    placeholders: List[str] = []
    for seg in message:
        if seg.type == "text":
            parts.append(str(seg.data.get("text") or ""))
            continue
        if seg.type == "at":
            qq = str(seg.data.get("qq") or "").strip()
            if not qq:
                continue
            if qq == "all":
                parts.append("@全体成员")
                continue
            placeholder = f"__AT_{len(at_list)}__"
            at_list.append(qq)
            placeholders.append(placeholder)
            parts.append(placeholder)
            continue
        # Skip other CQ segments to avoid leaking URLs/IDs.
    text = "".join(parts).strip()
    if not text:
        return ""
    if at_list:
        resolved = await asyncio.gather(
            *[_resolve_at_display_name(bot, event, qq) for qq in at_list],
            return_exceptions=True,
        )
        for idx, qq in enumerate(at_list):
            name = ""
            value = resolved[idx]
            if isinstance(value, str):
                name = value.strip()
            display = name or (f"用户{idx + 1}" if len(at_list) > 1 else "用户")
            text = text.replace(placeholders[idx], f"@{display}")
    text = _sanitize_cq_tokens(text)
    return _normalize_prompt_text(text)


async def _send_text_response(
    bot: Bot,
    event: MessageEvent,
    send_func,
    text: str,
) -> None:
    if not text:
        return
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return
    if len(normalized) > _forward_char_threshold():
        blocks = _split_by_double_newline(normalized)
        if not blocks:
            blocks = [normalized]
        nickname = _bot_display_name(bot)
        self_id = _coerce_int(getattr(event, "self_id", None))
        if self_id is None:
            await send_func(normalized)
            return
        nodes = [
            MessageSegment.node_custom(
                user_id=self_id,
                nickname=nickname,
                content=block,
            )
            for block in blocks
        ]
        try:
            if isinstance(event, GroupMessageEvent):
                await bot.send_group_forward_msg(group_id=event.group_id, messages=nodes)
            else:
                user_id = _coerce_int(event.get_user_id())
                if user_id is None:
                    await send_func(normalized)
                    return
                await bot.send_private_forward_msg(user_id=user_id, messages=nodes)
        except Exception:
            await send_func(normalized)
        return
    blocks = _split_by_double_newline(normalized)
    if len(blocks) > 1:
        delay = _message_send_delay_sec()
        for idx, block in enumerate(blocks):
            await send_func(block)
            if delay > 0 and idx < len(blocks) - 1:
                await asyncio.sleep(delay)
        return
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) <= _forward_line_threshold():
        await send_func(normalized)
        return
    nickname = _bot_display_name(bot)
    self_id = _coerce_int(getattr(event, "self_id", None))
    if self_id is None:
        await send_func(normalized)
        return
    nodes = [
        MessageSegment.node_custom(
            user_id=self_id,
            nickname=nickname,
            content=line,
        )
        for line in lines
    ]
    try:
        if isinstance(event, GroupMessageEvent):
            await bot.send_group_forward_msg(group_id=event.group_id, messages=nodes)
        else:
            user_id = _coerce_int(event.get_user_id())
            if user_id is None:
                await send_func(normalized)
                return
            await bot.send_private_forward_msg(user_id=user_id, messages=nodes)
    except Exception:
        await send_func(normalized)


def _transition_text(action: str) -> Optional[str]:
    # 默认过渡语：耗时操作时给用户“正在处理”的提示
    if action in {"image_create"}:
        return "正在生成图片，请稍候..."
    if action in {"image_generate"}:
        return "正在处理图片，请稍候..."
    return None


def _intent_transition_text(intent: dict) -> str:
    # NLP 可选生成的过渡语（params.reply），有就用，没有就空字符串
    params = _intent_params(intent)
    reply = params.get("reply")
    if isinstance(reply, str):
        return reply.strip()
    return ""


def _resolve_transition_text(action: str, intent: dict) -> Optional[str]:
    # 优先使用 intent 给的过渡语，否则回退默认提示
    reply = _intent_transition_text(intent)
    if reply:
        return reply
    return _transition_text(action)


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
    is_summary: bool = False


@dataclass
class CachedImage:
    ts: float
    url: Optional[str] = None
    file_id: Optional[str] = None
    content_type: Optional[str] = None
    data: Optional[bytes] = None


@dataclass
class SessionState:
    history: List[HistoryItem]
    last_image_id: Optional[int]
    image_cache: dict[int, CachedImage]
    image_cache_tasks: dict[int, asyncio.Task[None]]
    pending_image_waiters: dict[str, asyncio.Future[int]]
    handled_message_ids: dict[int, float]
    handled_texts: dict[str, float]
    history_lock: asyncio.Lock
    summary_last_ts: float
    summary_in_progress: bool


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
            last_image_id=None,
            image_cache={},
            image_cache_tasks={},
            pending_image_waiters={},
            handled_message_ids={},
            handled_texts={},
            history_lock=asyncio.Lock(),
            summary_last_ts=0.0,
            summary_in_progress=False,
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


def _history_compress_enabled() -> bool:
    try:
        return bool(getattr(config, "history_compress_enable", True))
    except Exception:
        return True


def _history_reference_only() -> bool:
    try:
        return bool(getattr(config, "history_reference_only", True))
    except Exception:
        return True


def _history_compress_trigger() -> int:
    try:
        value = int(getattr(config, "history_compress_trigger", 0))
    except Exception:
        value = 0
    if value <= 0:
        try:
            base = int(getattr(config, "history_max_messages", 10))
        except Exception:
            base = 10
        return max(16, base * 2)
    return value


def _history_compress_keep() -> int:
    try:
        value = int(getattr(config, "history_compress_keep", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 6
    return value


def _history_compress_min_messages() -> int:
    try:
        value = int(getattr(config, "history_compress_min_messages", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 6
    return value


def _history_compress_max_chars() -> int:
    try:
        value = int(getattr(config, "history_compress_max_chars", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 600
    return value


def _history_hard_limit() -> int:
    try:
        base = int(getattr(config, "history_max_messages", 10))
    except Exception:
        base = 10
    trigger = _history_compress_trigger()
    return max(50, base * 5, trigger * 2)


def _count_non_summary_items(items: List[HistoryItem]) -> int:
    return sum(1 for item in items if not item.is_summary)


def _history_item_label(item: HistoryItem) -> str:
    if item.role == "model":
        return item.user_name or _model_user_name()
    if item.is_summary:
        return item.user_name or "系统摘要"
    return item.user_name or "用户"


def _history_item_to_line(item: HistoryItem) -> str:
    text = _ensure_plain_text(str(item.text))
    if not text:
        return ""
    text = _truncate(text, _HISTORY_SUMMARY_ITEM_MAX_CHARS)
    name = _history_item_label(item)
    if name:
        return f"{name}: {text}"
    return text


def _build_history_summary_input(items: List[HistoryItem]) -> str:
    lines: List[str] = []
    for item in items:
        line = _history_item_to_line(item)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _build_history_reference_text(state: SessionState) -> str:
    lines: List[str] = []
    for item in state.history:
        if item.is_summary:
            line = _history_item_to_line(item)
            if line:
                lines.append(line)
            continue
        if item.role == "user" and not item.to_bot:
            continue
        line = _history_item_to_line(item)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _wrap_prompt_with_reference(
    state: SessionState,
    prompt: str,
    *,
    current_label: str,
) -> str:
    if not _history_reference_only():
        return prompt
    reference_text = _build_history_reference_text(state)
    if not reference_text:
        return prompt
    return f"参考对话(仅供参考，不需要回复):\n{reference_text}\n\n{current_label}:\n{prompt}"


async def _summarize_history_items(items: List[HistoryItem]) -> Optional[str]:
    if not items:
        return None
    input_text = _build_history_summary_input(items)
    if not input_text:
        return None
    max_chars = _history_compress_max_chars()
    user_prompt = (
        f"请总结以下对话记录，输出一段简短摘要，控制在{max_chars}字以内。\n"
        f"对话记录:\n{input_text}"
    )
    client = _get_client()
    config_obj, system_used = _build_generate_config(
        system_instruction=_HISTORY_SUMMARY_SYSTEM_PROMPT
    )
    if _HISTORY_SUMMARY_SYSTEM_PROMPT and not system_used:
        user_prompt = f"{_HISTORY_SUMMARY_SYSTEM_PROMPT}\n\n{user_prompt}"
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=config.gemini_text_model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
            config=config_obj,
        ),
        timeout=config.request_timeout,
    )
    if config.gemini_log_response:
        logger.info("Gemini history summary response: {}", _dump_response(response))
        _log_response_text("Gemini history summary content", response)
    if response.text:
        cleaned = _format_reply_text(response.text.strip())
        cleaned = _compact_reply_lines(cleaned)
        cleaned = _limit_reply_text(cleaned, max_chars)
        return cleaned
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        text_value = _extract_text_value(part)
        if text_value:
            text_parts.append(text_value)
    cleaned = _format_reply_text("\n".join(text_parts).strip())
    cleaned = _compact_reply_lines(cleaned)
    cleaned = _limit_reply_text(cleaned, max_chars)
    return cleaned


async def _maybe_compress_history(state: SessionState) -> bool:
    # 历史压缩：达到阈值时把旧记录摘要成一条“系统摘要”，保留最近若干条
    if not _history_compress_enabled():
        return False
    if not config.google_api_key:
        return False
    if state.summary_in_progress:
        return False
    trigger = _history_compress_trigger()
    if len(state.history) < trigger:
        return False
    keep = max(0, _history_compress_keep())
    if keep >= len(state.history):
        return False
    compress_items = state.history[:-keep] if keep > 0 else list(state.history)
    if _count_non_summary_items(compress_items) < _history_compress_min_messages():
        return False
    if not any(item.to_bot or item.role == "model" for item in compress_items):
        return False
    state.summary_in_progress = True
    try:
        summary = await _summarize_history_items(compress_items)
    except Exception as exc:
        logger.error("History summary failed: {}", _safe_error_message(exc))
        return False
    finally:
        state.summary_in_progress = False
    if not summary:
        return False
    ts = compress_items[-1].ts if compress_items else _now()
    summary_item = HistoryItem(
        role="user",
        text=summary,
        ts=ts,
        user_name="系统摘要",
        to_bot=True,
        is_summary=True,
    )
    keep_items = state.history[-keep:] if keep > 0 else []
    state.history = [summary_item, *keep_items]
    state.summary_last_ts = _now()
    return True


def _image_cache_max_images() -> int:
    try:
        value = int(getattr(config, "image_cache_max_images", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 10
    return max(1, value)


def _prune_state(state: SessionState, *, trim_history: bool = True) -> None:
    ttl = max(30, int(config.history_ttl_sec))
    cutoff = _now() - ttl
    state.history = [item for item in state.history if item.ts >= cutoff]
    hard_limit = _history_hard_limit()
    if hard_limit > 0 and len(state.history) > hard_limit:
        state.history = state.history[-hard_limit:]
    if trim_history:
        max_messages = max(1, int(config.history_max_messages))
        if len(state.history) > max_messages:
            state.history = state.history[-max_messages:]
    removed_image_ids: set[int] = set()
    if state.image_cache:
        for msg_id, cached in list(state.image_cache.items()):
            if cached.ts < cutoff:
                removed_image_ids.add(msg_id)
                state.image_cache.pop(msg_id, None)
        limit = _image_cache_max_images()
        if limit > 0 and len(state.image_cache) > limit:
            keep_ids = {
                msg_id
                for msg_id, _ in sorted(
                    state.image_cache.items(),
                    key=lambda kv: kv[1].ts,
                    reverse=True,
                )[:limit]
            }
            for msg_id in list(state.image_cache.keys()):
                if msg_id not in keep_ids:
                    removed_image_ids.add(msg_id)
                    state.image_cache.pop(msg_id, None)
    if removed_image_ids and state.image_cache_tasks:
        for msg_id in removed_image_ids:
            task = state.image_cache_tasks.pop(msg_id, None)
            if task and not task.done():
                task.cancel()
    if state.image_cache_tasks:
        for msg_id, task in list(state.image_cache_tasks.items()):
            if msg_id not in state.image_cache or task.done():
                state.image_cache_tasks.pop(msg_id, None)
    if state.last_image_id is not None and state.last_image_id not in state.image_cache:
        state.last_image_id = None
    if state.last_image_id is None and state.image_cache:
        state.last_image_id = max(state.image_cache.items(), key=lambda kv: kv[1].ts)[0]
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
    state.last_image_id = None
    state.image_cache = {}
    if state.image_cache_tasks:
        for task in state.image_cache_tasks.values():
            if task and not task.done():
                task.cancel()
    state.image_cache_tasks = {}
    state.summary_last_ts = 0.0
    state.summary_in_progress = False
    if state.pending_image_waiters:
        for waiter in state.pending_image_waiters.values():
            if not waiter.done():
                waiter.cancel()
    state.pending_image_waiters = {}
    state.handled_message_ids = {}
    state.handled_texts = {}


_UNSUPPORTED_IMAGE_EXTS: Tuple[str, ...] = ()


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
    cleaned = lower.split("?", 1)[0].split("#", 1)[0]
    for ext in _UNSUPPORTED_IMAGE_EXTS:
        if cleaned.endswith(ext):
            return False
    return True


def _extract_first_image_meta(message: Message) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """Extract (url, file_id) from the first image segment.

    Some OneBot implementations may not include a direct URL, but always include `file`
    which can be resolved via `get_image` later.
    """
    for seg in message:
        if seg.type not in {"image", "mface"}:
            continue
        url_raw = seg.data.get("url")
        file_raw = seg.data.get("file")
        url = str(url_raw).strip() if isinstance(url_raw, str) else ""
        file_id = str(file_raw).strip() if isinstance(file_raw, str) else ""

        # Some implementations may place the URL into `file`.
        if not url and file_id and (
            file_id.lower().startswith("http://")
            or file_id.lower().startswith("https://")
            or file_id.lower().startswith("data:image")
            or file_id.lower().startswith("base64://")
        ):
            url = file_id

        if url and not _is_supported_image_url(url):
            url = ""
        if file_id and not _is_supported_image_url(file_id):
            file_id = ""

        if not url and not file_id:
            continue
        return (url or None, file_id or None)
    return None


def _extract_first_image_url(message: Message) -> Optional[str]:
    meta = _extract_first_image_meta(message)
    if not meta:
        return None
    url, file_id = meta
    return url or file_id


def _cache_image_meta(
    state: SessionState,
    message_id: int,
    *,
    ts: float,
    url: Optional[str],
    file_id: Optional[str],
    update_last: bool = True,
) -> None:
    message_id = int(message_id)
    cached = state.image_cache.get(message_id)
    if cached is None:
        cached = CachedImage(ts=ts, url=url, file_id=file_id)
    else:
        cached.ts = ts
        if url:
            cached.url = url
        if file_id:
            cached.file_id = file_id
    state.image_cache[message_id] = cached
    if update_last:
        state.last_image_id = message_id
    _prune_state(state)


def _extract_at_users(message: Message, self_id: Optional[object]) -> List[str]:
    users: List[str] = []
    seen: set[str] = set()
    self_str = str(self_id) if self_id is not None else None
    for seg in message:
        if seg.type != "at":
            continue
        qq = seg.data.get("qq")
        if not qq or qq == "all":
            continue
        qq_str = str(qq)
        if self_str and qq_str == self_str:
            continue
        if qq_str in seen:
            continue
        seen.add(qq_str)
        users.append(qq_str)
    return users


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
            name = _normalize_user_name(item.user_name) or "用户"
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
    # 只回复当前消息：历史作为“参考文本”拼到当前指令里
    if _history_reference_only():
        prompt = _wrap_prompt_with_reference(state, prompt, current_label="当前消息")
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
    else:
        contents = _history_to_gemini(state)
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )
    config_obj, system_used = _build_generate_config(system_instruction=_CHAT_SYSTEM_PROMPT)
    if _CHAT_SYSTEM_PROMPT and not system_used:
        contents.insert(
            0,
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=_CHAT_SYSTEM_PROMPT)],
            ),
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


def _build_travel_prompt(intent: dict, *, raw_text: str = "") -> str:
    params = _intent_params(intent)
    destination = params.get("destination") or ""
    dest_text = str(destination).strip()
    days = _coerce_int(params.get("days"))
    nights = _coerce_int(params.get("nights"))
    cleaned_instruction = _strip_travel_duration(str(raw_text or ""))
    if dest_text and cleaned_instruction:
        cleaned_instruction = _collapse_spaces(cleaned_instruction.replace(dest_text, " "))
    if cleaned_instruction:
        for kw in _TRAVEL_KEYWORDS:
            cleaned_instruction = cleaned_instruction.replace(kw, " ")
        for kw in _TRAVEL_WEAK_KEYWORDS:
            cleaned_instruction = cleaned_instruction.replace(kw, " ")
        cleaned_instruction = (
            cleaned_instruction.replace("去", " ").replace("到", " ").replace("在", " ")
        )
        cleaned_instruction = _collapse_spaces(cleaned_instruction)
    parts = [_TRAVEL_SYSTEM_PROMPT.strip()]
    if dest_text:
        parts.append(f"请规划{dest_text}旅行行程。")
    else:
        parts.append("请规划旅行行程。")
    if days is not None and nights is not None:
        parts.append(f"行程时长：{days}天{nights}晚")
    elif days is not None:
        parts.append(f"行程时长：{days}天")
    elif nights is not None:
        parts.append(f"住宿：{nights}晚")
    if cleaned_instruction:
        parts.append(f"需求补充：{cleaned_instruction}")
    parts.append(
        "输出要求：纯文本，结构清晰，可自然换行，包含景点/活动/用餐/交通/住宿要点。"
    )
    return "\n".join(parts)


async def _call_gemini_travel_plan(
    intent: dict, state: SessionState, *, raw_text: str = ""
) -> str:
    client = _get_client()
    prompt = _build_travel_prompt(intent, raw_text=raw_text)
    # 只回复当前需求：历史作为参考文本
    if _history_reference_only():
        prompt = _wrap_prompt_with_reference(state, prompt, current_label="当前需求")
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
    else:
        contents = _history_to_gemini(state)
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )
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


_DATA_URL_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.I | re.S
)


def _cache_image_ref(message_id: int) -> str:
    return f"{_IMAGE_CACHE_REF_PREFIX}{message_id}"


def _parse_cache_image_ref(ref: object) -> Optional[int]:
    if not isinstance(ref, str):
        return None
    if not ref.startswith(_IMAGE_CACHE_REF_PREFIX):
        return None
    suffix = ref[len(_IMAGE_CACHE_REF_PREFIX) :].strip()
    if not suffix.isdigit():
        return None
    try:
        return int(suffix)
    except Exception:
        return None


async def _onebot_get_image_url(bot: Bot, file_id: str) -> Optional[str]:
    file_id = str(file_id or "").strip()
    if not file_id:
        return None
    try:
        info = await bot.get_image(file=file_id)
    except Exception:
        try:
            info = await bot.call_api("get_image", file=file_id)
        except Exception:
            return None
    if not isinstance(info, dict):
        return None
    url = info.get("url") or info.get("file")
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def _decode_data_url(value: str) -> Optional[Tuple[str, bytes]]:
    match = _DATA_URL_RE.match(value or "")
    if not match:
        return None
    content_type = match.group(1).strip() or "image/jpeg"
    payload = match.group(2) or ""
    try:
        data = base64.b64decode(payload)
    except Exception:
        return None
    return content_type, data


def _decode_base64_ref(value: str) -> Optional[Tuple[str, bytes]]:
    raw = str(value or "")
    if not raw.lower().startswith("base64://"):
        return None
    payload = raw[len("base64://") :]
    try:
        data = base64.b64decode(payload)
    except Exception:
        return None
    return "image/jpeg", data


async def _download_image_bytes_from_url(url: str) -> Tuple[str, bytes]:
    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg")
        data = resp.content
    if isinstance(content_type, str):
        content_type = content_type.split(";", 1)[0].strip() or "image/jpeg"
    return content_type, data


async def _download_image_bytes_ref(bot: Bot, ref: str) -> Tuple[str, bytes]:
    ref = str(ref or "").strip()
    if not ref:
        raise RuntimeError("图片引用为空")
    if ref.lower().startswith("data:image"):
        decoded = _decode_data_url(ref)
        if decoded:
            return decoded
        raise RuntimeError("无法解析 data URL")
    if ref.lower().startswith("base64://"):
        decoded = _decode_base64_ref(ref)
        if decoded:
            return decoded
        raise RuntimeError("无法解析 base64 图片")
    if ref.lower().startswith("http://") or ref.lower().startswith("https://"):
        return await _download_image_bytes_from_url(ref)
    url = await _onebot_get_image_url(bot, ref)
    if not url:
        raise RuntimeError("无法获取图片下载链接")
    return await _download_image_bytes_from_url(url)


async def _get_cached_image_bytes(
    bot: Bot, state: SessionState, message_id: int
) -> Tuple[str, bytes]:
    cached = state.image_cache.get(int(message_id))
    if not cached:
        raise RuntimeError("未找到缓存图片")
    if cached.data and cached.content_type:
        return cached.content_type, cached.data
    candidates: List[str] = []
    if cached.url:
        candidates.append(cached.url)
    if cached.file_id and cached.file_id not in candidates:
        candidates.append(cached.file_id)
    last_exc: Optional[Exception] = None
    for ref in candidates:
        try:
            content_type, data = await _download_image_bytes_ref(bot, ref)
            cached.content_type = content_type
            if len(data) <= _IMAGE_CACHE_MAX_BYTES:
                cached.data = data
            state.image_cache[int(message_id)] = cached
            return content_type, data
        except UnsupportedImageError:
            raise
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise RuntimeError(_safe_error_message(last_exc))
    raise RuntimeError("图片缓存失效")


async def _prefetch_cached_image(bot: Bot, state: SessionState, message_id: int) -> None:
    try:
        await _get_cached_image_bytes(bot, state, int(message_id))
    except UnsupportedImageError:
        return
    except Exception:
        return


async def _resolve_image_bytes(bot: Bot, state: SessionState, image_ref: str) -> Tuple[str, bytes]:
    cache_id = _parse_cache_image_ref(image_ref)
    if cache_id is not None:
        return await _get_cached_image_bytes(bot, state, cache_id)
    return await _download_image_bytes_ref(bot, image_ref)


async def _call_gemini_image(
    bot: Bot, prompt: str, image_ref: str, state: SessionState
) -> Tuple[bool, str]:
    client = _get_client()
    content_type, image_bytes = await _resolve_image_bytes(bot, state, image_ref)
    # 参考历史 + 当前指令 + 参考图，进行图片编辑
    prompt = _wrap_prompt_with_reference(state, prompt, current_label="当前指令")
    if _history_reference_only():
        contents = []
    else:
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


async def _call_gemini_image_chat(
    bot: Bot, prompt: str, image_ref: str, state: SessionState
) -> str:
    client = _get_client()
    content_type, image_bytes = await _resolve_image_bytes(bot, state, image_ref)
    # 参考历史 + 当前指令 + 参考图，只要文本回答（聊图）
    prompt = _wrap_prompt_with_reference(state, prompt, current_label="当前指令")
    if _history_reference_only():
        contents = []
    else:
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
    # 只回复当前指令：可附带历史参考文本
    prompt = _wrap_prompt_with_reference(state, prompt, current_label="当前指令")
    if _history_reference_only():
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
    else:
        contents = _history_to_gemini(state)
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )
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


async def _append_history(
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
    async with state.history_lock:
        if role == "model" and not user_name:
            user_name = _model_user_name()
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
        _prune_state(state, trim_history=False)
        await _maybe_compress_history(state)
        _prune_state(state)


history_collector = on_message(priority=99, block=False)
nlp_handler = on_message(priority=15, block=False)
avatar_handler = on_command("处理头像", priority=5)
chat_handler = on_command("技能", aliases={"聊天", "对话"}, priority=5)
weather_handler = on_command("天气", aliases={"查询天气", "查天气"}, priority=5)
travel_handler = on_command("旅行规划", aliases={"旅行计划", "行程规划", "旅行", "行程"}, priority=5)


@history_collector.handle()
async def _collect_history(bot: Bot, event: MessageEvent):
    session_id = _session_id(event)
    state = _get_state(session_id)

    # Keep a safe plaintext record (avoid leaking CQ/QQ ids to external APIs via history).
    text = _sanitize_cq_tokens(event.get_plaintext().strip())
    image_meta = _extract_first_image_meta(event.get_message())
    if image_meta:
        url, file_id = image_meta
        msg_id = getattr(event, "message_id", None)
        if isinstance(msg_id, int):
            ts = _event_ts(event)
            _cache_image_meta(state, msg_id, ts=ts, url=url, file_id=file_id)
            _notify_pending_image(state, str(event.get_user_id()), msg_id)
            task = state.image_cache_tasks.get(msg_id)
            if task is None or task.done():
                state.image_cache_tasks[msg_id] = asyncio.create_task(
                    _prefetch_cached_image(bot, state, msg_id)
                )

    if text:
        user_name = _event_user_name(event)
        await _append_history(
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
                return item.text, (item.user_name or None)
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
    reply_id = getattr(reply, "message_id", None)
    if isinstance(reply_id, int) and reply_id in state.image_cache:
        return _cache_image_ref(reply_id)
    reply_message = getattr(reply, "message", None)
    if reply_message:
        meta = _extract_first_image_meta(reply_message)
        if meta:
            url, file_id = meta
            if isinstance(reply_id, int):
                _cache_image_meta(
                    state,
                    reply_id,
                    ts=_event_ts(event),
                    url=url,
                    file_id=file_id,
                    update_last=False,
                )
                return _cache_image_ref(reply_id)
            return url or file_id
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
        msg_id = getattr(event, "message_id", None)
        if isinstance(msg_id, int) and msg_id in state.image_cache:
            return _cache_image_ref(msg_id)
        return current_image_url
    if target == "reply_image":
        reply = getattr(event, "reply", None)
        reply_id = getattr(reply, "message_id", None) if reply else None
        if isinstance(reply_id, int) and reply_id in state.image_cache:
            return _cache_image_ref(reply_id)
        return reply_image_url
    if target == "at_user":
        return _avatar_url(at_user) if at_user else None
    if target == "last_image":
        if state.last_image_id is None:
            return None
        if int(state.last_image_id) in state.image_cache:
            return _cache_image_ref(int(state.last_image_id))
        return None
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
        if int(msg_id) in state.image_cache:
            return _cache_image_ref(int(msg_id))
        return None
    if target == "wait_next":
        msg_id = await _wait_next_image(state, user_id, _WAIT_NEXT_IMAGE_SEC)
        if msg_id is None:
            return None
        if int(msg_id) in state.image_cache:
            return _cache_image_ref(int(msg_id))
        return None
    return None


def _collect_context_messages(
    state: SessionState,
    *,
    ts: float,
    limit: int,
    future: bool,
    current_text: str,
    current_message_id: Optional[int],
) -> List[str]:
    if limit <= 0:
        return []
    texts: List[str] = []
    items = state.history if future else reversed(state.history)
    for item in items:
        if item.role not in {"user", "model"}:
            continue
        if item.role == "user" and not item.to_bot:
            continue
        if future and item.ts <= ts:
            continue
        if not future and item.ts > ts:
            continue
        if current_message_id is not None and item.message_id == current_message_id:
            continue
        if item.role == "model":
            name = item.user_name or _model_user_name()
        else:
            name = item.user_name or "用户"
        line = _format_context_line(item.text, name)
        texts.append(line)
        if len(texts) >= limit:
            break
    if future:
        return texts
    return list(reversed(texts))


def _notify_pending_image(state: SessionState, user_id: str, message_id: int) -> None:
    waiter = state.pending_image_waiters.pop(user_id, None)
    if waiter and not waiter.done():
        waiter.set_result(int(message_id))


async def _wait_next_image(
    state: SessionState,
    user_id: str,
    timeout_sec: float,
) -> Optional[int]:
    waiter = state.pending_image_waiters.get(user_id)
    if waiter and not waiter.done():
        waiter.cancel()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[int] = loop.create_future()
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
    current_name = _event_user_name(event)
    current_message_id = getattr(event, "message_id", None)
    reply_text, reply_name = _extract_reply_context(event, state)

    prev_texts = _collect_context_messages(
        state,
        ts=ts,
        limit=max_prev,
        future=False,
        current_text=text,
        current_message_id=current_message_id if isinstance(current_message_id, int) else None,
    )
    future_texts: List[str] = []
    if max_future > 0:
        if wait_sec > 0:
            await asyncio.sleep(wait_sec)
        future_texts = _collect_context_messages(
            state,
            ts=ts,
            limit=max_future,
            future=True,
            current_text=text,
            current_message_id=current_message_id if isinstance(current_message_id, int) else None,
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
        for part in [_format_context_line(text, current_name), reply_line, *prev_texts, *future_texts]
        if part
    ]
    if not combined:
        return _format_context_line(text, current_name)
    return "\n".join(combined)


def _build_primary_intent_text(
    event: MessageEvent,
    state: SessionState,
    text: str,
) -> str:
    current_name = _event_user_name(event)
    reply_text, reply_name = _extract_reply_context(event, state)
    if not reply_text:
        return _format_context_line(text, current_name)
    if reply_text.strip() == text.strip():
        return _format_context_line(text, current_name)
    reply_line = (
        _format_context_line(reply_text, reply_name)
        if reply_name
        else f"回复内容: {reply_text}"
    )
    return "\n".join([_format_context_line(text, current_name), reply_line])


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
    at_users: List[str],
    state: SessionState,
) -> Optional[dict]:
    if not isinstance(intent, dict):
        return None
    action = str(intent.get("action", "")).strip().lower()
    if action not in _ALLOWED_ACTIONS:
        return None
    params = _intent_params(intent)
    target = str(intent.get("target", "")).strip().lower()

    if action == "ignore":
        return {"action": "ignore", "target": "none", "params": {}}

    if action == "chat":
        return {"action": "chat", "target": "none", "params": params}

    if action == "history_clear":
        return {"action": "history_clear", "target": "none", "params": {}}

    if action == "image_create":
        return {
            "action": action,
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
            elif at_users:
                target = "at_user"
            elif state.last_image_id is not None:
                target = "last_image"
            else:
                target = "wait_next"
        return {
            "action": action,
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
            "target": target,
            "params": params,
        }

    if action == "weather":
        raw_city = params.get("city")
        city = raw_city.strip() if isinstance(raw_city, str) else ""
        normalized_params: dict[str, object] = {}
        if city:
            normalized_params["city"] = city
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": "city",
            "params": normalized_params,
        }

    if action == "travel_plan":
        days = _coerce_int(params.get("days"))
        nights = _coerce_int(params.get("nights"))
        destination = ""
        raw_destination = params.get("destination") or params.get("city")
        if isinstance(raw_destination, str):
            destination = raw_destination.strip()
        normalized_params: dict[str, object] = {}
        if days is not None:
            normalized_params["days"] = days
        if nights is not None:
            normalized_params["nights"] = nights
        if destination:
            normalized_params["destination"] = destination
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": "trip",
            "params": normalized_params,
        }

    return {"action": action, "target": target or "none", "params": params}


async def _build_travel_plan_reply(
    intent: dict,
    state: SessionState,
    event: MessageEvent,
    *,
    raw_text: str,
) -> Optional[str]:
    params = _intent_params(intent)
    destination = params.get("destination")
    destination_text = destination.strip() if isinstance(destination, str) else ""
    days = _coerce_int(params.get("days"))
    nights = _coerce_int(params.get("nights"))
    raw_text = str(raw_text or "").strip()
    if raw_text:
        if not destination_text:
            destination_text = _extract_travel_destination(raw_text) or ""
        if days is None or nights is None:
            parsed_days, parsed_nights = _extract_travel_duration(raw_text)
            if days is None:
                days = parsed_days
            if nights is None:
                nights = parsed_nights
    if not destination_text:
        return "请告诉我目的地，例如：北京"
    normalized_params = dict(params)
    normalized_params["destination"] = destination_text
    if days is not None:
        normalized_params["days"] = days
    if nights is not None:
        normalized_params["nights"] = nights
    intent = dict(intent)
    intent["params"] = normalized_params
    reply = await _call_gemini_travel_plan(intent, state, raw_text=raw_text)
    if not reply:
        return None
    summary_parts: List[str] = [destination_text]
    if days is not None and nights is not None:
        summary_parts.append(f"{days}天{nights}晚")
    elif days is not None:
        summary_parts.append(f"{days}天")
    elif nights is not None:
        summary_parts.append(f"{nights}晚")
    summary = " ".join([part for part in summary_parts if part]).strip()
    cleaned_instruction = _strip_travel_duration(raw_text)
    if cleaned_instruction:
        cleaned_instruction = _collapse_spaces(cleaned_instruction.replace(destination_text, " "))
        for kw in _TRAVEL_KEYWORDS:
            cleaned_instruction = cleaned_instruction.replace(kw, " ")
        for kw in _TRAVEL_WEAK_KEYWORDS:
            cleaned_instruction = cleaned_instruction.replace(kw, " ")
        cleaned_instruction = (
            cleaned_instruction.replace("去", " ").replace("到", " ").replace("在", " ")
        )
        cleaned_instruction = re.sub(r"[，,。.!！?？/]", " ", cleaned_instruction)
        cleaned_instruction = _collapse_spaces(cleaned_instruction)
    if cleaned_instruction and cleaned_instruction not in summary:
        summary = f"{summary} 需求:{cleaned_instruction}"
    user_name = _event_user_name(event)
    await _append_history(
        state,
        "user",
        f"旅行规划：{summary}",
        user_id=str(event.get_user_id()),
        user_name=user_name,
        to_bot=True,
    )
    await _append_history(state, "model", reply)
    return reply


async def _dispatch_intent(
    bot: Bot,
    intent: dict,
    state: SessionState,
    event: MessageEvent,
    text: str,
    *,
    image_url: Optional[str],
    reply_image_url: Optional[str],
    at_users: List[str],
    send_func,
) -> None:
    # 意图分发：按 action 走不同处理链路
    action = str(intent.get("action", "ignore")).lower()
    if action == "ignore":
        return
    transition_sent = False
    if action in {
        "weather",
        "travel_plan",
        "avatar_get",
        "image_chat",
        "image_generate",
        "image_create",
    }:
        transition_text = _intent_transition_text(intent)
        transition_text = _format_reply_text(transition_text)
        if transition_text:
            await send_func(transition_text)
            transition_sent = True
            raw_params = intent.get("params")
            if isinstance(raw_params, dict):
                raw_params.pop("reply", None)
    user_name = _event_user_name(event)
    at_user = at_users[0] if at_users else None
    raw_message_text = ""
    if action in {
        "chat",
        "weather",
        "travel_plan",
        "image_create",
        "image_chat",
        "image_generate",
    }:
        raw_message_text = await _event_message_text(bot, event)

    if action == "chat":
        # 普通聊天（文本）
        raw_text = _strip_leading_command(
            raw_message_text,
            ("技能", "聊天", "对话"),
        )
        user_text = raw_text.strip()
        if not user_text:
            return
        reply_text, _ = _extract_reply_context(event, state)
        prompt = user_text
        if reply_text and reply_text.strip() and reply_text.strip() != user_text.strip():
            prompt = f"{user_text}\n回复内容: {reply_text.strip()}"
        try:
            reply = await _call_gemini_text(str(prompt), state)
            if not reply:
                return
            await _append_history(
                state,
                "user",
                user_text,
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            await _append_history(state, "model", reply)
            await _send_text_response(bot, event, send_func, reply)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP chat failed: {}", _safe_error_message(exc))
        return

    if action == "weather":
        # 天气查询
        params = _intent_params(intent)
        raw_city = params.get("city")
        query = raw_city.strip() if isinstance(raw_city, str) else ""
        if not query:
            raw_text = _strip_leading_command(
                raw_message_text,
                ("天气", "查询天气", "查天气"),
            )
            query = raw_text.strip()
        if not query:
            await send_func("请告诉我城市或地区，例如：天气 北京")
            return
        if not transition_sent:
            await _send_transition(action, send_func)
        try:
            messages = await _build_weather_messages(query)
            if not messages:
                return
            reply_text = "\n".join(messages)
            await _append_history(
                state,
                "user",
                f"天气：{query}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            await _append_history(state, "model", reply_text)
            await _send_text_response(bot, event, send_func, reply_text)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP weather failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action == "travel_plan":
        # 旅行规划
        raw_text = _strip_leading_command(
            raw_message_text,
            ("旅行规划", "旅行计划", "行程规划", "旅行", "行程"),
        )
        if not transition_sent:
            await _send_transition(action, send_func)
        try:
            reply = await _build_travel_plan_reply(intent, state, event, raw_text=raw_text)
            if not reply:
                return
            await _send_text_response(bot, event, send_func, reply)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP travel failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action == "history_clear":
        # 清空当前会话历史
        _clear_session_state(state)
        await send_func("已清除当前会话记录，可以继续聊啦。")
        return

    if action == "avatar_get":
        # 获取头像（发送者/群/指定QQ等）
        target = str(intent.get("target") or "").lower()
        params = _intent_params(intent)
        if target == "qq_avatar" and not params.get("qq"):
            await send_func("请提供 QQ 号。")
            return
        if not transition_sent:
            await _send_transition(action, send_func)
        if target == "at_user" and len(at_users) > 1:
            for qq in at_users:
                await send_func(_image_segment_from_result(_avatar_url(qq)))
            _mark_handled_request(state, event, text)
            return
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

    target = str(intent.get("target") or "").lower()
    params = _intent_params(intent)

    if action == "image_create":
        # 无参考图的图片生成
        prompt = raw_message_text.strip()
        if not prompt:
            await send_func("请告诉我你想生成什么样的图片。")
            return
        if not transition_sent:
            transition_text = _resolve_transition_text(action, intent)
            if transition_text:
                await send_func(transition_text)
        try:
            is_image, result = await _call_gemini_text_to_image(prompt, state)
            await _append_history(
                state,
                "user",
                f"生成图片：{prompt}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            if is_image:
                await _append_history(state, "model", "[已生成图片]")
                await send_func(_image_segment_from_result(result))
                _mark_handled_request(state, event, text)
            else:
                await _append_history(state, "model", result)
                await _send_text_response(bot, event, send_func, f"生成结果：{result}")
                _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP image create failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action not in {"image_chat", "image_generate"}:
        return

    prompt = _strip_leading_command(raw_message_text, ("处理头像",)).strip()
    if not prompt:
        prompt = raw_message_text.strip()
    if not prompt:
        await send_func("请把你的需求说清楚一点，例如：把这张图变成赛博朋克风。")
        return

    if target == "qq_avatar" and not params.get("qq"):
        await send_func("请提供 QQ 号。")
        return
    if target == "message_id" and not params.get("message_id"):
        await send_func("请提供消息 ID。")
        return
    if target == "wait_next":
        await send_func("请在60秒内发送图片。")

    if target == "at_user" and len(at_users) > 1:
        resolved = await asyncio.gather(
            *[_resolve_at_display_name(bot, event, qq) for qq in at_users],
            return_exceptions=True,
        )

        def _display_name(idx: int) -> str:
            value = resolved[idx]
            if isinstance(value, str) and value.strip():
                return value.strip()
            return f"用户{idx + 1}"

        if action == "image_chat":
            try:
                if not transition_sent:
                    await _send_transition(action, send_func)
                for idx, qq in enumerate(at_users):
                    display_name = _display_name(idx)
                    avatar_url = _avatar_url(qq)
                    reply = await _call_gemini_image_chat(bot, prompt, avatar_url, state)
                    if not reply:
                        continue
                    await _append_history(
                        state,
                        "user",
                        f"聊图({display_name})：{prompt}",
                        user_id=str(event.get_user_id()),
                        user_name=user_name,
                        to_bot=True,
                    )
                    await _append_history(state, "model", reply)
                    await _send_text_response(
                        bot, event, send_func, f"{display_name}：{reply}"
                    )
                _mark_handled_request(state, event, text)
            except UnsupportedImageError:
                await send_func("这个格式我处理不了，换张图片试试。")
            except Exception as exc:
                logger.error("NLP image chat failed: {}", _safe_error_message(exc))
                await send_func(f"出错了：{_safe_error_message(exc)}")
            return
        if action == "image_generate":
            try:
                if not transition_sent:
                    transition_text = _resolve_transition_text(action, intent)
                    if transition_text:
                        await send_func(transition_text)
                for idx, qq in enumerate(at_users):
                    display_name = _display_name(idx)
                    avatar_url = _avatar_url(qq)
                    is_image, result = await _call_gemini_image(bot, prompt, avatar_url, state)
                    await _append_history(
                        state,
                        "user",
                        f"处理头像({display_name})：{prompt}",
                        user_id=str(event.get_user_id()),
                        user_name=user_name,
                        to_bot=True,
                    )
                    if is_image:
                        await _append_history(state, "model", "[已生成图片]")
                        await send_func(f"{display_name} 已完成修改。")
                        await send_func(_image_segment_from_result(result))
                    else:
                        await _append_history(state, "model", result)
                        await _send_text_response(
                            bot,
                            event,
                            send_func,
                            f"{display_name} 修改结果：{result}",
                        )
                _mark_handled_request(state, event, text)
            except UnsupportedImageError:
                await send_func("这个格式我处理不了，换张图片试试。")
            except Exception as exc:
                logger.error("NLP image failed: {}", _safe_error_message(exc))
                await send_func(f"出错了：{_safe_error_message(exc)}")
            return

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
        # 聊图：有参考图，仅文本回答
        try:
            if not transition_sent:
                await _send_transition(action, send_func)
            reply = await _call_gemini_image_chat(bot, prompt, image_url, state)
            if not reply:
                return
            await _append_history(
                state,
                "user",
                f"聊图：{prompt}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            await _append_history(state, "model", reply)
            await _send_text_response(bot, event, send_func, reply)
            _mark_handled_request(state, event, text)
        except UnsupportedImageError:
            await send_func("这个格式我处理不了，换张图片试试。")
        except Exception as exc:
            logger.error("NLP image chat failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    try:
        # 处理头像/图片：有参考图，可能返回图片或文本
        if not transition_sent:
            transition_text = _resolve_transition_text(action, intent)
            if transition_text:
                await send_func(transition_text)
        is_image, result = await _call_gemini_image(bot, prompt, image_url, state)
        await _append_history(
            state,
            "user",
            f"处理头像：{prompt}",
            user_id=str(event.get_user_id()),
            user_name=user_name,
            to_bot=True,
        )
        if is_image:
            await _append_history(state, "model", "[已修改图片]")
            await send_func(_image_segment_from_result(result))
            _mark_handled_request(state, event, text)
        else:
            await _append_history(state, "model", result)
            await _send_text_response(bot, event, send_func, f"修改结果：{result}")
            _mark_handled_request(state, event, text)
    except UnsupportedImageError:
        await send_func("这个格式我处理不了，换张图片试试。")
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
    at_users: List[str],
) -> Optional[dict]:
    if not config.google_api_key:
        return None
    client = _get_client()
    system = _INTENT_SYSTEM_PROMPT
    user_prompt = (
        f"文本: {text}\n"
        f"消息包含图片: {has_image}\n"
        f"回复里有图片: {has_reply_image}\n"
        f"是否@用户: {bool(at_users)}\n"
        f"是否有最近图片: {bool(state.last_image_id)}\n"
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
    # Avoid leaking CQ/QQ ids to external APIs in intent classification.
    text = _sanitize_cq_tokens(event.get_plaintext().strip())
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
    image_meta = _extract_first_image_meta(event.get_message())
    image_url = (image_meta[0] or image_meta[1]) if image_meta else None
    if image_meta:
        url, file_id = image_meta
        msg_id = getattr(event, "message_id", None)
        if isinstance(msg_id, int):
            ts = _event_ts(event)
            _cache_image_meta(state, msg_id, ts=ts, url=url, file_id=file_id)
            task = state.image_cache_tasks.get(msg_id)
            if task is None or task.done():
                state.image_cache_tasks[msg_id] = asyncio.create_task(
                    _prefetch_cached_image(bot, state, msg_id)
                )
    at_users = _extract_at_users(event.get_message(), event.self_id)
    reply_image_url = _extract_reply_image_url(event, state)
    has_image = image_url is not None
    has_reply_image = reply_image_url is not None

    try:
        primary_text = _build_primary_intent_text(event, state, text)
        intent_raw = await _classify_intent(
            primary_text, state, has_image, has_reply_image, at_users
        )
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        return

    intent = _normalize_intent(intent_raw, has_image, has_reply_image, at_users, state)
    if not intent:
        try:
            intent_text = await _build_intent_text(event, state, text)
            if intent_text and intent_text != primary_text:
                intent_raw = await _classify_intent(
                    intent_text, state, has_image, has_reply_image, at_users
                )
                intent = _normalize_intent(
                    intent_raw, has_image, has_reply_image, at_users, state
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
        bot,
        intent,
        state,
        event,
        text,
        image_url=image_url,
        reply_image_url=reply_image_url,
        at_users=at_users,
        send_func=nlp_handler.send,
    )


async def _handle_command_via_intent(
    bot: Bot,
    event: MessageEvent,
    *,
    text: str,
    send_func,
) -> None:
    if not config.google_api_key:
        await send_func("未配置 GOOGLE_API_KEY")
        return
    # Avoid leaking CQ/QQ ids to external APIs in intent classification.
    text = _sanitize_cq_tokens(text)
    session_id = _session_id(event)
    state = _get_state(session_id)
    image_meta = _extract_first_image_meta(event.get_message())
    image_url = (image_meta[0] or image_meta[1]) if image_meta else None
    if image_meta:
        url, file_id = image_meta
        msg_id = getattr(event, "message_id", None)
        if isinstance(msg_id, int):
            ts = _event_ts(event)
            _cache_image_meta(state, msg_id, ts=ts, url=url, file_id=file_id)
            task = state.image_cache_tasks.get(msg_id)
            if task is None or task.done():
                state.image_cache_tasks[msg_id] = asyncio.create_task(
                    _prefetch_cached_image(bot, state, msg_id)
                )
    at_users = _extract_at_users(event.get_message(), event.self_id)
    reply_image_url = _extract_reply_image_url(event, state)
    has_image = image_url is not None
    has_reply_image = reply_image_url is not None
    try:
        intent_raw = await _classify_intent(
            text, state, has_image, has_reply_image, at_users
        )
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        await send_func("意图解析失败，请稍后再试。")
        return
    intent = _normalize_intent(intent_raw, has_image, has_reply_image, at_users, state)
    if not intent:
        await send_func(_clarify_intent_text(has_image))
        return
    reply = getattr(event, "reply", None)
    reply_id = getattr(reply, "message_id", None) if reply else None
    if reply_id is not None and isinstance(intent.get("params"), dict):
        intent["params"].setdefault("message_id", reply_id)
    await _dispatch_intent(
        bot,
        intent,
        state,
        event,
        text,
        image_url=image_url,
        reply_image_url=reply_image_url,
        at_users=at_users,
        send_func=send_func,
    )


@avatar_handler.handle()
async def handle_avatar(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    prompt = args.extract_plain_text().strip()
    if not prompt:
        await avatar_handler.finish("请告诉我你想怎么处理头像，例如：处理头像 变成赛博朋克风")
    await _handle_command_via_intent(
        bot,
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
        bot,
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
        bot,
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
        bot,
        event,
        text=f"旅行规划 {text}",
        send_func=travel_handler.send,
    )
