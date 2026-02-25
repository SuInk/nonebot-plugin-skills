from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from google import genai
from google.genai import types
from nonebot import get_driver, logger, on_command, on_message, on_notice
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent, MessageSegment
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from .config import config

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-skills",
    description="基于 Gemini 的技能插件，支持图片处理、聊天、天气、网页总结与番剧下载",
    usage="指令：处理头像 <指令> / 技能|聊天 <内容> / 天气 <城市> / 网页总结 <网页链接> / 番剧下载 <关键词> / 查看转发 / 查看撤回",
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
_BOTBR_MARK_RE = re.compile(r"\s*<\s*botbr\s*>\s*", re.I)
_SEND_REPLY_TAG_RE = re.compile(r"\[(?:REPLY|reply)\s*:\s*(\d+)\]")
_SEND_AT_TAG_RE = re.compile(r"\[(?:AT|at)\s*:\s*(all|\d+)\]")
_SEND_CQ_REPLY_RE = re.compile(r"\[CQ:reply,id=(\d+)(?:,[^\]]*)?\]", re.I)
_SEND_CQ_AT_RE = re.compile(r"\[CQ:at,qq=(all|\d+)(?:,[^\]]*)?\]", re.I)

_MODEL_REPLY_MAX_CHARS = 0
_WAIT_NEXT_IMAGE_SEC = 60.0
_IMAGE_CACHE_REF_PREFIX = "cache:"
_IMAGE_CACHE_MAX_BYTES = 8 * 1024 * 1024
_DEFAULT_IMAGE_DOWNLOAD_MAX_BYTES = 15 * 1024 * 1024
_DEFAULT_WEB_FETCH_MAX_BYTES = 2 * 1024 * 1024
_DEFAULT_WEB_EXTRACT_MAX_CHARS = 12000
_FORWARD_FETCH_NODE_LIMIT = 30
_FORWARD_FETCH_CHAR_LIMIT = 2200
_FORWARD_PROMPT_NODE_LIMIT = 12
_FORWARD_PROMPT_CHAR_LIMIT = 1000
_RECALL_MAX_ITEMS = 100

_WEB_SUMMARY_HOST_LABELS = {
    "github.com": "GitHub",
    "raw.githubusercontent.com": "GitHub",
    "v2ex.com": "V2EX",
    "linux.do": "LinuxDo",
    "news.ycombinator.com": "Hacker News",
    "hackernews.com": "Hacker News",
    "bilibili.com": "Bilibili",
    "b23.tv": "Bilibili",
    "zhihu.com": "知乎",
    "x.com": "X",
    "twitter.com": "X",
}
_WEB_SUMMARY_ALLOWED_ROOT_HOSTS = tuple(_WEB_SUMMARY_HOST_LABELS.keys())
_WEB_SUMMARY_BARE_HOST_PATTERN = (
    r"(?:[a-zA-Z0-9-]+\.)*"
    r"(?:github\.com|raw\.githubusercontent\.com|v2ex\.com|linux\.do|news\.ycombinator\.com|hackernews\.com|"
    r"bilibili\.com|b23\.tv|zhihu\.com|x\.com|twitter\.com)"
)
_URL_IN_TEXT_RE = re.compile(
    rf"(https?://[^\s<>{{}}\"'|\\^`]+|{_WEB_SUMMARY_BARE_HOST_PATTERN}(?:/[^\s<>{{}}\"'|\\^`]*)?)",
    re.I,
)
_URL_TRAILING_PUNCT = " \t\r\n,.;:!?)]}>\"'，。；：！？）】》"
_URL_LEADING_PUNCT = " \t\r\n<([{\"'（【《"
_ARIA2_DUPLICATE_INFOHASH_RE = re.compile(
    r"InfoHash\s+([0-9a-fA-F]{40})\s+is\s+already\s+registered",
    re.I,
)

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_DROP_BLOCK_RE = re.compile(
    r"<(script|style|noscript|svg|iframe|canvas)[^>]*>.*?</\1>",
    re.I | re.S,
)
_HTML_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|section|article|main|header|footer|aside|li|tr|h[1-6]|pre|blockquote|br)[^>]*>",
    re.I,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_WEB_COMMON_BLOCK_PATTERNS = (
    re.compile(r"<article[^>]*>(.*?)</article>", re.I | re.S),
    re.compile(r"<div[^>]*role=[\"']main[\"'][^>]*>(.*?)</div>", re.I | re.S),
    re.compile(r"<div[^>]*id=[\"']readme[\"'][^>]*>(.*?)</div>", re.I | re.S),
    re.compile(r"<main[^>]*>(.*?)</main>", re.I | re.S),
)
_WEB_SITE_BLOCK_PATTERNS = {
    "github.com": (
        re.compile(r"<article[^>]*>(.*?)</article>", re.I | re.S),
        re.compile(r"<div[^>]*id=[\"']readme[\"'][^>]*>(.*?)</div>", re.I | re.S),
        re.compile(r"<main[^>]*>(.*?)</main>", re.I | re.S),
    ),
    "v2ex.com": (
        re.compile(r"<div[^>]*id=[\"']Main[\"'][^>]*>(.*?)</div>", re.I | re.S),
    ),
    "linux.do": (
        re.compile(r"<article[^>]*class=[\"'][^\"']*topic-post[^\"']*[\"'][^>]*>(.*?)</article>", re.I | re.S),
        re.compile(r"<div[^>]*class=[\"'][^\"']*topic-body[^\"']*[\"'][^>]*>(.*?)</div>", re.I | re.S),
    ),
    "news.ycombinator.com": (
        re.compile(r"<table[^>]*class=[\"'][^\"']*itemlist[^\"']*[\"'][^>]*>(.*?)</table>", re.I | re.S),
    ),
}
_QQ_BOTBR_RULES = (
    "- 你可以使用多条消息回复，每两条消息之间使用<botbr>分隔，<botbr>前后不需要包含额外的换行和空格。\n"
    "- 除<botbr>外，消息中不应该包含其他类似的标记。\n"
    "- 不要使用markdown或者html，聊天软件不支持解析，换行请用换行符。\n"
)

_CHAT_SINGLE_SYSTEM_PROMPT = (
    "Role\n"
    "你是聊天助手，需保证事实准确。\n"
    "Goal\n"
    "直接给出可发送到QQ的自然语言回复。\n"
    "Rules\n"
    "- 只输出最终回复内容，不要 JSON，不要解释。\n"
    "- 信息不足时，先用一句话追问澄清。\n"
    "- 不要修改数字、时间、专有名词、链接、命令、QQ号、消息ID。\n"
    "- 输出纯文本，不使用 Markdown 或 HTML。\n"
    f"{_QQ_BOTBR_RULES}"
    "Output\n"
    "只输出最终回复内容。\n"
)

_IMAGE_CHAT_SYSTEM_PROMPT = (
    "Role\n"
    "你是asoul成员嘉然，正在进行图片内容对话。\n"
    "Goal\n"
    "只回答当前图片相关的指令或问题，给出简洁、口语化、可直接发QQ的回复。\n"
    "Rules\n"
    "- 只围绕当前图片和当前问题回复，不要补充已回复过的历史话题。\n"
    "- 先说可见事实，再给简短判断或建议；看不清就直接说明不确定。\n"
    "- 输出纯文本，不使用 Markdown 或代码块。\n"
    f"{_QQ_BOTBR_RULES}"
    "Output\n"
    "只输出最终回复内容。\n"
)

_TRAVEL_SYSTEM_PROMPT = (
    "Role\n"
    "你是旅行规划助手和asoul成员嘉然，给出清晰、实用、可执行的旅行建议。\n"
    "Goal\n"
    "根据对方消息输出可落地的旅行行程建议，适合QQ聊天阅读。\n"
    "Rules\n"
    "- 输出纯文本，不使用 Markdown 或代码块。\n"
    "- 结构清晰，尽量不要空行，包含景点/活动/用餐/交通/住宿/注意事项等要点。\n"
    "- 自动生成该城市最常见的规划天数。\n"
    f"{_QQ_BOTBR_RULES}"
    "Output\n"
    "只输出最终回复内容。\n"
)

_WEB_SUMMARY_SYSTEM_PROMPT = (
    "Role\n"
    "你是网页总结助手和asoul成员嘉然，负责总结网页正文。\n"
    "Goal\n"
    "先给重点结论，再给关键细节，信息不足时明确说明。\n"
    "Rules\n"
    "- 输出纯文本，不使用 Markdown 或代码块。\n"
    "- 语言简洁，适合QQ聊天。\n"
    f"{_QQ_BOTBR_RULES}"
    "Output\n"
    "只输出最终回复内容。\n"
)

_INTENT_SYSTEM_PROMPT = (
    "你是消息意图解析器，只输出 JSON，不要解释或补充说明。"
    "不要输出拒绝/免责声明/权限说明（例如“我无法访问账号”）。"
    "只输出单一 JSON 对象，格式如下："
    "{"
    "\"action\": \"chat|image_chat|image_generate|image_create|weather|avatar_get|user_info|travel_plan|web_summary|bangumi_download|forward_view|recall_view|history_clear|ignore\","
    "\"target\": \"message_image|reply_image|at_user|last_image|sender_avatar|group_avatar|qq_avatar|sender_user|qq_user|message_id|wait_next|city|trip|url|mikan|message_forward|reply_forward|recent_recall|none\","
    "\"params\": {\"qq\": \"string\", \"message_id\": \"int\", \"city\": \"string\","
    " \"destination\": \"string\", \"days\": \"int\", \"nights\": \"int\","
    " \"url\": \"string\", \"focus\": \"string\","
    " \"keyword\": \"string\", \"episode\": \"int\", \"latest\": \"bool\","
    " \"uri\": \"string\","
    " \"count\": \"int\", \"limit\": \"int\","
    " \"subtitle_group\": \"string\", \"mode\": \"download|subscribe|unsubscribe|list|check\","
    " \"subscribe\": \"bool\", \"reply\": \"string\"}"
    "}"
    "规则："
    "- action=ignore：target=none，params={}。"
    "- action=chat：普通聊天；target=none。"
    "- action=image_chat：聊这张图（不生成图）；target 用于选图：message_image/reply_image/at_user/last_image/sender_avatar/group_avatar/qq_avatar/message_id/wait_next。"
    "- action=image_generate：基于参考图生成/编辑；target 用于选图：同上。"
    "- action=image_create：无参考图生成；target=none。"
    "- action=weather：查询天气；target=city；params.city 为地点（没有就留空）。"
    "- action=avatar_get：获取头像；target 可为 sender_avatar/group_avatar/qq_avatar/at_user；target=qq_avatar 时填 params.qq。"
    "- action=user_info：查询用户信息；target 可为 sender_user/at_user/qq_user；target=qq_user 时填 params.qq。"
    "- action=travel_plan：旅行规划；target=trip；params.destination/days/nights 可填则填。"
    "- action=web_summary：网页总结；target=url；params.url 为链接（没有就留空），支持 github.com、v2ex.com、linux.do、news.ycombinator.com、bilibili.com、zhihu.com、x.com、twitter.com，params.focus 可选。"
    "- action=bangumi_download：从蜜柑下载番剧；target=mikan；params.keyword 为番剧名。params.episode 可选，不填时可用 latest=true 下载最新一集。params.subtitle_group 可选。"
    "- action=forward_view：查看转发聊天记录；target=message_forward 或 reply_forward；params.limit 可选。"
    "- action=recall_view：查看最近撤回消息；target=recent_recall；params.count 可选。"
    "- 若当前消息包含转发且用户要求“查看/展开/总结转发”，优先 action=forward_view。"
    "- 若用户提到“撤回了什么/看撤回消息”，优先 action=recall_view。"
    "- 如果消息是蜜柑链接、.torrent 链接或“种子文件下载”，优先输出 action=bangumi_download，并在 params.uri 填链接（有就填）。"
    "- bangumi_download 的 params.mode：download(默认)/subscribe(订阅并下载)/unsubscribe(取消订阅)/list(查看订阅)/check(检查订阅更新并下载新集)。"
    "- action=history_clear：清除当前会话历史；target=none。"
    "- target=message_id 时填写 params.message_id。"
    "- 上下文可能含多条消息，优先依据最后一条用户消息判断 action。"
    "- 如果完全无法判断，或与上述能力都不相关，输出 action=ignore，target=none，params={}。"
    "- params 仅在对应 target/场景需要时填写，其余为空对象。"
    "- params 不要携带无关字段，不要凭空虚构参数。"
    "- 若旅行或天气缺关键信息，仍输出对应 action，缺失字段留空"
    "- 当需要调用第三方工具且可能耗时（如 weather、image_create、image_generate、image_chat、avatar_get、travel_plan、web_summary、bangumi_download）时，可在 params.reply 中给等待/过渡语。"
    "- 若消息里 @ 多人，仍输出 target=at_user，系统会按顺序处理多个头像。"
    "- 上下文可能包含“昵称: 内容”的格式，需识别说话人。"
)

_BANGUMI_META_SYSTEM_PROMPT = (
    "Role\n"
    "你是番剧资源标题结构化解析器。\n"
    "Goal\n"
    "把每条标题解析为 episode(集数,int或null)、subtitle_group(字幕组名,string)、bangumi_title(番剧名,string)。\n"
    "Rules\n"
    "- 只输出 JSON，不要输出解释。\n"
    "- 不要输出 Markdown。\n"
    f"{_QQ_BOTBR_RULES}"
    "Output\n"
    "只输出 JSON。\n"
)

_BANGUMI_LATEST_SYSTEM_PROMPT = (
    "Role\n"
    "你是番剧 RSS 最新一集判定器。\n"
    "Goal\n"
    "根据条目标题判断最新一集正片，并在同一集多来源时优先 baha/巴哈/Bahamut/动画疯。\n"
    "Rules\n"
    "- 忽略预告、合集、特典、总集篇、PV、NCOP/NCED。\n"
    "- 只输出 JSON，不要输出解释。\n"
    f"{_QQ_BOTBR_RULES}"
    "Output\n"
    "只输出 {\"index\":序号或null,\"episode\":集数或null}。\n"
)

_DUPLICATE_TEXT_TTL_SEC = 60.0
_HISTORY_ITEM_MAX_CHARS = 400


class UnsupportedImageError(RuntimeError):
    pass

_SELF_ID_PATTERNS = [
    re.compile(r"^(作为|我作为)(一名|一个)?(LLM|大语言模型|语言模型|AI|模型).*?[，,。]\s*", re.I),
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


def _format_context_line(
    text: str,
    user_name: Optional[str],
    *,
    user_id: Optional[str] = None,
    message_id: Optional[int] = None,
) -> str:
    name = _normalize_user_name(user_name)
    meta_parts: List[str] = []
    qq = str(user_id or "").strip()
    if qq:
        meta_parts.append(f"qq={qq}")
    if isinstance(message_id, int):
        meta_parts.append(f"message_id={message_id}")
    meta = f"[{', '.join(meta_parts)}] " if meta_parts else ""
    if name:
        return f"{meta}{name}: {text}"
    return f"{meta}{text}"


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


def _normalize_botbr_markers(text: str) -> str:
    if not text:
        return ""
    return _BOTBR_MARK_RE.sub("<botbr>", str(text))


def _contains_botbr(text: str) -> bool:
    if not text:
        return False
    return _BOTBR_MARK_RE.search(str(text)) is not None


def _split_by_botbr(text: str) -> List[str]:
    normalized = _normalize_botbr_markers(text)
    if not normalized:
        return []
    if "<botbr>" not in normalized:
        return [normalized.strip()] if normalized.strip() else []
    parts = [part.strip() for part in normalized.split("<botbr>") if part.strip()]
    return parts


def _botbr_send_parts(text: str) -> List[str]:
    parts = _split_by_botbr(text)
    return parts


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
        return 150
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


def _command_starts() -> List[str]:
    try:
        starts = list(get_driver().config.command_start or [])
    except Exception:
        starts = ["/"]
    return [str(item) for item in starts if isinstance(item, str)]


def _strip_leading_command(text: str, words: Tuple[str, ...]) -> str:
    """Strip leading nonebot command prefix + command word (when it looks like a command).

    This is used to recover the raw user intent text for business logic after intent JSON routing.
    """
    value = str(text or "").strip()
    if not value:
        return ""
    starts = _command_starts()
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


def _extract_forward_ids(message: Message) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for seg in message:
        if getattr(seg, "type", "") != "forward":
            continue
        data = getattr(seg, "data", {}) or {}
        forward_id = str(data.get("id") or data.get("resid") or "").strip()
        if not forward_id or forward_id in seen:
            continue
        seen.add(forward_id)
        ids.append(forward_id)
    return ids


def _extract_reply_forward_ids(event: MessageEvent) -> List[str]:
    reply = getattr(event, "reply", None)
    if not reply:
        return []
    reply_message = getattr(reply, "message", None)
    if not isinstance(reply_message, Message):
        return []
    return _extract_forward_ids(reply_message)


def _segment_plain_text(segment: object) -> str:
    seg_type = ""
    data: dict[str, object] = {}
    if isinstance(segment, dict):
        seg_type = str(segment.get("type") or "")
        raw_data = segment.get("data")
        if isinstance(raw_data, dict):
            data = raw_data
    else:
        seg_type = str(getattr(segment, "type", "") or "")
        raw_data = getattr(segment, "data", None)
        if isinstance(raw_data, dict):
            data = raw_data
    if seg_type == "text":
        value = data.get("text")
        return str(value) if value is not None else ""
    if seg_type == "at":
        qq = str(data.get("qq") or "").strip()
        if qq == "all":
            return "@全体成员"
        return "@用户" if qq else ""
    if seg_type == "image":
        return "[图片]"
    if seg_type == "face":
        return "[表情]"
    if seg_type == "file":
        return "[文件]"
    if seg_type == "video":
        return "[视频]"
    if seg_type == "record":
        return "[语音]"
    if seg_type == "forward":
        return "[转发聊天记录]"
    return ""


def _message_obj_plain_text(message_obj: object) -> str:
    if isinstance(message_obj, str):
        return _sanitize_cq_tokens(message_obj)
    if isinstance(message_obj, Message):
        parts = [_segment_plain_text(seg) for seg in message_obj]
        return _normalize_prompt_text("".join(parts))
    if isinstance(message_obj, list):
        parts = [_segment_plain_text(seg) for seg in message_obj]
        return _normalize_prompt_text("".join(parts))
    if isinstance(message_obj, dict):
        if "type" in message_obj:
            return _normalize_prompt_text(_segment_plain_text(message_obj))
        msg = message_obj.get("message")
        if msg is not None:
            return _message_obj_plain_text(msg)
        content = message_obj.get("content")
        if content is not None:
            return _message_obj_plain_text(content)
    return ""


def _forward_nodes_from_payload(payload: object) -> List[Tuple[str, str]]:
    if not isinstance(payload, dict):
        return []
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    result: List[Tuple[str, str]] = []
    for node in messages:
        if not isinstance(node, dict):
            continue
        sender = node.get("sender")
        name = ""
        if isinstance(sender, dict):
            name = _normalize_user_name(sender.get("card") or sender.get("nickname"))
        if not name:
            name = "用户"
        content = node.get("content")
        if content is None:
            content = node.get("message")
        text = _message_obj_plain_text(content)
        text = _normalize_prompt_text(text)
        if not text:
            text = "[非文本消息]"
        result.append((name, text))
        if len(result) >= _FORWARD_FETCH_NODE_LIMIT:
            break
    return result


def _format_forward_nodes(
    nodes: List[Tuple[str, str]],
    *,
    limit_nodes: int,
    limit_chars: int,
    header: str,
) -> str:
    if not nodes:
        return ""
    display_nodes = nodes[: max(1, limit_nodes)]
    lines: List[str] = [header]
    for idx, (name, text) in enumerate(display_nodes):
        body = _truncate(text, 220).replace("\n", " ")
        lines.append(f"{idx + 1}. {name}: {body}")
    if len(nodes) > len(display_nodes):
        lines.append(f"...... 还有 {len(nodes) - len(display_nodes)} 条")
    merged = "\n".join(lines).strip()
    if len(merged) > limit_chars:
        merged = merged[: max(50, limit_chars - 20)].rstrip() + "\n...... 内容过长已截断"
    return merged


async def _fetch_forward_nodes(bot: Bot, forward_id: str) -> List[Tuple[str, str]]:
    if not forward_id:
        return []
    try:
        payload = await asyncio.wait_for(
            bot.get_forward_msg(id=forward_id),
            timeout=_request_timeout_seconds(),
        )
    except Exception:
        return []
    return _forward_nodes_from_payload(payload)


async def _build_forward_summary_text(
    bot: Bot,
    forward_ids: List[str],
    *,
    limit_nodes: int,
    limit_chars: int,
    header_prefix: str,
) -> str:
    if not forward_ids:
        return ""
    tasks = [_fetch_forward_nodes(bot, fid) for fid in forward_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    sections: List[str] = []
    for idx, result in enumerate(results):
        if isinstance(result, BaseException):
            continue
        nodes = result
        if not nodes:
            continue
        header = header_prefix if len(forward_ids) == 1 else f"{header_prefix}{idx + 1}"
        block = _format_forward_nodes(
            nodes,
            limit_nodes=limit_nodes,
            limit_chars=limit_chars,
            header=header,
        )
        if block:
            sections.append(block)
    return "\n\n".join(sections).strip()


async def _event_message_text(bot: Bot, event: MessageEvent) -> str:
    """Build a safe text representation of the current message for LLM/tool prompts.

    - Preserve @ mentions as @昵称 (best-effort), but never leak QQ numbers.
    - Strip CQ tokens if they appear as literal text.
    - Resolve merged-forward segments into plain text summaries.
    - Ignore other non-text segments (images, files, etc.) to avoid leaking URLs/IDs.
    """
    message = event.get_message()
    parts: List[str] = []
    at_list: List[str] = []
    placeholders: List[str] = []
    forward_ids: List[str] = []
    forward_placeholders: List[str] = []
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
        if seg.type == "forward":
            forward_id = str(seg.data.get("id") or seg.data.get("resid") or "").strip()
            if not forward_id:
                continue
            placeholder = f"__FWD_{len(forward_ids)}__"
            forward_ids.append(forward_id)
            forward_placeholders.append(placeholder)
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
    if forward_ids:
        summary_blocks = await asyncio.gather(
            *[
                _build_forward_summary_text(
                    bot,
                    [forward_id],
                    limit_nodes=_FORWARD_PROMPT_NODE_LIMIT,
                    limit_chars=_FORWARD_PROMPT_CHAR_LIMIT,
                    header_prefix="转发聊天记录",
                )
                for forward_id in forward_ids
            ],
            return_exceptions=True,
        )
        for idx, block in enumerate(summary_blocks):
            value = "[转发聊天记录]"
            if isinstance(block, str) and block.strip():
                value = block.strip()
            text = text.replace(forward_placeholders[idx], value)
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
    normalized = _normalize_botbr_markers(normalized)
    if not normalized:
        return
    if _contains_botbr(normalized):
        parts = _botbr_send_parts(normalized)
        if len(parts) > 1:
            merged = "\n\n".join(parts).strip()
            # For <botbr> multi-part replies, use total length before splitting.
            # If it is already over threshold, keep it merged so only one forward
            # message is sent instead of multiple forwarded batches.
            if merged and len(merged) > _forward_char_threshold():
                normalized = merged
            else:
                delay = _message_send_delay_sec()
                for idx, part in enumerate(parts):
                    await _send_text_response(bot, event, send_func, part)
                    if delay > 0 and idx < len(parts) - 1:
                        await asyncio.sleep(delay)
                return
        if len(parts) == 1:
            normalized = parts[0]
        else:
            return
    reply_id, normalized = _parse_send_reply_id(normalized)
    at_targets, normalized = _parse_send_at_targets(normalized)
    normalized = _format_reply_text(normalized)
    if reply_id is not None or at_targets:
        rich_message = _build_rich_message(
            normalized,
            reply_id=reply_id,
            at_targets=at_targets,
        )
        if rich_message is not None:
            await send_func(rich_message)
        return
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
    if action in {"bangumi_download"}:
        return "正在检索蜜柑并提交到 aria2，请稍候..."
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
    cleaned = _normalize_botbr_markers(cleaned)
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


def _normalize_reply_text(text: str) -> str:
    cleaned = _format_reply_text(text)
    cleaned = _compact_reply_lines(cleaned)
    return cleaned


def _is_skip_reply_text(text: str) -> bool:
    normalized = _format_reply_text(str(text or ""))
    if not normalized:
        return True
    return len(_split_by_botbr(normalized)) == 0


def _parse_send_reply_id(text: str) -> Tuple[Optional[int], str]:
    reply_id: Optional[int] = None

    def _collect(match: re.Match[str]) -> str:
        nonlocal reply_id
        if reply_id is None:
            try:
                value = int(match.group(1))
            except Exception:
                value = None
            if value is not None and value > 0:
                reply_id = value
        return ""

    cleaned = _SEND_CQ_REPLY_RE.sub(_collect, str(text or ""))
    cleaned = _SEND_REPLY_TAG_RE.sub(_collect, cleaned)
    return reply_id, cleaned


def _parse_send_at_targets(text: str) -> Tuple[List[str], str]:
    targets: List[str] = []
    seen: set[str] = set()

    def _collect(match: re.Match[str]) -> str:
        qq = str(match.group(1) or "").strip()
        if not qq:
            return ""
        if qq != "all" and not qq.isdigit():
            return ""
        if qq in seen:
            return ""
        seen.add(qq)
        targets.append(qq)
        return ""

    cleaned = _SEND_CQ_AT_RE.sub(_collect, str(text or ""))
    cleaned = _SEND_AT_TAG_RE.sub(_collect, cleaned)
    return targets, cleaned


def _build_rich_message(
    text: str,
    *,
    reply_id: Optional[int],
    at_targets: List[str],
) -> Optional[Message]:
    content = _format_reply_text(text)
    message = Message()
    if reply_id is not None and reply_id > 0:
        message += MessageSegment.reply(reply_id)
    for qq in at_targets:
        if qq == "all":
            message += MessageSegment.at("all")
        else:
            message += MessageSegment.at(int(qq))
    if content:
        if at_targets:
            message += MessageSegment.text(" ")
        message += MessageSegment.text(content)
    if len(message) <= 0:
        return None
    return message


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
class CachedImage:
    ts: float
    url: Optional[str] = None
    file_id: Optional[str] = None
    content_type: Optional[str] = None
    data: Optional[bytes] = None


@dataclass
class RecalledMessage:
    ts: float
    message_id: int
    sender_user_id: Optional[str] = None
    operator_user_id: Optional[str] = None
    sender_name: Optional[str] = None
    text: str = ""


@dataclass
class BangumiRelease:
    title: str
    page_url: str
    download_url: str
    guid: str
    size_bytes: Optional[int]
    release_date: str
    subtitle_group: str
    bangumi_title: str
    episode: Optional[int]


@dataclass
class BangumiNotifyTarget:
    group_id: Optional[int] = None
    user_id: Optional[int] = None


@dataclass
class SessionState:
    history: List[HistoryItem]
    last_image_id: Optional[int]
    image_cache: dict[int, CachedImage]
    image_cache_tasks: dict[int, asyncio.Task[None]]
    pending_image_waiters: dict[str, asyncio.Future[int]]
    handled_message_ids: dict[int, float]
    handled_texts: dict[str, float]
    recalled_messages: List[RecalledMessage]
    history_lock: asyncio.Lock
    bangumi_episode_group_map: dict[str, dict[int, str]]


_SESSIONS: dict[str, SessionState] = {}
_CLIENT: Optional[genai.Client] = None
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
_BANGUMI_SUBS_LOCK: Optional[asyncio.Lock] = None
_BANGUMI_DOWNLOAD_TASKS: dict[str, asyncio.Task[None]] = {}


def _bangumi_subs_lock() -> asyncio.Lock:
    global _BANGUMI_SUBS_LOCK
    if _BANGUMI_SUBS_LOCK is None:
        _BANGUMI_SUBS_LOCK = asyncio.Lock()
    return _BANGUMI_SUBS_LOCK


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


def _format_event_time_text(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return ""


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
            recalled_messages=[],
            history_lock=asyncio.Lock(),
            bangumi_episode_group_map={},
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


def _request_timeout_seconds() -> float:
    try:
        timeout = float(getattr(config, "request_timeout", 30.0))
    except Exception:
        timeout = 30.0
    if timeout <= 0:
        return 30.0
    return timeout


def _image_download_max_bytes() -> int:
    try:
        value = int(getattr(config, "image_download_max_bytes", 0))
    except Exception:
        value = 0
    if value <= 0:
        return _DEFAULT_IMAGE_DOWNLOAD_MAX_BYTES
    return max(256 * 1024, value)


def _web_fetch_max_bytes() -> int:
    try:
        value = int(getattr(config, "web_fetch_max_bytes", 0))
    except Exception:
        value = 0
    if value <= 0:
        return _DEFAULT_WEB_FETCH_MAX_BYTES
    return max(256 * 1024, value)


def _web_extract_max_chars() -> int:
    try:
        value = int(getattr(config, "web_extract_max_chars", 0))
    except Exception:
        value = 0
    if value <= 0:
        return _DEFAULT_WEB_EXTRACT_MAX_CHARS
    return max(1000, value)


def _aria2_timeout_seconds() -> float:
    try:
        timeout = float(getattr(config, "aria2_timeout", 20.0))
    except Exception:
        timeout = 20.0
    if timeout <= 0:
        return 20.0
    return timeout


def _aria2_rpc_url() -> str:
    raw = str(getattr(config, "aria2_rpc_url", "") or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    return raw


def _aria2_download_dir() -> str:
    raw = str(getattr(config, "aria2_download_dir", "") or "").strip()
    if not raw:
        raw = "download"
    plugin_dir = Path(__file__).resolve().parent
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = plugin_dir / path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return str(path)
    return str(path)


def _aria2_bt_trackers() -> List[str]:
    raw = getattr(config, "aria2_bt_trackers", [])
    values: List[str] = []
    if isinstance(raw, str):
        values = [part.strip() for part in re.split(r"[,;\r\n]+", raw) if part.strip()]
    elif isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                values.append(text)
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        parsed = urlsplit(value)
        if parsed.scheme not in {"udp", "http", "https", "ws", "wss"}:
            continue
        if not parsed.netloc:
            continue
        normalized = value.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _aria2_bt_max_peers() -> int:
    try:
        value = int(getattr(config, "aria2_bt_max_peers", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 0
    return max(20, min(1000, value))


def _aria2_disable_seed() -> bool:
    raw = getattr(config, "aria2_disable_seed", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "是", "要", "开启"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "不要", "关闭"}:
        return False
    return True


def _bangumi_auto_send_file() -> bool:
    try:
        return bool(getattr(config, "bangumi_auto_send_file", True))
    except Exception:
        return True


def _bangumi_auto_delete_after_send() -> bool:
    try:
        return bool(getattr(config, "bangumi_auto_delete_after_send", True))
    except Exception:
        return True


def _bangumi_send_text_reply() -> bool:
    try:
        return bool(getattr(config, "bangumi_send_text_reply", True))
    except Exception:
        return True


def _bangumi_progress_notify_interval_sec() -> float:
    try:
        value = float(getattr(config, "bangumi_progress_notify_interval_sec", 20.0))
    except Exception:
        value = 20.0
    if value <= 0:
        return 20.0
    return max(5.0, min(120.0, value))


def _bangumi_download_watch_timeout_sec() -> float:
    try:
        value = float(getattr(config, "bangumi_download_watch_timeout_sec", 21600.0))
    except Exception:
        value = 21600.0
    if value <= 0:
        return 21600.0
    return max(300.0, min(172800.0, value))


def _bangumi_torrent_max_bytes() -> int:
    try:
        value = int(getattr(config, "bangumi_torrent_max_bytes", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 4 * 1024 * 1024
    return max(512 * 1024, min(16 * 1024 * 1024, value))


def _bangumi_torrent_illegal_keywords() -> List[str]:
    raw = getattr(config, "bangumi_torrent_illegal_keywords", [])
    values: List[str] = []
    if isinstance(raw, str):
        values = [part.strip() for part in re.split(r"[,;\r\n]+", raw) if part.strip()]
    elif isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                values.append(text)
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(value)
    return result


def _bangumi_torrent_block_exts() -> set[str]:
    raw = getattr(config, "bangumi_torrent_block_exts", [])
    values: List[str] = []
    if isinstance(raw, str):
        values = [part.strip() for part in re.split(r"[,;\r\n]+", raw) if part.strip()]
    elif isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                values.append(text)
    result: set[str] = set()
    for value in values:
        lowered = value.lower()
        if not lowered:
            continue
        if not lowered.startswith("."):
            lowered = f".{lowered}"
        result.add(lowered)
    return result


def _mikan_base_url() -> str:
    value = str(getattr(config, "mikan_base_url", "") or "").strip()
    if not value:
        value = "https://mikanani.me"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "https://mikanani.me"
    return value.rstrip("/")


def _mikan_hosts() -> Tuple[str, ...]:
    hosts: List[str] = []
    base_host = str(urlsplit(_mikan_base_url()).hostname or "").strip().lower()
    if base_host:
        hosts.append(base_host)
    raw = getattr(config, "mikan_hosts", [])
    values: List[str] = []
    if isinstance(raw, str):
        values = [part.strip() for part in re.split(r"[,;\r\n]+", raw) if part.strip()]
    elif isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                values.append(text)
    for value in values:
        text = value
        if "://" not in text:
            text = f"https://{text}"
        parsed = urlsplit(text)
        host = str(parsed.hostname or "").strip().lower()
        if not host:
            continue
        hosts.append(host)
    deduped: List[str] = []
    seen: set[str] = set()
    for host in hosts:
        if host in seen:
            continue
        seen.add(host)
        deduped.append(host)
    if not deduped:
        return ("mikanani.me",)
    return tuple(deduped)


def _host_is_mikan(host: str) -> bool:
    normalized = str(host or "").strip().lower().rstrip(".")
    if not normalized:
        return False
    if "mikan" in normalized:
        return True
    return any(normalized == item or normalized.endswith(f".{item}") for item in _mikan_hosts())


def _mikan_search_limit() -> int:
    try:
        value = int(getattr(config, "mikan_search_limit", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 120
    return max(20, min(300, value))


def _bangumi_ai_classify_limit() -> int:
    try:
        value = int(getattr(config, "bangumi_ai_classify_limit", 0))
    except Exception:
        value = 0
    if value <= 0:
        return 60
    return max(5, min(120, value))


def _bangumi_subscription_enabled() -> bool:
    try:
        return bool(getattr(config, "bangumi_subscription_enable", True))
    except Exception:
        return True


def _bangumi_subscription_file() -> Path:
    raw = str(getattr(config, "bangumi_subscription_file", "") or "").strip()
    if not raw:
        raw = "data/nonebot_plugin_skills/bangumi_subscriptions.json"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    return path


def _extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    match = _URL_IN_TEXT_RE.search(text)
    if not match:
        return None
    url = match.group(1).strip()
    while url and url[-1] in _URL_TRAILING_PUNCT:
        url = url[:-1]
    while url and url[0] in _URL_LEADING_PUNCT:
        url = url[1:]
    return url or None


def _web_summary_root_host(host: str) -> str:
    normalized = str(host or "").strip().lower().rstrip(".")
    if not normalized:
        return ""
    for root in _WEB_SUMMARY_ALLOWED_ROOT_HOSTS:
        if normalized == root or normalized.endswith(f".{root}"):
            return root
    return ""


def _web_summary_site_name(host: str) -> str:
    root = _web_summary_root_host(host)
    if root:
        return _WEB_SUMMARY_HOST_LABELS.get(root, root)
    return ""


def _is_allowed_web_summary_host(host: str) -> bool:
    return bool(_web_summary_site_name(host))


def _normalize_web_summary_url(raw_url: str) -> Optional[str]:
    candidate = str(raw_url or "").strip()
    if not candidate:
        return None
    candidate = candidate.strip(" \t\r\n<>[]()\"'")
    if "://" not in candidate:
        host_part = candidate.split("/", 1)[0].strip()
        if ":" in host_part:
            host_part = host_part.split(":", 1)[0].strip()
        if _is_allowed_web_summary_host(host_part):
            candidate = f"https://{candidate}"

    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None
    if parsed.username or parsed.password:
        return None
    host = (parsed.hostname or "").strip()
    if not _is_allowed_web_summary_host(host):
        return None
    if parsed.port not in {None, 80, 443}:
        return None
    netloc = parsed.netloc
    if not netloc:
        return None
    path = parsed.path or "/"
    normalized = urlunsplit((scheme, netloc, path, parsed.query, ""))
    return normalized


def _web_summary_source_label(url: str) -> str:
    parsed = urlsplit(url or "")
    return _web_summary_site_name(parsed.hostname or "") or "网页"


def _resolve_web_summary_url(intent: dict, raw_text: str) -> str:
    params = _intent_params(intent)
    raw_url = params.get("url")
    url = raw_url.strip() if isinstance(raw_url, str) else ""
    if not url:
        url = _extract_first_url(raw_text) or ""
    return _normalize_web_summary_url(url) or ""


def _rewrite_github_blob_url_for_fetch(url: str) -> str:
    parsed = urlsplit(url or "")
    if _web_summary_root_host(parsed.hostname or "") != "github.com":
        return url
    path = parsed.path or ""
    marker = "/blob/"
    if marker not in path:
        return url
    rewritten_path = path.replace(marker, "/raw/", 1)
    return urlunsplit((parsed.scheme, parsed.netloc, rewritten_path, parsed.query, ""))


def _extract_plain_page_text(text: str) -> str:
    if not text:
        return ""
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = _collapse_spaces(raw_line)
        if line:
            lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return ""
    limit = _web_extract_max_chars()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "\n[内容已截断]"
    return cleaned


def _normalize_github_url(raw_url: str) -> Optional[str]:
    # Backward compatibility wrapper
    return _normalize_web_summary_url(raw_url)


def _normalize_content_type(value: object, *, default: str = "image/jpeg") -> str:
    if isinstance(value, str):
        cleaned = value.split(";", 1)[0].strip().lower()
        if cleaned:
            return cleaned
    return default


def _validate_image_bytes(content_type: str, data: bytes) -> Tuple[str, bytes]:
    normalized_type = _normalize_content_type(content_type)
    if normalized_type in {"application/octet-stream", "binary/octet-stream"}:
        normalized_type = "image/jpeg"
    if not normalized_type.startswith("image/"):
        raise UnsupportedImageError("只支持图片内容。")
    if not data:
        raise UnsupportedImageError("图片内容为空。")
    limit = _image_download_max_bytes()
    if len(data) > limit:
        limit_mb = max(1, limit // (1024 * 1024))
        raise UnsupportedImageError(f"图片过大，请限制在 {limit_mb}MB 内。")
    return normalized_type, data


def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(_request_timeout_seconds()),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _HTTP_CLIENT


@get_driver().on_shutdown
async def _close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        return
    try:
        await _HTTP_CLIENT.aclose()
    finally:
        _HTTP_CLIENT = None


def _history_reference_only() -> bool:
    try:
        return bool(getattr(config, "history_reference_only", True))
    except Exception:
        return True


def _clamp_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


def _chat_knowledge_temperature() -> float:
    value = getattr(config, "chat_knowledge_temperature", 0.2)
    return _clamp_float(value, default=0.2, minimum=0.0, maximum=2.0)


def _history_item_label(item: HistoryItem) -> str:
    if item.role == "model":
        return item.user_name or _model_user_name()
    return item.user_name or "用户"


def _history_item_to_line(item: HistoryItem) -> str:
    text = _ensure_plain_text(str(item.text))
    if not text:
        return ""
    text = _truncate(text, _HISTORY_ITEM_MAX_CHARS)
    return _format_context_line(
        text,
        _history_item_label(item),
        user_id=item.user_id,
        message_id=item.message_id,
    )


def _build_history_reference_text(state: SessionState) -> str:
    lines: List[str] = []
    for item in state.history:
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
    if state.recalled_messages:
        state.recalled_messages = [
            item for item in state.recalled_messages if item.ts >= cutoff
        ]
        if len(state.recalled_messages) > _RECALL_MAX_ITEMS:
            state.recalled_messages = state.recalled_messages[-_RECALL_MAX_ITEMS:]


def _clear_session_state(state: SessionState) -> None:
    state.history = []
    state.last_image_id = None
    state.image_cache = {}
    if state.image_cache_tasks:
        for task in state.image_cache_tasks.values():
            if task and not task.done():
                task.cancel()
    state.image_cache_tasks = {}
    if state.pending_image_waiters:
        for waiter in state.pending_image_waiters.values():
            if not waiter.done():
                waiter.cancel()
    state.pending_image_waiters = {}
    state.handled_message_ids = {}
    state.handled_texts = {}
    state.recalled_messages = []
    state.bangumi_episode_group_map = {}


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


def _group_role_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "owner":
        return "群主"
    if text == "admin":
        return "管理员"
    if text == "member":
        return "群成员"
    return ""


def _sex_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "male":
        return "男"
    if text == "female":
        return "女"
    if text in {"unknown", ""}:
        return "未知"
    return str(value or "").strip()


def _format_unix_time(value: object) -> str:
    timestamp = _coerce_int(value)
    if timestamp is None or timestamp <= 0:
        return ""
    return _format_event_time_text(float(timestamp))


async def _fetch_user_profile(
    bot: Bot,
    event: MessageEvent,
    qq: str,
) -> Tuple[dict[str, object], dict[str, object]]:
    group_info: dict[str, object] = {}
    stranger_info: dict[str, object] = {}
    user_id = _coerce_int(qq)
    if user_id is None:
        return group_info, stranger_info
    if isinstance(event, GroupMessageEvent):
        try:
            data = await bot.get_group_member_info(
                group_id=event.group_id,
                user_id=user_id,
                no_cache=True,
            )
            if isinstance(data, dict):
                group_info = data
        except Exception:
            group_info = {}
    try:
        data = await bot.get_stranger_info(user_id=user_id, no_cache=True)
        if isinstance(data, dict):
            stranger_info = data
    except Exception:
        stranger_info = {}
    return group_info, stranger_info


def _format_user_info_text(
    qq: str,
    *,
    group_info: dict[str, object],
    stranger_info: dict[str, object],
) -> str:
    qq_text = str(qq).strip()
    card = str(group_info.get("card") or "").strip()
    nickname = str(group_info.get("nickname") or stranger_info.get("nickname") or "").strip()
    display_name = card or nickname or "未知"
    lines = [
        f"用户信息 QQ:{qq_text}",
        f"显示名：{display_name}",
    ]
    if card:
        lines.append(f"群名片：{card}")
    if nickname:
        lines.append(f"昵称：{nickname}")
    sex = _sex_text(group_info.get("sex") or stranger_info.get("sex"))
    if sex:
        lines.append(f"性别：{sex}")
    age = _coerce_int(group_info.get("age"))
    if age is None:
        age = _coerce_int(stranger_info.get("age"))
    if age is not None and age >= 0:
        lines.append(f"年龄：{age}")
    role = _group_role_text(group_info.get("role"))
    if role:
        lines.append(f"群角色：{role}")
    title = str(group_info.get("title") or "").strip()
    if title:
        lines.append(f"头衔：{title}")
    area = str(stranger_info.get("area") or "").strip()
    if area:
        lines.append(f"地区：{area}")
    level = str(group_info.get("level") or "").strip()
    if level:
        lines.append(f"群等级：{level}")
    join_time = _format_unix_time(group_info.get("join_time"))
    if join_time:
        lines.append(f"入群时间：{join_time}")
    last_sent_time = _format_unix_time(group_info.get("last_sent_time"))
    if last_sent_time:
        lines.append(f"最后发言：{last_sent_time}")
    return "\n".join(lines)


async def _build_user_info_reply(
    bot: Bot,
    event: MessageEvent,
    intent: dict,
    *,
    at_users: List[str],
) -> Optional[str]:
    target = str(intent.get("target") or "").lower()
    params = _intent_params(intent)
    qq_targets: List[str] = []
    if target == "qq_user":
        raw_qq = params.get("qq")
        qq = str(raw_qq or "").strip()
        if not qq or not qq.isdigit():
            return "请提供有效的 QQ 号。"
        qq_targets = [qq]
    elif target == "at_user":
        if at_users:
            qq_targets = list(at_users)
        else:
            raw_qq = params.get("qq")
            qq = str(raw_qq or "").strip()
            if qq and qq.isdigit():
                qq_targets = [qq]
            else:
                return "请先@要查询的用户，或提供 QQ 号。"
    else:
        qq_targets = [str(event.get_user_id())]
    if not qq_targets:
        return None
    messages: List[str] = []
    for qq in qq_targets[:5]:
        group_info, stranger_info = await _fetch_user_profile(bot, event, qq)
        messages.append(
            _format_user_info_text(
                qq,
                group_info=group_info,
                stranger_info=stranger_info,
            )
        )
    return "\n\n".join(messages).strip()


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
    client = _get_http_client()
    resp = await client.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=params,
        timeout=_request_timeout_seconds(),
    )
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
    client = _get_http_client()
    resp = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=_request_timeout_seconds(),
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return None
    return data


_MikanEpisodePatterns = (
    re.compile(r"(?:第\s*)?([0-9]{1,3})\s*(?:集|话|話)\b", re.I),
    re.compile(r"\b(?:ep|e)\s*([0-9]{1,3})\b", re.I),
    re.compile(r"\[([0-9]{1,3})(?:v[0-9]+)?(?:\s*(?:END|FINAL))?\]", re.I),
)
_BANGUMI_GROUP_RE = re.compile(r"^[\[\(【]([^【】\[\]\(\)]{1,24}(?:字幕组|字幕社|Sub|SUB|字幕))[\]）】]", re.I)
_BANGUMI_SUB_GROUP_HINT_RE = re.compile(r"([A-Za-z0-9\u4e00-\u9fff]{1,24}(?:字幕组|字幕社|Sub|SUB|字幕))")
_BANGUMI_LATEST_HINTS = ("最新", "最新一集", "最新集", "latest", "newest")
_BANGUMI_SUB_CHECK_HINTS = ("检查订阅", "更新订阅", "订阅更新", "check")
_BANGUMI_SUB_LIST_HINTS = ("订阅列表", "查看订阅", "列出订阅", "list")
_BANGUMI_SUB_REMOVE_HINTS = ("取消订阅", "退订", "删除订阅", "unsubscribe", "remove")
_BANGUMI_SUB_ADD_HINTS = ("订阅下载", "添加订阅", "订阅", "subscribe")
_BANGUMI_COMMAND_WORDS = ("番剧下载", "蜜柑下载", "订阅下载", "番剧订阅", "追番下载")
_BANGUMI_BAHA_SOURCE_HINTS = ("baha", "bahamut", "巴哈", "動畫瘋", "动画疯")


def _extract_json_array(text: str) -> List[dict]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parsed: Optional[object] = None
    try:
        parsed = json.loads(raw)
    except Exception:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                parsed = None
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _coerce_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "y", "on", "是", "要", "开启"}:
            return True
        if cleaned in {"0", "false", "no", "n", "off", "否", "不要", "关闭"}:
            return False
    return None


def _normalize_bangumi_keyword(value: str) -> str:
    cleaned = _collapse_spaces(str(value or ""))
    cleaned = cleaned.strip(" \t\r\n,，。.!！？?？:：;；[]【】()（）")
    return cleaned


def _normalize_subtitle_group(value: str) -> str:
    cleaned = str(value or "").strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _bangumi_episode_key(keyword: str) -> str:
    return _normalize_bangumi_keyword(keyword).lower()


def _remember_episode_group(
    state: SessionState,
    *,
    keyword: str,
    episode: Optional[int],
    subtitle_group: str,
) -> None:
    if episode is None:
        return
    group_text = str(subtitle_group or "").strip()
    if not group_text:
        return
    key = _bangumi_episode_key(keyword)
    if not key:
        return
    mapping = state.bangumi_episode_group_map.get(key)
    if mapping is None:
        mapping = {}
        state.bangumi_episode_group_map[key] = mapping
    mapping[int(episode)] = group_text


def _known_episode_group(
    state: SessionState,
    *,
    keyword: str,
    episode: Optional[int],
) -> str:
    if episode is None:
        return ""
    key = _bangumi_episode_key(keyword)
    if not key:
        return ""
    mapping = state.bangumi_episode_group_map.get(key) or {}
    known = mapping.get(int(episode))
    if not isinstance(known, str):
        return ""
    return known.strip()


def _strip_xml_tag_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _xml_child_text(item: ET.Element, name: str) -> str:
    for child in list(item):
        if _strip_xml_tag_ns(child.tag) != name:
            continue
        text = child.text or ""
        cleaned = text.strip()
        if cleaned:
            return cleaned
    return ""


def _xml_enclosure_url(item: ET.Element) -> str:
    for child in list(item):
        if _strip_xml_tag_ns(child.tag) != "enclosure":
            continue
        url = str(child.attrib.get("url") or "").strip()
        if url:
            return url
    return ""


def _xml_enclosure_length(item: ET.Element) -> Optional[int]:
    for child in list(item):
        if _strip_xml_tag_ns(child.tag) != "enclosure":
            continue
        return _coerce_int(child.attrib.get("length"))
    return None


def _extract_release_date(url: str) -> str:
    match = re.search(r"/Download/([0-9]{8})/", str(url or ""))
    if not match:
        return ""
    return match.group(1)


def _extract_episode_from_title(title: str) -> Optional[int]:
    text = str(title or "")
    if not text:
        return None
    for pattern in _MikanEpisodePatterns:
        match = pattern.search(text)
        if not match:
            continue
        value = _coerce_int(match.group(1))
        if value is None:
            continue
        if 0 <= value <= 999:
            return value
    return None


def _extract_subtitle_group_from_title(title: str) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    match = _BANGUMI_GROUP_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _BANGUMI_SUB_GROUP_HINT_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def _extract_bangumi_title_from_release_title(title: str, subtitle_group: str) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    if subtitle_group:
        text = re.sub(
            rf"^[\[\(【]\s*{re.escape(subtitle_group)}\s*[\]）】]\s*",
            "",
            text,
            flags=re.I,
        )
    text = text.strip()
    parts = re.split(r"\[[^\]]*\]|【[^】]*】|\([^)]+\)|（[^）]+）", text)
    for part in parts:
        cleaned = _collapse_spaces(part)
        if len(cleaned) < 2:
            continue
        if re.fullmatch(r"[0-9A-Za-z._\- ]+", cleaned):
            continue
        return cleaned
    if parts:
        fallback = _collapse_spaces(parts[0])
        if fallback:
            return fallback
    return text


def _torrent_display_name(raw_name: str) -> str:
    text = _collapse_spaces(str(raw_name or ""))
    return text or "种子文件"


def _normalize_bangumi_uri(raw_uri: str) -> str:
    value = str(raw_uri or "").strip()
    if not value:
        return ""
    value = value.strip(" \t\r\n<>[]()\"'")
    lowered = value.lower()
    if lowered.startswith("magnet:?"):
        return value
    if "://" not in value:
        host = value.split("/", 1)[0].strip().lower()
        if _host_is_mikan(host):
            value = f"https://{value}"
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return ""
    if parsed.username or parsed.password:
        return ""
    netloc = parsed.netloc
    if not netloc:
        return ""
    path = parsed.path or "/"
    normalized = urlunsplit((scheme, netloc, path, parsed.query, ""))
    return normalized


def _is_mikan_url(url: str) -> bool:
    parsed = urlsplit(str(url or ""))
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False
    return _host_is_mikan(host)


def _is_torrent_url(url: str) -> bool:
    parsed = urlsplit(str(url or ""))
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    path = str(parsed.path or "").lower()
    if path.endswith(".torrent"):
        return True
    query = str(parsed.query or "").lower()
    return ".torrent" in query


def _is_magnet_uri(url: str) -> bool:
    return str(url or "").strip().lower().startswith("magnet:?xt=urn:btih:")


def _is_bangumi_direct_uri(url: str) -> bool:
    normalized = _normalize_bangumi_uri(url)
    if not normalized:
        normalized = str(url or "").strip()
    if _is_magnet_uri(normalized):
        return True
    return _is_mikan_url(normalized) or _is_torrent_url(normalized)


def _extract_bangumi_uri_from_text(raw_text: str) -> str:
    first = _extract_first_url(str(raw_text or "")) or ""
    normalized = _normalize_bangumi_uri(first)
    if normalized and _is_bangumi_direct_uri(normalized):
        return normalized
    if _is_magnet_uri(first):
        return first.strip()
    return ""


def _extract_first_file_meta(message: Message) -> Optional[dict[str, str]]:
    for seg in message:
        if seg.type != "file":
            continue
        name = str(seg.data.get("name") or "").strip()
        url = str(seg.data.get("url") or "").strip()
        file_ref = str(seg.data.get("file") or "").strip()
        path = str(seg.data.get("path") or "").strip()
        if not url and file_ref and (file_ref.lower().startswith("http://") or file_ref.lower().startswith("https://")):
            url = file_ref
        return {
            "name": name,
            "url": url,
            "file": file_ref,
            "path": path,
        }
    return None


def _looks_like_torrent_file_name(name: str) -> bool:
    return str(name or "").strip().lower().endswith(".torrent")


def _build_bangumi_nlp_hint(message: Message, plain_text: str) -> str:
    uri = _extract_bangumi_uri_from_text(plain_text)
    if uri:
        return f"番剧下载 {uri}"
    file_meta = _extract_first_file_meta(message)
    if not file_meta:
        return ""
    file_name = str(file_meta.get("name") or file_meta.get("file") or "").strip()
    file_url = _normalize_bangumi_uri(file_meta.get("url") or file_meta.get("file") or "")
    if file_url and (_is_torrent_url(file_url) or _is_mikan_url(file_url)):
        return f"番剧下载 {file_url}"
    local_path = file_meta.get("path") or file_meta.get("file") or ""
    if _maybe_local_path(local_path) is not None and _looks_like_torrent_file_name(local_path):
        return f"番剧下载 种子文件 {_torrent_display_name(file_name)}"
    if _looks_like_torrent_file_name(file_name):
        return f"番剧下载 种子文件 {_torrent_display_name(file_name)}"
    return ""


def _maybe_local_path(value: str) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser()
    except Exception:
        return None
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    if path.exists() and path.is_file():
        return path
    return None


async def _download_binary_with_limit(url: str, *, max_bytes: int) -> bytes:
    client = _get_http_client()
    chunks: List[bytes] = []
    total = 0
    async with client.stream("GET", url, timeout=_request_timeout_seconds()) as resp:
        resp.raise_for_status()
        content_length = _coerce_int(resp.headers.get("content-length"))
        if content_length is not None and content_length > max_bytes:
            max_mb = max(1, max_bytes // (1024 * 1024))
            raise RuntimeError(f"文件过大，请限制在 {max_mb}MB 内。")
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                max_mb = max(1, max_bytes // (1024 * 1024))
                raise RuntimeError(f"文件过大，请限制在 {max_mb}MB 内。")
            chunks.append(chunk)
    return b"".join(chunks)


def _bdecode_value(data: bytes, index: int = 0) -> Tuple[object, int]:
    if index >= len(data):
        raise ValueError("unexpected end")
    token = data[index : index + 1]
    if token == b"i":
        end = data.find(b"e", index + 1)
        if end == -1:
            raise ValueError("invalid int")
        number = int(data[index + 1 : end])
        return number, end + 1
    if token == b"l":
        idx = index + 1
        values = []
        while idx < len(data) and data[idx : idx + 1] != b"e":
            item, idx = _bdecode_value(data, idx)
            values.append(item)
        if idx >= len(data):
            raise ValueError("invalid list")
        return values, idx + 1
    if token == b"d":
        idx = index + 1
        values = {}
        while idx < len(data) and data[idx : idx + 1] != b"e":
            key, idx = _bdecode_value(data, idx)
            val, idx = _bdecode_value(data, idx)
            values[key] = val
        if idx >= len(data):
            raise ValueError("invalid dict")
        return values, idx + 1
    if 48 <= token[0] <= 57:
        colon = data.find(b":", index)
        if colon == -1:
            raise ValueError("invalid bytes")
        size = int(data[index:colon])
        start = colon + 1
        end = start + size
        if end > len(data):
            raise ValueError("bytes out of range")
        return data[start:end], end
    raise ValueError("invalid token")


def _bdecode_torrent(data: bytes) -> Optional[dict]:
    try:
        value, idx = _bdecode_value(data, 0)
    except Exception:
        return None
    if idx != len(data):
        return None
    if not isinstance(value, dict):
        return None
    return value


def _dict_get(mapping: object, key: str) -> Optional[object]:
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return mapping.get(key)
    key_bytes = key.encode("utf-8", errors="ignore")
    if key_bytes in mapping:
        return mapping.get(key_bytes)
    return None


def _torrent_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        for encoding in ("utf-8", "gb18030", "big5"):
            try:
                return value.decode(encoding).strip()
            except Exception:
                continue
        return value.decode("utf-8", errors="ignore").strip()
    return str(value or "").strip()


def _torrent_file_paths(meta: dict) -> List[str]:
    info = _dict_get(meta, "info")
    if not isinstance(info, dict):
        return []
    result: List[str] = []
    top_name = _torrent_text(_dict_get(info, "name.utf-8") or _dict_get(info, "name"))
    files = _dict_get(info, "files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            raw_parts = _dict_get(item, "path.utf-8") or _dict_get(item, "path")
            if not isinstance(raw_parts, list):
                continue
            parts = [_torrent_text(part) for part in raw_parts]
            parts = [part for part in parts if part]
            if not parts:
                continue
            combined = "/".join(parts)
            if top_name and not combined.startswith(top_name):
                combined = f"{top_name}/{combined}"
            result.append(combined)
        return result
    if top_name:
        return [top_name]
    return []


def _check_torrent_compliance(torrent_bytes: bytes, *, display_name: str = "") -> Tuple[bool, str]:
    if not torrent_bytes:
        return False, "种子文件为空，已拒绝下载。"
    meta = _bdecode_torrent(torrent_bytes)
    if meta is None:
        return False, "种子文件无法解析，已拒绝下载。"
    candidates = _torrent_file_paths(meta)
    if not candidates and display_name:
        candidates = [_torrent_display_name(display_name)]

    block_keywords = _bangumi_torrent_illegal_keywords()
    if block_keywords:
        lowered_pairs = [(word, word.lower()) for word in block_keywords]
        for item in candidates:
            lowered_item = item.lower()
            for original, lowered_word in lowered_pairs:
                if lowered_word and lowered_word in lowered_item:
                    return False, f"种子文件疑似违规，命中关键词：{original}"

    block_exts = _bangumi_torrent_block_exts()
    if block_exts:
        for item in candidates:
            suffix = Path(item).suffix.lower()
            if suffix and suffix in block_exts:
                return False, f"种子文件疑似违规，包含危险类型：{suffix}"
    return True, ""


async def _resolve_mikan_download_uri(url: str) -> str:
    normalized = _normalize_bangumi_uri(url)
    if not normalized or not _is_mikan_url(normalized):
        return normalized
    if _is_torrent_url(normalized):
        return normalized
    client = _get_http_client()
    try:
        resp = await client.get(normalized, timeout=_request_timeout_seconds())
        resp.raise_for_status()
    except Exception:
        return normalized
    html_text = str(resp.text or "")
    base = urlsplit(normalized)
    link_matches = re.findall(
        r"https?://[^\"'\s>]+\.torrent|/[^\"'\s>]+\.torrent",
        html_text,
        flags=re.I,
    )
    for match in link_matches:
        candidate = str(match or "").strip()
        if not candidate:
            continue
        if candidate.startswith("/"):
            candidate = urlunsplit((base.scheme, base.netloc, candidate, "", ""))
        resolved = _normalize_bangumi_uri(candidate)
        if resolved and _is_torrent_url(resolved):
            if _is_mikan_url(resolved) or resolved.startswith("http://") or resolved.startswith("https://"):
                return resolved
    return normalized


async def _load_torrent_bytes_from_event(
    event: MessageEvent,
    *,
    fallback_uri: str,
) -> Tuple[Optional[bytes], str, str]:
    message = event.get_message()
    file_meta = _extract_first_file_meta(message)
    name = ""
    direct_uri = _normalize_bangumi_uri(fallback_uri)
    if file_meta:
        name = _torrent_display_name(file_meta.get("name") or file_meta.get("file") or "种子文件")
        local_path = _maybe_local_path(file_meta.get("path") or file_meta.get("file") or "")
        if local_path and local_path.suffix.lower() == ".torrent":
            try:
                return local_path.read_bytes(), name or local_path.name, direct_uri
            except Exception:
                pass
        file_url = _normalize_bangumi_uri(file_meta.get("url") or file_meta.get("file") or "")
        if file_url and (_is_torrent_url(file_url) or _is_mikan_url(file_url)):
            direct_uri = file_url
    if not direct_uri:
        return None, name, ""
    if _is_mikan_url(direct_uri) and not _is_torrent_url(direct_uri):
        direct_uri = await _resolve_mikan_download_uri(direct_uri)
    if not _is_torrent_url(direct_uri):
        return None, name, direct_uri
    torrent_bytes = await _download_binary_with_limit(
        direct_uri,
        max_bytes=_bangumi_torrent_max_bytes(),
    )
    return torrent_bytes, name, direct_uri

def _mikan_search_url(keyword: str) -> str:
    encoded = quote(keyword, safe="")
    return f"{_mikan_base_url()}/RSS/Search?searchstr={encoded}"


def _parse_mikan_rss(xml_text: str) -> List[BangumiRelease]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    channel = root.find("channel")
    if channel is None:
        return []
    result: List[BangumiRelease] = []
    for item in channel.findall("item"):
        title = _xml_child_text(item, "title")
        if not title:
            continue
        page_url = _xml_child_text(item, "link")
        guid = _xml_child_text(item, "guid")
        download_url = _xml_enclosure_url(item)
        if not download_url:
            # 兼容个别 feed 场景下放在 namespaced torrent 节点。
            download_url = _xml_child_text(item, "torrent")
        if not download_url:
            continue
        release_date = _extract_release_date(download_url)
        subtitle_group = _extract_subtitle_group_from_title(title)
        episode = _extract_episode_from_title(title)
        bangumi_title = _extract_bangumi_title_from_release_title(title, subtitle_group)
        result.append(
            BangumiRelease(
                title=title,
                page_url=page_url,
                download_url=download_url,
                guid=guid or title,
                size_bytes=_xml_enclosure_length(item),
                release_date=release_date,
                subtitle_group=subtitle_group,
                bangumi_title=bangumi_title,
                episode=episode,
            )
        )
    return result


async def _classify_bangumi_meta_with_ai(releases: List[BangumiRelease]) -> None:
    if not releases:
        return
    if not config.google_api_key:
        return
    limit = min(len(releases), _bangumi_ai_classify_limit())
    if limit <= 0:
        return
    title_lines = [f"{idx + 1}. {releases[idx].title}" for idx in range(limit)]
    prompt = (
        "请解析下面番剧资源标题，返回 JSON 数组，每项格式："
        "{\"index\":序号,\"episode\":集数或null,\"subtitle_group\":\"字幕组\",\"bangumi_title\":\"番剧名\"}\n"
        "序号从 1 开始。\n"
        "只输出 JSON 数组。\n"
        + "\n".join(title_lines)
    )
    client = _get_client()
    config_obj, system_used = _build_generate_config(
        system_instruction=_BANGUMI_META_SYSTEM_PROMPT,
        response_mime_type="application/json",
    )
    if _BANGUMI_META_SYSTEM_PROMPT and not system_used:
        prompt = f"{_BANGUMI_META_SYSTEM_PROMPT}\n\n{prompt}"
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=config.gemini_text_model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config_obj,
            ),
            timeout=config.request_timeout,
        )
    except Exception:
        return
    payload = _extract_json_array(getattr(response, "text", "") or "")
    if not payload:
        return
    for row in payload:
        idx = _coerce_int(row.get("index"))
        if idx is None or idx <= 0 or idx > limit:
            continue
        item = releases[idx - 1]
        episode = _coerce_int(row.get("episode"))
        if episode is not None and 0 <= episode <= 999:
            item.episode = episode
        subtitle_group = row.get("subtitle_group")
        if isinstance(subtitle_group, str) and subtitle_group.strip():
            item.subtitle_group = subtitle_group.strip()
        bangumi_title = row.get("bangumi_title")
        if isinstance(bangumi_title, str) and bangumi_title.strip():
            item.bangumi_title = bangumi_title.strip()


async def _search_mikan_releases(keyword: str) -> List[BangumiRelease]:
    normalized = _normalize_bangumi_keyword(keyword)
    if not normalized:
        return []
    client = _get_http_client()
    url = _mikan_search_url(normalized)
    resp = await client.get(url, timeout=_request_timeout_seconds())
    resp.raise_for_status()
    releases = _parse_mikan_rss(resp.text)
    if not releases:
        return []
    # RSS 按时间倒序返回，截断可控上限。
    return releases[: _mikan_search_limit()]


def _release_group_matches(value: str, wanted: str) -> bool:
    group = _normalize_subtitle_group(value)
    target = _normalize_subtitle_group(wanted)
    if not target:
        return True
    if not group:
        return False
    return target in group or group in target


def _release_is_baha_source(release: BangumiRelease) -> bool:
    merged = " ".join(
        [
            str(release.title or ""),
            str(release.subtitle_group or ""),
            str(release.bangumi_title or ""),
            str(release.page_url or ""),
            str(release.download_url or ""),
        ]
    ).lower()
    for hint in _BANGUMI_BAHA_SOURCE_HINTS:
        if hint.lower() in merged:
            return True
    return False


def _release_source_priority(release: BangumiRelease) -> int:
    if _release_is_baha_source(release):
        return 1
    return 0


def _release_sort_key(release: BangumiRelease) -> Tuple[int, str, int]:
    episode = release.episode if release.episode is not None else -1
    date = release.release_date or ""
    return episode, date, _release_source_priority(release)


def _pick_best_release(candidates: List[BangumiRelease]) -> Optional[BangumiRelease]:
    if not candidates:
        return None
    sorted_candidates = sorted(candidates, key=_release_sort_key, reverse=True)
    return sorted_candidates[0]


async def _choose_latest_release_with_ai(
    *,
    keyword: str,
    releases: List[BangumiRelease],
) -> Optional[BangumiRelease]:
    if not releases:
        return None
    if not config.google_api_key:
        return None
    limit = min(len(releases), _bangumi_ai_classify_limit())
    if limit <= 0:
        return None
    lines: List[str] = []
    for idx in range(limit):
        item = releases[idx]
        episode_text = str(item.episode) if item.episode is not None else "null"
        source = "baha" if _release_is_baha_source(item) else "other"
        lines.append(
            f"{idx + 1}. source={source}; episode={episode_text}; title={item.title}"
        )
    prompt = (
        f"番剧关键词：{keyword}\n"
        "请在下面 RSS 条目里找出最新一集正片，并返回 JSON 对象："
        "{\"index\":序号或null,\"episode\":集数或null}\n"
        "如果同一集有多条，请优先 source=baha。\n"
        "只输出 JSON，不要解释。\n"
        + "\n".join(lines)
    )
    client = _get_client()
    config_obj, system_used = _build_generate_config(
        system_instruction=_BANGUMI_LATEST_SYSTEM_PROMPT,
        response_mime_type="application/json",
    )
    if _BANGUMI_LATEST_SYSTEM_PROMPT and not system_used:
        prompt = f"{_BANGUMI_LATEST_SYSTEM_PROMPT}\n\n{prompt}"
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=config.gemini_text_model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config_obj,
            ),
            timeout=config.request_timeout,
        )
    except Exception:
        return None
    payload = _extract_json(getattr(response, "text", "") or "")
    if not isinstance(payload, dict):
        return None
    index = _coerce_int(payload.get("index"))
    if index is not None and 1 <= index <= limit:
        chosen = releases[index - 1]
        if chosen.download_url:
            return chosen
    episode = _coerce_int(payload.get("episode"))
    if episode is not None:
        same_episode = [
            item for item in releases if item.download_url and item.episode == episode
        ]
        return _pick_best_release(same_episode)
    return None


async def _choose_bangumi_release(
    state: SessionState,
    *,
    keyword: str,
    releases: List[BangumiRelease],
    episode: Optional[int],
    latest: bool,
    subtitle_group: str,
) -> Tuple[Optional[BangumiRelease], str]:
    if not releases:
        return None, "没有找到可下载的资源。"

    candidates = [item for item in releases if item.download_url]
    if subtitle_group:
        candidates = [
            item
            for item in candidates
            if _release_group_matches(item.subtitle_group, subtitle_group)
        ]
        if not candidates:
            return None, f"没有找到字幕组“{subtitle_group}”的资源。"

    target_episode = episode
    if target_episode is not None:
        candidates = [item for item in candidates if item.episode == target_episode]
        if not candidates:
            return None, f"没有找到第{target_episode}集资源。"
    elif latest:
        ai_selected = await _choose_latest_release_with_ai(
            keyword=keyword,
            releases=candidates,
        )
        if ai_selected is not None:
            if ai_selected.episode is not None:
                target_episode = ai_selected.episode
                candidates = [item for item in candidates if item.episode == target_episode]
            else:
                candidates = [item for item in candidates if item.guid == ai_selected.guid]
                if not candidates:
                    candidates = [ai_selected]
        else:
            known_episodes = [item.episode for item in candidates if item.episode is not None]
            if known_episodes:
                target_episode = max(known_episodes)
                candidates = [item for item in candidates if item.episode == target_episode]

    if not candidates:
        return None, "没有找到匹配条件的资源。"

    selected = _pick_best_release(candidates)
    if selected is None:
        return None, "没有找到匹配条件的资源。"
    selected_episode = selected.episode
    if selected_episode is not None:
        known_group = _known_episode_group(state, keyword=keyword, episode=selected_episode)
        if known_group and not _release_group_matches(selected.subtitle_group, known_group):
            same_episode_candidates = [
                item for item in candidates if item.episode == selected_episode
            ]
            same_episode_candidates = sorted(
                same_episode_candidates,
                key=_release_sort_key,
                reverse=True,
            )
            for item in same_episode_candidates:
                if item.episode != selected_episode:
                    continue
                if _release_group_matches(item.subtitle_group, known_group):
                    selected = item
                    break
            else:
                return (
                    None,
                    f"第{selected_episode}集已下载过字幕组“{known_group}”，为避免同集不同字幕组重复，本次不下载。",
                )
    return selected, ""


def _aria2_secret_token() -> str:
    return str(getattr(config, "aria2_rpc_secret", "") or "").strip()


def _aria2_params(params: List[object]) -> List[object]:
    payload: List[object] = []
    token = _aria2_secret_token()
    if token:
        payload.append(f"token:{token}")
    payload.extend(params)
    return payload


async def _aria2_rpc_call(method: str, params: List[object]) -> dict:
    rpc_url = _aria2_rpc_url()
    if not rpc_url:
        raise RuntimeError("未配置 ARIA2_RPC_URL，或配置格式无效。")
    payload = {
        "jsonrpc": "2.0",
        "id": str(int(_now() * 1000)),
        "method": method,
        "params": _aria2_params(params),
    }
    client = _get_http_client()
    resp = await client.post(
        rpc_url,
        json=payload,
        timeout=_aria2_timeout_seconds(),
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("aria2 返回数据格式异常。")
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or "未知错误"
        raise RuntimeError(f"aria2 错误：{message}")
    return data


def _aria2_download_options() -> dict[str, str]:
    options: dict[str, str] = {"follow-torrent": "true"}
    trackers = _aria2_bt_trackers()
    if trackers:
        options["bt-tracker"] = ",".join(trackers)
    bt_max_peers = _aria2_bt_max_peers()
    if bt_max_peers > 0:
        options["bt-max-peers"] = str(bt_max_peers)
    if _aria2_disable_seed():
        options["seed-time"] = "0"
        options["seed-ratio"] = "0"
    download_dir = _aria2_download_dir()
    if download_dir:
        options["dir"] = download_dir
    return options


async def _aria2_add_uri(uri: str) -> str:
    uri_text = str(uri or "").strip()
    if not uri_text:
        raise RuntimeError("下载链接为空。")
    options = _aria2_download_options()
    params: List[object] = [[uri_text], options]
    try:
        data = await _aria2_rpc_call("aria2.addUri", params)
    except Exception as exc:
        infohash = _aria2_duplicate_infohash(str(exc))
        if infohash:
            existing = await _aria2_find_status_by_infohash(infohash)
            existing_gid = str(existing.get("gid") or "").strip() if existing else ""
            existing_state = str(existing.get("status") or "").strip().lower() if existing else ""
            if existing_gid and existing_state == "complete":
                deleted_count = await _aria2_delete_task_files(existing_gid)
                await _aria2_remove_download_result(existing_gid)
                logger.info(
                    "aria2 addUri duplicate complete gid removed (files deleted: {}), retry add: {}",
                    deleted_count,
                    existing_gid,
                )
                data = await _aria2_rpc_call("aria2.addUri", params)
                return _aria2_parse_result_gid(data)
            if existing_gid:
                logger.info("aria2 addUri duplicate infohash detected, reuse gid: {}", existing_gid)
                return existing_gid
        raise
    return _aria2_parse_result_gid(data)


async def _aria2_add_torrent_bytes(torrent_bytes: bytes) -> str:
    if not torrent_bytes:
        raise RuntimeError("种子内容为空。")
    payload = base64.b64encode(torrent_bytes).decode("ascii")
    options = _aria2_download_options()
    params: List[object] = [payload, [], options]
    try:
        data = await _aria2_rpc_call("aria2.addTorrent", params)
    except Exception as exc:
        infohash = _aria2_duplicate_infohash(str(exc))
        if infohash:
            existing = await _aria2_find_status_by_infohash(infohash)
            existing_gid = str(existing.get("gid") or "").strip() if existing else ""
            existing_state = str(existing.get("status") or "").strip().lower() if existing else ""
            if existing_gid and existing_state == "complete":
                deleted_count = await _aria2_delete_task_files(existing_gid)
                await _aria2_remove_download_result(existing_gid)
                logger.info(
                    "aria2 addTorrent duplicate complete gid removed (files deleted: {}), retry add: {}",
                    deleted_count,
                    existing_gid,
                )
                data = await _aria2_rpc_call("aria2.addTorrent", params)
                return _aria2_parse_result_gid(data)
            if existing_gid:
                logger.info("aria2 addTorrent duplicate infohash detected, reuse gid: {}", existing_gid)
                return existing_gid
        raise
    return _aria2_parse_result_gid(data)


def _aria2_parse_result_gid(data: dict) -> str:
    gid = data.get("result")
    if not isinstance(gid, str) or not gid.strip():
        raise RuntimeError("aria2 未返回任务 ID。")
    return gid.strip()


def _aria2_duplicate_infohash(message: str) -> str:
    text = str(message or "")
    if not text:
        return ""
    matched = _ARIA2_DUPLICATE_INFOHASH_RE.search(text)
    if not matched:
        return ""
    return str(matched.group(1) or "").strip().lower()


def _aria2_task_infohash(status: object) -> str:
    if not isinstance(status, dict):
        return ""
    value = status.get("infoHash")
    if isinstance(value, str) and len(value.strip()) == 40:
        return value.strip().lower()
    bittorrent = status.get("bittorrent")
    if isinstance(bittorrent, dict):
        value = bittorrent.get("infoHash")
        if isinstance(value, str) and len(value.strip()) == 40:
            return value.strip().lower()
        info = bittorrent.get("info")
        if isinstance(info, dict):
            value = info.get("infoHash")
            if isinstance(value, str) and len(value.strip()) == 40:
                return value.strip().lower()
    return ""


async def _aria2_list_status_rows(method: str, keys: List[str]) -> List[dict]:
    if method == "aria2.tellActive":
        data = await _aria2_rpc_call(method, [keys])
    else:
        data = await _aria2_rpc_call(method, [0, 1000, keys])
    result = data.get("result")
    if not isinstance(result, list):
        return []
    rows: List[dict] = []
    for item in result:
        if isinstance(item, dict):
            rows.append(item)
    return rows


async def _aria2_find_status_by_infohash(infohash: str) -> Optional[dict]:
    target = str(infohash or "").strip().lower()
    if len(target) != 40:
        return None
    keys = ["gid", "status", "infoHash", "bittorrent"]
    try:
        active, waiting, stopped = await asyncio.gather(
            _aria2_list_status_rows("aria2.tellActive", keys),
            _aria2_list_status_rows("aria2.tellWaiting", keys),
            _aria2_list_status_rows("aria2.tellStopped", keys),
        )
    except Exception:
        return None
    for row in [*active, *waiting, *stopped]:
        if _aria2_task_infohash(row) == target:
            return row
    return None


async def _aria2_find_gid_by_infohash(infohash: str) -> str:
    row = await _aria2_find_status_by_infohash(infohash)
    if not isinstance(row, dict):
        return ""
    gid = str(row.get("gid") or "").strip()
    return gid


async def _aria2_tell_status(gid: str) -> dict:
    gid_text = str(gid or "").strip()
    if not gid_text:
        raise RuntimeError("任务 ID 为空。")
    keys = [
        "gid",
        "status",
        "totalLength",
        "completedLength",
        "downloadSpeed",
        "errorMessage",
        "files",
        "followedBy",
    ]
    data = await _aria2_rpc_call("aria2.tellStatus", [gid_text, keys])
    result = data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("aria2 任务状态格式异常。")
    return result


async def _aria2_delete_task_files(gid: str) -> int:
    gid_text = str(gid or "").strip()
    if not gid_text:
        return 0
    try:
        status = await _aria2_tell_status(gid_text)
    except Exception:
        return 0
    paths = _status_file_paths(status)
    deleted_count = 0
    for path in paths:
        if _delete_local_file(path):
            deleted_count += 1
        if path.suffix.lower() == ".torrent":
            sidecar = Path(f"{path}.aria2")
            if _delete_local_file(sidecar):
                deleted_count += 1
    return deleted_count


async def _aria2_remove_download_result(gid: str) -> None:
    gid_text = str(gid or "").strip()
    if not gid_text:
        return
    try:
        await _aria2_rpc_call("aria2.removeDownloadResult", [gid_text])
    except Exception:
        return


def _event_notify_target(event: MessageEvent) -> BangumiNotifyTarget:
    if isinstance(event, GroupMessageEvent):
        return BangumiNotifyTarget(group_id=int(event.group_id))
    user_id = _coerce_int(event.get_user_id())
    return BangumiNotifyTarget(user_id=user_id)


def _notify_target_key(target: BangumiNotifyTarget) -> str:
    if target.group_id is not None:
        return f"group:{target.group_id}"
    if target.user_id is not None:
        return f"private:{target.user_id}"
    return "unknown"


async def _send_text_to_target(bot: Bot, target: BangumiNotifyTarget, text: str) -> None:
    normalized = _normalize_reply_text(text)
    if not normalized:
        return
    parts = _botbr_send_parts(normalized)
    if not parts:
        return
    delay = _message_send_delay_sec()
    for idx, part in enumerate(parts):
        if target.group_id is not None:
            await bot.call_api("send_group_msg", group_id=target.group_id, message=part)
        elif target.user_id is not None:
            await bot.call_api("send_private_msg", user_id=target.user_id, message=part)
        if delay > 0 and idx < len(parts) - 1:
            await asyncio.sleep(delay)


async def _send_bangumi_text_to_target(bot: Bot, target: BangumiNotifyTarget, text: str) -> None:
    if not _bangumi_send_text_reply():
        return
    await _send_text_to_target(bot, target, text)


async def _upload_file_to_target(
    bot: Bot,
    target: BangumiNotifyTarget,
    *,
    file_path: Path,
    display_name: str,
) -> None:
    absolute_path = str(file_path.resolve())
    if target.group_id is not None:
        await bot.call_api(
            "upload_group_file",
            group_id=target.group_id,
            file=absolute_path,
            name=display_name,
        )
        return
    if target.user_id is not None:
        await bot.call_api(
            "upload_private_file",
            user_id=target.user_id,
            file=absolute_path,
            name=display_name,
        )
        return
    raise RuntimeError("未识别到可发送的会话目标。")


def _status_file_paths(status: dict) -> List[Path]:
    rows = status.get("files")
    if not isinstance(rows, list):
        return []
    result: List[Path] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("path") or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(os.getcwd()) / path).resolve()
        key = str(path)
        if key in seen:
            continue
        if not path.exists() or not path.is_file():
            continue
        seen.add(key)
        result.append(path)
    return result


_BANGUMI_VIDEO_EXTS = {
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
    ".m2ts",
    ".webm",
}


def _is_bangumi_video_file(path: Path) -> bool:
    return path.suffix.lower() in _BANGUMI_VIDEO_EXTS


def _media_file_priority(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in _BANGUMI_VIDEO_EXTS:
        return 3
    if ext in {".ass", ".ssa", ".srt"}:
        return 2
    if ext in {".torrent", ".aria2"}:
        return 0
    return 1


def _pick_notify_file(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    candidates = [path for path in paths if _is_bangumi_video_file(path)]
    if not candidates:
        return None
    candidates = sorted(
        candidates,
        key=lambda item: (_media_file_priority(item), item.stat().st_size if item.exists() else 0),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _delete_local_file(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def _delete_torrent_files(paths: List[Path]) -> int:
    deleted_count = 0
    for path in paths:
        if path.suffix.lower() != ".torrent":
            continue
        if _delete_local_file(path):
            deleted_count += 1
        sidecar = Path(f"{path}.aria2")
        if _delete_local_file(sidecar):
            deleted_count += 1
    return deleted_count


async def _watch_bangumi_download_and_notify(
    bot: Bot,
    *,
    target: BangumiNotifyTarget,
    gid: str,
    keyword: str,
) -> None:
    interval = _bangumi_progress_notify_interval_sec()
    timeout = _bangumi_download_watch_timeout_sec()
    title = _normalize_bangumi_keyword(keyword) or "番剧任务"
    start_ts = _now()
    current_gid = str(gid).strip()
    if not current_gid:
        return
    visited: set[str] = set()
    while True:
        if _now() - start_ts > timeout:
            await _send_bangumi_text_to_target(
                bot,
                target,
                f"{title} 下载监控超时，任务ID {current_gid}，请手动查看 aria2 状态。",
            )
            return
        try:
            status = await _aria2_tell_status(current_gid)
        except Exception as exc:
            await _send_bangumi_text_to_target(
                bot,
                target,
                f"{title} 状态查询失败：{_safe_error_message(exc)}",
            )
            return
        followed = status.get("followedBy")
        if isinstance(followed, list) and followed:
            next_gid = str(followed[0] or "").strip()
            if next_gid and next_gid != current_gid and next_gid not in visited:
                visited.add(current_gid)
                current_gid = next_gid
                continue

        state = str(status.get("status") or "").strip().lower()

        if state in {"active", "waiting", "paused"}:
            await asyncio.sleep(interval)
            continue

        if state == "complete":
            paths = _status_file_paths(status)
            send_file = _pick_notify_file(paths)
            if not send_file:
                _delete_torrent_files(paths)
                await _send_bangumi_text_to_target(
                    bot,
                    target,
                    f"{title} 下载完成，但没有找到可发送文件，任务ID {current_gid}。",
                )
                await _aria2_remove_download_result(current_gid)
                return
            if not _bangumi_auto_send_file():
                _delete_torrent_files(paths)
                await _send_bangumi_text_to_target(
                    bot,
                    target,
                    f"{title} 下载完成：{send_file.name}",
                )
                await _aria2_remove_download_result(current_gid)
                return
            await _send_bangumi_text_to_target(
                bot,
                target,
                f"{title} 下载完成，正在发送文件：{send_file.name}",
            )
            try:
                await _upload_file_to_target(
                    bot,
                    target,
                    file_path=send_file,
                    display_name=send_file.name,
                )
            except Exception as exc:
                _delete_torrent_files(paths)
                await _send_bangumi_text_to_target(
                    bot,
                    target,
                    f"{title} 文件发送失败：{_safe_error_message(exc)}",
                )
                await _aria2_remove_download_result(current_gid)
                return
            await _send_bangumi_text_to_target(
                bot,
                target,
                f"{title} 已发送完成：{send_file.name}",
            )
            deleted_count = 0
            if _bangumi_auto_delete_after_send():
                if _delete_local_file(send_file):
                    deleted_count += 1
                sidecar = Path(f"{send_file}.aria2")
                if _delete_local_file(sidecar):
                    deleted_count += 1
            deleted_count += _delete_torrent_files(paths)
            if deleted_count > 0:
                await _send_bangumi_text_to_target(
                    bot,
                    target,
                    f"{title} 已清理本地文件 {deleted_count} 个。",
                )
            await _aria2_remove_download_result(current_gid)
            return

        if state in {"error", "removed"}:
            err = _collapse_spaces(str(status.get("errorMessage") or "").strip())
            if state == "error" and err:
                await _send_bangumi_text_to_target(
                    bot,
                    target,
                    f"{title} 下载失败：{err}",
                )
            else:
                await _send_bangumi_text_to_target(
                    bot,
                    target,
                    f"{title} 任务已结束，状态：{state or 'unknown'}。",
                )
            await _aria2_remove_download_result(current_gid)
            return

        await asyncio.sleep(interval)


def _start_bangumi_download_watch(
    bot: Bot,
    event: MessageEvent,
    *,
    gid: str,
    keyword: str,
) -> None:
    target = _event_notify_target(event)
    target_key = _notify_target_key(target)
    task_key = f"{target_key}:{gid}"
    existing = _BANGUMI_DOWNLOAD_TASKS.get(task_key)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _watch_bangumi_download_and_notify(
            bot,
            target=target,
            gid=gid,
            keyword=keyword,
        )
    )
    _BANGUMI_DOWNLOAD_TASKS[task_key] = task

    def _cleanup(done_task: asyncio.Task[None]) -> None:
        _BANGUMI_DOWNLOAD_TASKS.pop(task_key, None)
        try:
            done_task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Bangumi download watch failed: {}", _safe_error_message(exc))

    task.add_done_callback(_cleanup)


def _subscription_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"download", "subscribe", "unsubscribe", "list", "check"}:
        return mode
    return "download"


def _subscription_key(session_id: str, keyword: str, subtitle_group: str) -> str:
    return "|".join(
        [
            session_id.strip(),
            _normalize_bangumi_keyword(keyword).lower(),
            _normalize_subtitle_group(subtitle_group),
        ]
    )


def _load_subscription_data() -> dict:
    path = _bangumi_subscription_file()
    if not path.exists():
        return {"subscriptions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"subscriptions": []}
    if not isinstance(data, dict):
        return {"subscriptions": []}
    subs = data.get("subscriptions")
    if not isinstance(subs, list):
        data["subscriptions"] = []
    return data


def _save_subscription_data(data: dict) -> None:
    path = _bangumi_subscription_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


async def _list_subscriptions(session_id: str) -> List[dict]:
    async with _bangumi_subs_lock():
        data = _load_subscription_data()
        rows = data.get("subscriptions")
        if not isinstance(rows, list):
            return []
        result: List[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("session_id") or "") != session_id:
                continue
            result.append(row)
        return result


async def _upsert_subscription(
    session_id: str,
    *,
    keyword: str,
    subtitle_group: str,
) -> dict:
    normalized_keyword = _normalize_bangumi_keyword(keyword)
    normalized_group = str(subtitle_group or "").strip()
    if not normalized_keyword:
        raise RuntimeError("订阅关键词为空。")
    key = _subscription_key(session_id, normalized_keyword, normalized_group)
    now_ts = _now()
    async with _bangumi_subs_lock():
        data = _load_subscription_data()
        rows = data.get("subscriptions")
        if not isinstance(rows, list):
            rows = []
            data["subscriptions"] = rows
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("key") or "") != key:
                continue
            row["keyword"] = normalized_keyword
            row["subtitle_group"] = normalized_group
            row["updated_at"] = now_ts
            _save_subscription_data(data)
            return row
        row = {
            "key": key,
            "session_id": session_id,
            "keyword": normalized_keyword,
            "subtitle_group": normalized_group,
            "last_guid": "",
            "last_episode": None,
            "created_at": now_ts,
            "updated_at": now_ts,
        }
        rows.append(row)
        _save_subscription_data(data)
        return row


async def _remove_subscription(
    session_id: str,
    *,
    keyword: str,
    subtitle_group: str,
) -> bool:
    normalized_keyword = _normalize_bangumi_keyword(keyword)
    if not normalized_keyword:
        return False
    normalized_group = _normalize_subtitle_group(subtitle_group)
    target_key = _subscription_key(session_id, normalized_keyword, subtitle_group)
    async with _bangumi_subs_lock():
        data = _load_subscription_data()
        rows = data.get("subscriptions")
        if not isinstance(rows, list):
            return False
        kept: List[dict] = []
        removed = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            if normalized_group:
                if str(row.get("key") or "") == target_key:
                    removed = True
                    continue
            else:
                if str(row.get("session_id") or "") == session_id and _normalize_bangumi_keyword(
                    str(row.get("keyword") or "")
                ).lower() == normalized_keyword.lower():
                    removed = True
                    continue
            kept.append(row)
        if removed:
            data["subscriptions"] = kept
            _save_subscription_data(data)
        return removed


async def _mark_subscription_progress(
    session_id: str,
    *,
    keyword: str,
    subtitle_group: str,
    release: BangumiRelease,
) -> None:
    target_key = _subscription_key(session_id, keyword, subtitle_group)
    async with _bangumi_subs_lock():
        data = _load_subscription_data()
        rows = data.get("subscriptions")
        if not isinstance(rows, list):
            return
        changed = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("key") or "") != target_key:
                continue
            row["last_guid"] = release.guid
            row["last_episode"] = release.episode
            row["updated_at"] = _now()
            changed = True
            break
        if changed:
            _save_subscription_data(data)


def _format_release_size(size_bytes: Optional[int]) -> str:
    if size_bytes is None or size_bytes <= 0:
        return "大小未知"
    mib = float(size_bytes) / (1024.0 * 1024.0)
    if mib >= 1024:
        gib = mib / 1024.0
        return f"{gib:.2f}GB"
    return f"{mib:.1f}MB"


def _infer_subscription_mode(raw_text: str) -> str:
    text = str(raw_text or "").lower()
    for token in _BANGUMI_SUB_CHECK_HINTS:
        if token.lower() in text:
            return "check"
    for token in _BANGUMI_SUB_LIST_HINTS:
        if token.lower() in text:
            return "list"
    for token in _BANGUMI_SUB_REMOVE_HINTS:
        if token.lower() in text:
            return "unsubscribe"
    for token in _BANGUMI_SUB_ADD_HINTS:
        if token.lower() in text:
            return "subscribe"
    return "download"


def _extract_episode_from_text(raw_text: str) -> Optional[int]:
    text = str(raw_text or "")
    if not text:
        return None
    for pattern in _MikanEpisodePatterns:
        match = pattern.search(text)
        if not match:
            continue
        value = _coerce_int(match.group(1))
        if value is not None and 0 <= value <= 999:
            return value
    return None


def _extract_subtitle_group_from_text(raw_text: str) -> str:
    text = str(raw_text or "")
    if not text:
        return ""
    match = re.search(r"(?:字幕组|字幕社)[:：\s]*([A-Za-z0-9\u4e00-\u9fff]{1,24})", text, re.I)
    if match:
        return match.group(1).strip() + "字幕组"
    match = _BANGUMI_SUB_GROUP_HINT_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def _strip_bangumi_query_noise(raw_text: str) -> str:
    text = str(raw_text or "")
    for word in _BANGUMI_COMMAND_WORDS:
        text = text.replace(word, " ")
    text = re.sub(r"(?:第\s*[0-9]{1,3}\s*(?:集|话|話))", " ", text, flags=re.I)
    text = re.sub(r"\b(?:ep|e)\s*[0-9]{1,3}\b", " ", text, flags=re.I)
    for token in (
        "最新一集",
        "最新集",
        "最新",
        "订阅下载",
        "取消订阅",
        "查看订阅",
        "订阅列表",
        "检查订阅",
        "更新订阅",
        "订阅",
        "字幕组",
        "下载",
    ):
        text = text.replace(token, " ")
    text = re.sub(r"[，,。.!！？?？:：;；\[\]【】()（）]", " ", text)
    return _normalize_bangumi_keyword(text)


def _build_bangumi_request(intent: dict, raw_text: str) -> dict[str, object]:
    params = _intent_params(intent)
    mode = _subscription_mode(str(params.get("mode") or ""))
    if mode == "download":
        mode = _infer_subscription_mode(raw_text)

    raw_uri = params.get("uri") or params.get("download_url") or params.get("url")
    uri = raw_uri.strip() if isinstance(raw_uri, str) else ""
    if not uri:
        uri = _extract_bangumi_uri_from_text(raw_text)
    uri = _normalize_bangumi_uri(uri) if uri else ""
    if uri and not _is_bangumi_direct_uri(uri):
        uri = ""

    raw_keyword = params.get("keyword") or params.get("title")
    keyword = raw_keyword.strip() if isinstance(raw_keyword, str) else ""
    if not keyword and not uri:
        keyword = _strip_bangumi_query_noise(raw_text)

    episode = _coerce_int(params.get("episode"))
    if episode is None:
        episode = _extract_episode_from_text(raw_text)

    subtitle_group_raw = params.get("subtitle_group") or params.get("subgroup") or ""
    subtitle_group = subtitle_group_raw.strip() if isinstance(subtitle_group_raw, str) else ""
    if not subtitle_group:
        subtitle_group = _extract_subtitle_group_from_text(raw_text)

    latest_raw = _coerce_bool(params.get("latest"))
    if latest_raw is None:
        latest_raw = any(token in str(raw_text or "").lower() for token in _BANGUMI_LATEST_HINTS)
    latest = bool(latest_raw)
    if episode is not None:
        latest = False
    if episode is None and not latest and not uri:
        latest = True

    subscribe_raw = _coerce_bool(params.get("subscribe"))
    subscribe = bool(subscribe_raw) if subscribe_raw is not None else mode == "subscribe"
    if mode == "unsubscribe":
        subscribe = False

    return {
        "mode": mode,
        "keyword": _normalize_bangumi_keyword(keyword),
        "uri": uri,
        "episode": episode,
        "latest": latest,
        "subtitle_group": subtitle_group.strip(),
        "subscribe": subscribe,
    }


def _format_subscription_list_lines(rows: List[dict]) -> str:
    if not rows:
        return "当前没有订阅。"
    lines = ["当前订阅："]
    for idx, row in enumerate(rows, 1):
        keyword = str(row.get("keyword") or "").strip() or "未命名"
        group = str(row.get("subtitle_group") or "").strip()
        suffix = f" / 字幕组:{group}" if group else ""
        lines.append(f"{idx}. {keyword}{suffix}")
    return "\n".join(lines)


async def _check_subscriptions_and_download(
    state: SessionState,
    *,
    session_id: str,
    bot: Bot,
    event: MessageEvent,
) -> str:
    rows = await _list_subscriptions(session_id)
    if not rows:
        return "当前没有订阅。"
    success_lines: List[str] = []
    detail_lines: List[str] = []
    downloaded = 0
    skipped = 0
    for row in rows:
        keyword = _normalize_bangumi_keyword(str(row.get("keyword") or ""))
        subgroup = str(row.get("subtitle_group") or "").strip()
        if not keyword:
            skipped += 1
            continue
        try:
            releases = await _search_mikan_releases(keyword)
            if not releases:
                detail_lines.append(f"{keyword}：未找到资源")
                skipped += 1
                continue
            await _classify_bangumi_meta_with_ai(releases)
            selected, reason = await _choose_bangumi_release(
                state,
                keyword=keyword,
                releases=releases,
                episode=None,
                latest=True,
                subtitle_group=subgroup,
            )
            if not selected:
                detail_lines.append(f"{keyword}：{reason or '未命中'}")
                skipped += 1
                continue
            last_guid = str(row.get("last_guid") or "")
            last_episode = _coerce_int(row.get("last_episode"))
            if last_guid and selected.guid == last_guid:
                skipped += 1
                continue
            if last_episode is not None and selected.episode is not None and last_episode == selected.episode:
                skipped += 1
                continue
            gid = await _aria2_add_uri(selected.download_url)
            _start_bangumi_download_watch(
                bot,
                event,
                gid=gid,
                keyword=selected.bangumi_title or keyword,
            )
            _remember_episode_group(
                state,
                keyword=keyword,
                episode=selected.episode,
                subtitle_group=selected.subtitle_group,
            )
            await _mark_subscription_progress(
                session_id,
                keyword=keyword,
                subtitle_group=subgroup,
                release=selected,
            )
            downloaded += 1
            episode_text = f"第{selected.episode}集" if selected.episode is not None else "最新资源"
            group_text = selected.subtitle_group or "未知字幕组"
            success_lines.append(f"{keyword}：{episode_text} {group_text} 已提交 aria2，任务ID {gid}")
        except Exception as exc:
            skipped += 1
            detail_lines.append(f"{keyword}：检查失败 {_safe_error_message(exc)}")
    if not success_lines and not detail_lines:
        return "订阅检查完成，没有新资源。"
    head = f"订阅检查完成：新增{downloaded}，跳过{skipped}"
    lines = [head, *success_lines]
    if detail_lines:
        remaining = max(0, 10 - len(success_lines))
        if remaining > 0:
            lines.extend(detail_lines[:remaining])
            rest = len(detail_lines) - remaining
            if rest > 0:
                lines.append(f"其余{rest}项结果已省略。")
        else:
            lines.append(f"其余{len(detail_lines)}项结果已省略。")
    return "\n".join(lines)


async def _build_bangumi_download_reply(
    bot: Bot,
    intent: dict,
    state: SessionState,
    event: MessageEvent,
    *,
    raw_text: str,
) -> str:
    session_id = _session_id(event)
    request = _build_bangumi_request(intent, raw_text)
    mode = str(request.get("mode") or "download")
    keyword = str(request.get("keyword") or "").strip()
    subtitle_group = str(request.get("subtitle_group") or "").strip()
    episode = _coerce_int(request.get("episode"))
    latest = bool(request.get("latest"))
    subscribe = bool(request.get("subscribe"))
    uri = str(request.get("uri") or "").strip()
    if not uri:
        uri = _extract_bangumi_uri_from_text(raw_text)
    uri = _normalize_bangumi_uri(uri) if uri else ""
    file_meta = _extract_first_file_meta(event.get_message())
    file_name_hint = ""
    file_url_hint = ""
    local_path_hint = ""
    has_torrent_file_hint = False
    if file_meta:
        file_name_hint = str(file_meta.get("name") or file_meta.get("file") or "").strip()
        file_url_hint = _normalize_bangumi_uri(file_meta.get("url") or file_meta.get("file") or "")
        local_path_hint = str(file_meta.get("path") or file_meta.get("file") or "").strip()
        if _looks_like_torrent_file_name(file_name_hint):
            has_torrent_file_hint = True
        elif file_url_hint and _is_torrent_url(file_url_hint):
            has_torrent_file_hint = True
        elif _looks_like_torrent_file_name(local_path_hint):
            has_torrent_file_hint = True

    if mode == "list":
        rows = await _list_subscriptions(session_id)
        return _format_subscription_list_lines(rows)

    if mode == "check":
        if not _bangumi_subscription_enabled():
            return "订阅功能已关闭，开启 BANGUMI_SUBSCRIPTION_ENABLE 后可使用。"
        return await _check_subscriptions_and_download(
            state,
            session_id=session_id,
            bot=bot,
            event=event,
        )

    if mode == "unsubscribe":
        if not keyword:
            return "请提供要取消订阅的番剧名，例如：取消订阅 葬送的芙莉莲"
        removed = await _remove_subscription(
            session_id,
            keyword=keyword,
            subtitle_group=subtitle_group,
        )
        if removed:
            return f"已取消订阅：{keyword}"
        return f"未找到订阅：{keyword}"

    if uri or has_torrent_file_hint:
        torrent_bytes: Optional[bytes] = None
        torrent_name = ""
        resolved_uri = uri
        try:
            torrent_bytes, torrent_name, resolved_uri = await _load_torrent_bytes_from_event(
                event,
                fallback_uri=uri,
            )
        except Exception as exc:
            return f"下载链接处理失败：{_safe_error_message(exc)}"
        if torrent_bytes is None and not resolved_uri and has_torrent_file_hint:
            return "种子文件读取失败，无法获取可下载内容。"
        if torrent_bytes is None and resolved_uri and _is_mikan_url(resolved_uri) and not _is_torrent_url(resolved_uri):
            return "蜜柑链接解析失败，未找到可下载的种子地址。"
        if torrent_bytes is not None:
            ok, reason = _check_torrent_compliance(
                torrent_bytes,
                display_name=torrent_name,
            )
            if not ok:
                return reason
        try:
            if torrent_bytes is not None:
                gid = await _aria2_add_torrent_bytes(torrent_bytes)
            else:
                if not resolved_uri:
                    return "没有获取到可下载链接。"
                gid = await _aria2_add_uri(resolved_uri)
        except Exception as exc:
            return f"提交下载失败：{_safe_error_message(exc)}"
        title = keyword or _torrent_display_name(torrent_name) or "番剧资源"
        _start_bangumi_download_watch(
            bot,
            event,
            gid=gid,
            keyword=title,
        )
        lines = [f"已提交到 aria2 并开始下载，任务ID：{gid}", f"番剧：{title}"]
        if resolved_uri:
            lines.append(f"来源：{resolved_uri}")
        if torrent_bytes is not None:
            lines.append("已通过种子文件违规校验。")
        else:
            lines.append("非种子文件来源，跳过违规预检。")
        if _bangumi_auto_send_file():
            if _bangumi_auto_delete_after_send():
                lines.append("下载完成后会发送到QQ并删除本地文件。")
            else:
                lines.append("下载完成后会自动发送到QQ并删除种子文件。")

        if mode == "subscribe" or subscribe:
            if not keyword:
                lines.append("未提供番剧名，无法写入订阅。")
            elif not _bangumi_subscription_enabled():
                lines.append("订阅功能已关闭，未写入订阅。")
            else:
                await _upsert_subscription(
                    session_id,
                    keyword=keyword,
                    subtitle_group=subtitle_group,
                )
                lines.append("已加入订阅。后续可用“订阅下载 检查订阅”自动补新。")
        return "\n".join(lines)

    if not keyword:
        if file_meta:
            return "收到文件消息，但未识别到可下载的种子链接。请直接发送 .torrent 文件或蜜柑链接。"
        return "请告诉我番剧名，或直接发送蜜柑链接/.torrent 文件。"

    releases = await _search_mikan_releases(keyword)
    if not releases:
        return f"蜜柑没有找到“{keyword}”相关资源。"
    await _classify_bangumi_meta_with_ai(releases)
    selected, reason = await _choose_bangumi_release(
        state,
        keyword=keyword,
        releases=releases,
        episode=episode,
        latest=latest,
        subtitle_group=subtitle_group,
    )
    if not selected:
        return reason or "没有找到匹配资源。"

    try:
        gid = await _aria2_add_uri(selected.download_url)
    except Exception as exc:
        return f"提交下载失败：{_safe_error_message(exc)}"
    _start_bangumi_download_watch(
        bot,
        event,
        gid=gid,
        keyword=selected.bangumi_title or keyword,
    )
    _remember_episode_group(
        state,
        keyword=keyword,
        episode=selected.episode,
        subtitle_group=selected.subtitle_group,
    )

    lines = []
    lines.append(f"已提交到 aria2 并开始下载，任务ID：{gid}")
    lines.append(f"番剧：{selected.bangumi_title or keyword}")
    if selected.episode is not None:
        lines.append(f"集数：第{selected.episode}集")
    else:
        lines.append("集数：未识别")
    lines.append(f"字幕组：{selected.subtitle_group or '未识别'}")
    lines.append(f"体积：{_format_release_size(selected.size_bytes)}")
    if selected.page_url:
        lines.append(f"页面：{selected.page_url}")
    if _bangumi_auto_send_file():
        if _bangumi_auto_delete_after_send():
            lines.append("下载完成后会发送到QQ并删除本地文件。")
        else:
            lines.append("下载完成后会自动发送到QQ并删除种子文件。")

    if mode == "subscribe" or subscribe:
        if not _bangumi_subscription_enabled():
            lines.append("订阅功能已关闭，未写入订阅。")
        else:
            await _upsert_subscription(
                session_id,
                keyword=keyword,
                subtitle_group=subtitle_group,
            )
            await _mark_subscription_progress(
                session_id,
                keyword=keyword,
                subtitle_group=subtitle_group,
                release=selected,
            )
            lines.append("已加入订阅。后续可用“订阅下载 检查订阅”自动补新。")
    return "\n".join(lines)


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
    temperature: Optional[float] = None,
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
    allow_temperature = temperature is not None and (
        fields is None or "temperature" in fields
    )
    if not allow_system and not allow_mime and not allow_modalities and not allow_temperature:
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
    if allow_temperature and temperature is not None:
        try:
            config_obj["temperature"] = float(temperature)
        except Exception:
            pass
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


def _extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        text_value = _extract_text_value(part)
        if text_value:
            text_parts.append(text_value)
    return "\n".join(text_parts).strip()


def _finalize_chat_reply(text: str) -> str:
    cleaned = _format_reply_text(text)
    cleaned = _compact_reply_lines(cleaned)
    cleaned = _limit_reply_text(cleaned)
    return cleaned


def _build_chat_single_prompt(chat_prompt: str) -> str:
    return "\n".join(
        [
            "请直接输出自然语言回复。",
            "不要输出 JSON。",
            "输入中包含当前待回复消息和参考对话，只回答当前消息。",
            "输入开始：",
            chat_prompt,
            "输入结束。",
        ]
    )


def _chat_pipeline_clarify_text() -> str:
    return "我想先确认一下你的重点问题，可以再具体一点吗？"


async def _call_gemini_text(prompt: str, state: SessionState) -> str:
    try:
        client = _get_client()
        chat_prompt = _build_chat_single_prompt(prompt)
        if _history_reference_only():
            chat_prompt = _wrap_prompt_with_reference(state, chat_prompt, current_label="当前消息")
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=chat_prompt)],
                )
            ]
        else:
            contents = _history_to_gemini(state)
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=chat_prompt)],
                )
            )
        config_obj, system_used = _build_generate_config(
            system_instruction=_CHAT_SINGLE_SYSTEM_PROMPT,
            temperature=_chat_knowledge_temperature(),
        )
        if _CHAT_SINGLE_SYSTEM_PROMPT and not system_used:
            merged_prompt = f"{_CHAT_SINGLE_SYSTEM_PROMPT}\n\n{chat_prompt}"
            if _history_reference_only():
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=merged_prompt)],
                    )
                ]
            else:
                contents = _history_to_gemini(state)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=merged_prompt)],
                    )
                )
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=config.gemini_text_model,
                contents=contents,
                config=config_obj,
            ),
            timeout=config.request_timeout,
        )
    except Exception as exc:
        logger.warning("Chat single stage failed: {}", _safe_error_message(exc))
        return _chat_pipeline_clarify_text()

    if config.gemini_log_response:
        logger.info("Gemini chat single response: {}", _dump_response(response))
        _log_response_text("Gemini chat single content", response)
    reply = _finalize_chat_reply(_extract_response_text(response))
    if _is_skip_reply_text(reply):
        return _chat_pipeline_clarify_text()
    return reply


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


def _extract_html_title(html_text: str) -> str:
    if not html_text:
        return ""
    match = _HTML_TITLE_RE.search(html_text)
    if not match:
        return ""
    title = html.unescape(match.group(1))
    title = _collapse_spaces(title)
    return title


def _extract_web_main_html(html_text: str, *, host: str = "") -> str:
    if not html_text:
        return ""
    root_host = _web_summary_root_host(host)
    site_patterns = _WEB_SITE_BLOCK_PATTERNS.get(root_host, ())
    for pattern in site_patterns:
        match = pattern.search(html_text)
        if match and match.group(1).strip():
            return match.group(1)
    for pattern in _WEB_COMMON_BLOCK_PATTERNS:
        match = pattern.search(html_text)
        if match and match.group(1).strip():
            return match.group(1)
    return html_text


def _html_to_plain_text(html_text: str) -> str:
    if not html_text:
        return ""
    cleaned = _HTML_COMMENT_RE.sub(" ", html_text)
    cleaned = _HTML_DROP_BLOCK_RE.sub(" ", cleaned)
    cleaned = _HTML_BLOCK_TAG_RE.sub("\n", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned).replace("\xa0", " ")
    lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = _collapse_spaces(raw_line)
        if line:
            lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_web_page_text(html_text: str, *, host: str = "") -> str:
    if not html_text:
        return ""
    title = _extract_html_title(html_text)
    primary_html = _extract_web_main_html(html_text, host=host)
    primary_text = _html_to_plain_text(primary_html)
    if not primary_text:
        primary_text = _html_to_plain_text(html_text)
    if not primary_text:
        return ""

    if title:
        if title not in primary_text[:120]:
            primary_text = f"页面标题: {title}\n{primary_text}"

    limit = _web_extract_max_chars()
    if len(primary_text) > limit:
        primary_text = primary_text[:limit].rstrip() + "\n[内容已截断]"
    return primary_text


async def _fetch_web_page_text(url: str) -> str:
    client = _get_http_client()
    max_bytes = _web_fetch_max_bytes()
    request_url = _rewrite_github_blob_url_for_fetch(url)
    headers = {
        "User-Agent": "nonebot-plugin-skills/1.0",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
    }
    chunks: List[bytes] = []
    total = 0
    encoding = "utf-8"
    content_type = ""
    source_host = ""
    async with client.stream(
        "GET",
        request_url,
        headers=headers,
        timeout=_request_timeout_seconds(),
    ) as resp:
        resp.raise_for_status()
        final_host = str(getattr(resp.url, "host", "") or "").strip().lower()
        source_host = final_host
        if final_host and not _is_allowed_web_summary_host(final_host):
            raise RuntimeError("重定向后不是受支持站点，已拒绝访问")
        content_type = str(resp.headers.get("content-type") or "").lower()
        if content_type and "text/html" not in content_type and "text/plain" not in content_type:
            raise RuntimeError(f"链接不是网页文本内容（{content_type}）")
        encoding = resp.encoding or "utf-8"
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                limit_mb = max(1, max_bytes // (1024 * 1024))
                raise RuntimeError(f"网页内容过大，请限制在 {limit_mb}MB 内")
            chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise RuntimeError("网页内容为空")
    try:
        decoded_text = data.decode(encoding, errors="ignore")
    except LookupError:
        decoded_text = data.decode("utf-8", errors="ignore")
    if "text/plain" in content_type:
        text = _extract_plain_page_text(decoded_text)
    else:
        text = _extract_web_page_text(decoded_text, host=source_host)
    if not text:
        raise RuntimeError("未提取到可读正文")
    return text


def _build_web_summary_prompt(url: str, page_text: str, *, focus: str = "") -> str:
    source_label = _web_summary_source_label(url)
    parts: List[str] = [
        f"请总结下面{source_label}网页内容。",
        f"链接: {url}",
    ]
    if focus:
        parts.append(f"用户关注点: {focus}")
    parts.append("网页正文提取结果:")
    parts.append(page_text)
    parts.append("输出要求：纯文本，先给重点结论，再给关键细节。")
    return "\n".join(parts)


async def _call_gemini_web_summary(
    url: str,
    page_text: str,
    state: SessionState,
    *,
    focus: str = "",
) -> str:
    client = _get_client()
    prompt = _build_web_summary_prompt(url, page_text, focus=focus)
    config_obj, system_used = _build_generate_config(system_instruction=_WEB_SUMMARY_SYSTEM_PROMPT)
    if _WEB_SUMMARY_SYSTEM_PROMPT and not system_used:
        prompt = f"{_WEB_SUMMARY_SYSTEM_PROMPT}\n\n{prompt}"
    if _history_reference_only():
        prompt = _wrap_prompt_with_reference(state, prompt, current_label="当前任务")
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
    else:
        contents = _history_to_gemini(state)
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
        logger.info("Gemini web summary response: {}", _dump_response(response))
        _log_response_text("Gemini web summary content", response)
    if response.text:
        cleaned = _format_reply_text(response.text.strip())
        cleaned = _compact_reply_lines(cleaned)
        cleaned = _limit_reply_text(cleaned)
        return cleaned
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        text_value = _extract_text_value(part)
        if text_value:
            text_parts.append(text_value)
    cleaned = _format_reply_text("\n".join(text_parts).strip())
    cleaned = _compact_reply_lines(cleaned)
    cleaned = _limit_reply_text(cleaned)
    return cleaned


def _extract_web_summary_focus(raw_text: str, url: str) -> str:
    if not raw_text:
        return ""
    focus = str(raw_text)
    if url:
        variants = {
            url,
            url.replace("https://", "http://"),
            url.replace("http://", "https://"),
        }
        parsed = urlsplit(url)
        host = (parsed.netloc or "").strip()
        path = parsed.path or ""
        query = parsed.query or ""
        if host:
            variants.add(host)
            if path:
                plain = f"{host}{path}"
                variants.add(plain)
                variants.add(plain.lstrip("/"))
                if query:
                    variants.add(f"{plain}?{query}")
            elif query:
                variants.add(f"{host}?{query}")
        for value in variants:
            if value:
                focus = focus.replace(value, " ")
    focus = _collapse_spaces(focus)
    if len(focus) > 200:
        focus = focus[:200].rstrip() + "..."
    return focus


async def _build_web_summary_reply(
    intent: dict,
    state: SessionState,
    event: MessageEvent,
    *,
    raw_text: str,
) -> Optional[str]:
    normalized_url = _resolve_web_summary_url(intent, raw_text)
    if not normalized_url:
        return None
    params = _intent_params(intent)
    focus_raw = params.get("focus")
    focus = focus_raw.strip() if isinstance(focus_raw, str) else ""
    if not focus:
        focus = _extract_web_summary_focus(raw_text, normalized_url)
    page_text = await _fetch_web_page_text(normalized_url)
    reply = await _call_gemini_web_summary(normalized_url, page_text, state, focus=focus)
    if not reply:
        raise RuntimeError("总结结果为空，请稍后再试")
    user_name = _event_user_name(event)
    summary = normalized_url if not focus else f"{normalized_url} 关注:{focus}"
    await _append_history(
        state,
        "user",
        f"网页总结：{summary}",
        user_id=str(event.get_user_id()),
        user_name=user_name,
        to_bot=True,
    )
    await _append_history(state, "model", reply)
    return reply


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
    client = _get_http_client()
    max_bytes = _image_download_max_bytes()
    max_mb = max(1, max_bytes // (1024 * 1024))
    chunks: List[bytes] = []
    total = 0
    content_type = "image/jpeg"
    async with client.stream("GET", url, timeout=_request_timeout_seconds()) as resp:
        resp.raise_for_status()
        content_type = _normalize_content_type(resp.headers.get("content-type"), default="image/jpeg")
        if content_type in {"application/octet-stream", "binary/octet-stream"}:
            content_type = "image/jpeg"
        if not content_type.startswith("image/"):
            raise UnsupportedImageError("仅支持下载图片内容。")
        content_length = _coerce_int(resp.headers.get("content-length"))
        if content_length is not None and content_length > max_bytes:
            raise UnsupportedImageError(f"图片过大，请限制在 {max_mb}MB 内。")
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise UnsupportedImageError(f"图片过大，请限制在 {max_mb}MB 内。")
            chunks.append(chunk)
    data = b"".join(chunks)
    return _validate_image_bytes(content_type, data)


async def _download_image_bytes_ref(bot: Bot, ref: str) -> Tuple[str, bytes]:
    ref = str(ref or "").strip()
    if not ref:
        raise RuntimeError("图片引用为空")
    if ref.lower().startswith("data:image"):
        decoded = _decode_data_url(ref)
        if decoded:
            return _validate_image_bytes(decoded[0], decoded[1])
        raise RuntimeError("无法解析 data URL")
    if ref.lower().startswith("base64://"):
        decoded = _decode_base64_ref(ref)
        if decoded:
            return _validate_image_bytes(decoded[0], decoded[1])
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
        _prune_state(state)


history_collector = on_message(priority=99, block=False)
notice_collector = on_notice(priority=99, block=False)
nlp_handler = on_message(priority=15, block=False)
avatar_handler = on_command("处理头像", priority=5)
chat_handler = on_command("技能", aliases={"聊天", "对话"}, priority=5)
weather_handler = on_command("天气", aliases={"查询天气", "查天气"}, priority=5)
user_info_handler = on_command("用户信息", aliases={"查用户", "查用户信息"}, priority=5)
travel_handler = on_command("旅行规划", aliases={"旅行计划", "行程规划", "旅行", "行程"}, priority=5)
web_summary_handler = on_command(
    "网页总结",
    aliases={"总结网页", "总结链接", "链接总结", "网页摘要", "总结github"},
    priority=5,
)
bangumi_handler = on_command(
    "番剧下载",
    aliases={"蜜柑下载", "订阅下载", "番剧订阅", "追番下载"},
    priority=5,
)
forward_view_handler = on_command(
    "查看转发",
    aliases={"转发记录", "查看合并转发", "查看转发聊天记录"},
    priority=5,
)
recall_view_handler = on_command(
    "查看撤回",
    aliases={"撤回记录", "查看撤回消息", "撤回消息"},
    priority=5,
)


@history_collector.handle()
async def _collect_history(bot: Bot, event: MessageEvent):
    session_id = _session_id(event)
    state = _get_state(session_id)

    # Keep a safe plaintext record (avoid leaking CQ/QQ ids to external APIs via history).
    text = await _event_message_text(bot, event)
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


def _find_history_by_message_id(state: SessionState, message_id: int) -> Optional[HistoryItem]:
    for item in reversed(state.history):
        if item.message_id == message_id:
            return item
    return None


def _notice_session_id(event: object) -> Optional[str]:
    group_id = _coerce_int(getattr(event, "group_id", None))
    if group_id is not None:
        return f"group:{group_id}"
    user_id = _coerce_int(getattr(event, "user_id", None))
    if user_id is not None:
        return f"private:{user_id}"
    return None


async def _append_recalled_message(state: SessionState, item: RecalledMessage) -> None:
    async with state.history_lock:
        state.recalled_messages.append(item)
        if len(state.recalled_messages) > _RECALL_MAX_ITEMS:
            state.recalled_messages = state.recalled_messages[-_RECALL_MAX_ITEMS:]
        _prune_state(state, trim_history=False)


def _build_recalled_messages_text(state: SessionState, *, count: int) -> str:
    if not state.recalled_messages:
        return "最近没有记录到撤回消息。"
    size = max(1, min(20, count))
    rows = state.recalled_messages[-size:]
    lines = [f"最近撤回消息（最多{size}条）："]
    for idx, item in enumerate(reversed(rows)):
        time_text = _format_event_time_text(item.ts)
        sender = _normalize_user_name(item.sender_name) or "用户"
        text = _truncate(_collapse_spaces(item.text or ""), 120) or "[无文本内容]"
        head = f"{idx + 1}. {time_text} {sender}"
        if item.sender_user_id:
            head += f"(qq={item.sender_user_id})"
        lines.append(f"{head}：{text}")
    return "\n".join(lines)


@notice_collector.handle()
async def _collect_recall_notice(event):
    notice_type = str(getattr(event, "notice_type", "") or "").strip().lower()
    if notice_type not in {"group_recall", "friend_recall"}:
        return
    session_id = _notice_session_id(event)
    message_id = _coerce_int(getattr(event, "message_id", None))
    if not session_id or message_id is None:
        return
    state = _get_state(session_id)
    history_item = _find_history_by_message_id(state, message_id)
    sender_user_id = str(getattr(event, "user_id", "") or "").strip()
    operator_user_id = str(getattr(event, "operator_id", "") or "").strip()
    sender_name = history_item.user_name if history_item else ""
    text = history_item.text if history_item else ""
    await _append_recalled_message(
        state,
        RecalledMessage(
            ts=_now(),
            message_id=message_id,
            sender_user_id=sender_user_id or (history_item.user_id if history_item else None),
            operator_user_id=operator_user_id or None,
            sender_name=sender_name or None,
            text=text,
        ),
    )


def _is_command_message(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    starts = _command_starts()
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
        "用户信息",
        "查用户",
        "查用户信息",
        "旅行规划",
        "旅行计划",
        "行程规划",
        "旅行",
        "行程",
        "网页总结",
        "总结网页",
        "总结链接",
        "链接总结",
        "网页摘要",
        "总结github",
        "番剧下载",
        "蜜柑下载",
        "订阅下载",
        "番剧订阅",
        "追番下载",
        "查看转发",
        "转发记录",
        "查看合并转发",
        "查看转发聊天记录",
        "查看撤回",
        "撤回记录",
        "查看撤回消息",
        "撤回消息",
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
        line = _format_context_line(
            item.text,
            name,
            user_id=item.user_id,
            message_id=item.message_id,
        )
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
        max_prev = max(0, int(getattr(config, "nlp_context_history_messages", 60)))
    except Exception:
        max_prev = 60
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
    current_user_id = str(event.get_user_id())
    current_message_id = getattr(event, "message_id", None)
    reply_text, reply_name = _extract_reply_context(event, state)
    reply_event = getattr(event, "reply", None)
    reply_message_id = getattr(reply_event, "message_id", None) if reply_event else None
    reply_sender_id = getattr(getattr(reply_event, "sender", None), "user_id", None)
    if reply_sender_id is None and isinstance(reply_message_id, int):
        for item in reversed(state.history):
            if item.message_id == reply_message_id and item.user_id:
                reply_sender_id = item.user_id
                break

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
        reply_line = _format_context_line(
            reply_text,
            reply_name,
            user_id=str(reply_sender_id) if reply_sender_id is not None else None,
            message_id=reply_message_id if isinstance(reply_message_id, int) else None,
        )
    combined = [
        part
        for part in [
            _format_context_line(
                text,
                current_name,
                user_id=current_user_id,
                message_id=current_message_id if isinstance(current_message_id, int) else None,
            ),
            reply_line,
            *prev_texts,
            *future_texts,
        ]
        if part
    ]
    if not combined:
        return _format_context_line(
            text,
            current_name,
            user_id=current_user_id,
            message_id=current_message_id if isinstance(current_message_id, int) else None,
        )
    return "\n".join(combined)


def _build_primary_intent_text(
    event: MessageEvent,
    state: SessionState,
    text: str,
) -> str:
    current_name = _event_user_name(event)
    current_user_id = str(event.get_user_id())
    current_message_id = getattr(event, "message_id", None)
    reply_text, reply_name = _extract_reply_context(event, state)
    reply_event = getattr(event, "reply", None)
    reply_message_id = getattr(reply_event, "message_id", None) if reply_event else None
    reply_sender_id = getattr(getattr(reply_event, "sender", None), "user_id", None)
    if reply_sender_id is None and isinstance(reply_message_id, int):
        for item in reversed(state.history):
            if item.message_id == reply_message_id and item.user_id:
                reply_sender_id = item.user_id
                break
    if not reply_text:
        return _format_context_line(
            text,
            current_name,
            user_id=current_user_id,
            message_id=current_message_id if isinstance(current_message_id, int) else None,
        )
    if reply_text.strip() == text.strip():
        return _format_context_line(
            text,
            current_name,
            user_id=current_user_id,
            message_id=current_message_id if isinstance(current_message_id, int) else None,
        )
    reply_line = _format_context_line(
        reply_text,
        reply_name,
        user_id=str(reply_sender_id) if reply_sender_id is not None else None,
        message_id=reply_message_id if isinstance(reply_message_id, int) else None,
    )
    return "\n".join(
        [
            _format_context_line(
                text,
                current_name,
                user_id=current_user_id,
                message_id=current_message_id if isinstance(current_message_id, int) else None,
            ),
            reply_line,
        ]
    )


def _chat_context_history_messages() -> int:
    try:
        value = int(getattr(config, "nlp_context_history_messages", 60))
    except Exception:
        value = 60
    return max(0, min(60, value + 1))


def _build_chat_prompt(
    event: MessageEvent,
    state: SessionState,
    user_text: str,
) -> str:
    ts = _event_ts(event)
    current_name = _event_user_name(event)
    current_message_id = getattr(event, "message_id", None)
    msg_id = current_message_id if isinstance(current_message_id, int) else None
    reply_text, reply_name = _extract_reply_context(event, state)
    sender_user_id = str(event.get_user_id())
    reply_event = getattr(event, "reply", None)
    reply_message_id = getattr(reply_event, "message_id", None) if reply_event else None
    reply_sender_id = getattr(getattr(reply_event, "sender", None), "user_id", None)

    chat_type = "群聊" if isinstance(event, GroupMessageEvent) else "私聊"
    send_time_text = _format_event_time_text(ts)

    previous = _collect_context_messages(
        state,
        ts=ts,
        limit=_chat_context_history_messages(),
        future=False,
        current_text=user_text,
        current_message_id=msg_id,
    )

    parts = [
        "当前待回复消息:",
        f"聊天类型: {chat_type}",
        f"发送者昵称: {current_name or '用户'}",
        f"发送者QQ: {sender_user_id}",
        f"消息内容: {user_text}",
        "可用语法: [AT:QQ号], [REPLY:消息ID]",
    ]
    if msg_id is not None:
        parts.append(f"消息ID: {msg_id}")
    if send_time_text:
        parts.append(f"发送时间: {send_time_text}")
    if reply_text:
        parts.extend(
            [
                "被回复消息:",
                f"发送者昵称: {reply_name or '用户'}",
                f"消息内容: {reply_text}",
            ]
        )
        if isinstance(reply_message_id, int):
            parts.append(f"消息ID: {reply_message_id}")
        if reply_sender_id is not None:
            parts.append(f"发送者QQ: {reply_sender_id}")
    if previous:
        parts.extend(
            [
                "最近相关对话(仅供参考，不需要逐条回复):",
                "\n".join(previous),
            ]
        )
    parts.append("请只回复当前这条待回复消息。")
    return "\n".join(parts)


_ALLOWED_ACTIONS = {
    "chat",
    "image_chat",
    "image_generate",
    "image_create",
    "weather",
    "avatar_get",
    "user_info",
    "travel_plan",
    "web_summary",
    "bangumi_download",
    "forward_view",
    "recall_view",
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
    "sender_user",
    "qq_user",
    "message_id",
    "wait_next",
    "city",
    "trip",
    "url",
    "mikan",
    "message_forward",
    "reply_forward",
    "recent_recall",
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
    has_forward: bool,
    has_reply_forward: bool,
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

    if action == "forward_view":
        if target not in _ALLOWED_TARGETS:
            target = ""
        if not target or target == "none":
            if has_forward:
                target = "message_forward"
            elif has_reply_forward:
                target = "reply_forward"
            else:
                target = "message_forward"
        limit = _coerce_int(params.get("limit"))
        normalized_params: dict[str, object] = {}
        if limit is not None and limit > 0:
            normalized_params["limit"] = min(30, limit)
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": target,
            "params": normalized_params,
        }

    if action == "recall_view":
        count = _coerce_int(params.get("count"))
        normalized_params: dict[str, object] = {}
        if count is not None and count > 0:
            normalized_params["count"] = min(20, count)
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": "recent_recall",
            "params": normalized_params,
        }

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

    if action == "user_info":
        if target not in _ALLOWED_TARGETS:
            target = ""
        if not target or target == "none":
            target = "at_user" if at_users else "sender_user"
        normalized_params: dict[str, object] = {}
        raw_qq = params.get("qq")
        if isinstance(raw_qq, str) and raw_qq.strip():
            qq = raw_qq.strip()
            if qq.isdigit():
                normalized_params["qq"] = qq
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": target,
            "params": normalized_params,
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

    if action == "web_summary":
        raw_url = params.get("url")
        url = raw_url.strip() if isinstance(raw_url, str) else ""
        raw_focus = params.get("focus")
        focus = raw_focus.strip() if isinstance(raw_focus, str) else ""
        normalized_params: dict[str, object] = {}
        if url:
            normalized_params["url"] = url
        if focus:
            normalized_params["focus"] = focus
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": "url",
            "params": normalized_params,
        }

    if action == "bangumi_download":
        normalized_params: dict[str, object] = {}
        raw_keyword = params.get("keyword") or params.get("title")
        if isinstance(raw_keyword, str) and raw_keyword.strip():
            normalized_params["keyword"] = raw_keyword.strip()
        raw_uri = params.get("uri") or params.get("download_url") or params.get("url")
        if isinstance(raw_uri, str) and raw_uri.strip():
            normalized_params["uri"] = raw_uri.strip()
        episode = _coerce_int(params.get("episode"))
        if episode is not None:
            normalized_params["episode"] = episode
        latest = _coerce_bool(params.get("latest"))
        if latest is not None:
            normalized_params["latest"] = latest
        raw_subtitle_group = params.get("subtitle_group") or params.get("subgroup")
        if isinstance(raw_subtitle_group, str) and raw_subtitle_group.strip():
            normalized_params["subtitle_group"] = raw_subtitle_group.strip()
        mode = _subscription_mode(str(params.get("mode") or ""))
        if mode != "download":
            normalized_params["mode"] = mode
        subscribe = _coerce_bool(params.get("subscribe"))
        if subscribe is not None:
            normalized_params["subscribe"] = subscribe
        raw_reply = params.get("reply")
        if isinstance(raw_reply, str) and raw_reply.strip():
            normalized_params["reply"] = raw_reply.strip()
        return {
            "action": action,
            "target": "mikan",
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
    forward_ids: List[str],
    reply_forward_ids: List[str],
    at_users: List[str],
    send_func,
) -> None:
    # 意图分发：按 action 走不同处理链路
    action = str(intent.get("action", "ignore")).lower()
    if action == "ignore":
        return
    bangumi_text_reply = _bangumi_send_text_reply()
    transition_sent = False
    if action in {
        "weather",
        "travel_plan",
        "web_summary",
        "bangumi_download",
        "user_info",
        "avatar_get",
        "image_chat",
        "image_generate",
        "image_create",
    }:
        transition_text = _intent_transition_text(intent)
        transition_text = _format_reply_text(transition_text)
        if transition_text and (action != "bangumi_download" or bangumi_text_reply):
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
        "forward_view",
        "recall_view",
        "weather",
        "travel_plan",
        "web_summary",
        "bangumi_download",
        "user_info",
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
        prompt = _build_chat_prompt(event, state, user_text)
        try:
            reply = await _call_gemini_text(str(prompt), state)
            if _is_skip_reply_text(reply):
                _mark_handled_request(state, event, text)
                return
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

    if action == "forward_view":
        target = str(intent.get("target") or "").lower()
        params = _intent_params(intent)
        limit = _coerce_int(params.get("limit"))
        node_limit = max(1, min(30, limit if limit is not None else _FORWARD_FETCH_NODE_LIMIT))
        chosen_ids: List[str] = []
        if target == "message_forward":
            chosen_ids = forward_ids
        elif target == "reply_forward":
            chosen_ids = reply_forward_ids
        else:
            chosen_ids = forward_ids or reply_forward_ids
        if not chosen_ids:
            await send_func("没有找到可查看的转发聊天记录。")
            return
        summary = await _build_forward_summary_text(
            bot,
            chosen_ids,
            limit_nodes=node_limit,
            limit_chars=_FORWARD_FETCH_CHAR_LIMIT,
            header_prefix="转发聊天记录",
        )
        if not summary:
            await send_func("转发聊天记录读取失败或内容为空。")
            return
        await _append_history(
            state,
            "user",
            f"查看转发：{raw_message_text or text}",
            user_id=str(event.get_user_id()),
            user_name=user_name,
            to_bot=True,
        )
        await _append_history(state, "model", summary)
        await _send_text_response(bot, event, send_func, summary)
        _mark_handled_request(state, event, text)
        return

    if action == "recall_view":
        params = _intent_params(intent)
        count = _coerce_int(params.get("count"))
        reply = _build_recalled_messages_text(state, count=count or 5)
        await _append_history(
            state,
            "user",
            f"查看撤回：{raw_message_text or text}",
            user_id=str(event.get_user_id()),
            user_name=user_name,
            to_bot=True,
        )
        await _append_history(state, "model", reply)
        await _send_text_response(bot, event, send_func, reply)
        _mark_handled_request(state, event, text)
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

    if action == "web_summary":
        # 网页总结
        raw_text = _strip_leading_command(
            raw_message_text,
            ("网页总结", "总结网页", "总结链接", "链接总结", "总结github", "网页摘要"),
        )
        if not raw_text:
            raw_text = raw_message_text.strip()
        if not raw_text:
            return
        normalized_url = _resolve_web_summary_url(intent, raw_text)
        if not normalized_url:
            return
        if not transition_sent:
            await _send_transition(action, send_func)
        try:
            reply = await _build_web_summary_reply(intent, state, event, raw_text=raw_text)
            if not reply:
                return
            await _send_text_response(bot, event, send_func, reply)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP web summary failed: {}", _safe_error_message(exc))
            await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action == "bangumi_download":
        # 蜜柑番剧下载 / 订阅下载（aria2）
        raw_text = _strip_leading_command(raw_message_text, _BANGUMI_COMMAND_WORDS)
        if not raw_text:
            raw_text = raw_message_text.strip()
        if not raw_text:
            raw_text = text.strip()
        if bangumi_text_reply and not transition_sent:
            await _send_transition(action, send_func)
        try:
            reply = await _build_bangumi_download_reply(
                bot,
                intent,
                state,
                event,
                raw_text=raw_text,
            )
            if not reply:
                return
            await _append_history(
                state,
                "user",
                f"番剧下载：{raw_text or text}",
                user_id=str(event.get_user_id()),
                user_name=user_name,
                to_bot=True,
            )
            if bangumi_text_reply:
                await _append_history(state, "model", reply)
                await _send_text_response(bot, event, send_func, reply)
            _mark_handled_request(state, event, text)
        except Exception as exc:
            logger.error("NLP bangumi download failed: {}", _safe_error_message(exc))
            if bangumi_text_reply:
                await send_func(f"出错了：{_safe_error_message(exc)}")
        return

    if action == "history_clear":
        # 清空当前会话历史
        _clear_session_state(state)
        await send_func("已清除当前会话记录，可以继续聊啦。")
        return

    if action == "user_info":
        # 查询用户信息（发送者/@用户/指定QQ）
        reply = await _build_user_info_reply(
            bot,
            event,
            intent,
            at_users=at_users,
        )
        if not reply:
            return
        await _append_history(
            state,
            "user",
            f"用户信息查询：{raw_message_text or text}",
            user_id=str(event.get_user_id()),
            user_name=user_name,
            to_bot=True,
        )
        await _append_history(state, "model", reply)
        await _send_text_response(bot, event, send_func, reply)
        _mark_handled_request(state, event, text)
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
        return "我没太听懂，你是想聊这张图、处理图片、查天气、查用户信息、旅行规划、网页总结、番剧下载、查看转发还是查看撤回？"
    return "我没太听懂，你是想聊天、处理图片、无图生成、查天气、查用户信息、旅行规划、网页总结、番剧下载、查看转发、查看撤回还是清除历史？"


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
    has_forward: bool,
    has_reply_forward: bool,
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
        f"消息包含转发: {has_forward}\n"
        f"回复里有转发: {has_reply_forward}\n"
        f"是否@用户: {bool(at_users)}\n"
        f"是否有最近图片: {bool(state.last_image_id)}\n"
        f"是否有撤回记录: {bool(state.recalled_messages)}\n"
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
    forward_ids = _extract_forward_ids(event.get_message())
    # Avoid leaking CQ/QQ ids to external APIs in intent classification.
    plain_text = _sanitize_cq_tokens(event.get_plaintext().strip())
    hint_text = _build_bangumi_nlp_hint(event.get_message(), plain_text)
    text = plain_text or hint_text
    if not text and forward_ids:
        text = "查看转发聊天记录"
    if not text:
        return
    if plain_text and _is_command_message(plain_text):
        return
    if str(event.get_user_id()) == str(event.self_id):
        return
    if not _should_trigger_nlp(event, plain_text or text):
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
    reply_forward_ids = _extract_reply_forward_ids(event)
    has_image = image_url is not None
    has_reply_image = reply_image_url is not None
    has_forward = bool(forward_ids)
    has_reply_forward = bool(reply_forward_ids)

    try:
        primary_text = _build_primary_intent_text(event, state, text)
        intent_raw = await _classify_intent(
            primary_text,
            state,
            has_image,
            has_reply_image,
            has_forward,
            has_reply_forward,
            at_users,
        )
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        return

    intent = _normalize_intent(
        intent_raw,
        has_image,
        has_reply_image,
        has_forward,
        has_reply_forward,
        at_users,
        state,
    )
    if not intent:
        try:
            intent_text = await _build_intent_text(event, state, text)
            if intent_text and intent_text != primary_text:
                intent_raw = await _classify_intent(
                    intent_text,
                    state,
                    has_image,
                    has_reply_image,
                    has_forward,
                    has_reply_forward,
                    at_users,
                )
                intent = _normalize_intent(
                    intent_raw,
                    has_image,
                    has_reply_image,
                    has_forward,
                    has_reply_forward,
                    at_users,
                    state,
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
        forward_ids=forward_ids,
        reply_forward_ids=reply_forward_ids,
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
    forward_ids = _extract_forward_ids(event.get_message())
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
    reply_forward_ids = _extract_reply_forward_ids(event)
    has_image = image_url is not None
    has_reply_image = reply_image_url is not None
    has_forward = bool(forward_ids)
    has_reply_forward = bool(reply_forward_ids)
    try:
        intent_raw = await _classify_intent(
            text,
            state,
            has_image,
            has_reply_image,
            has_forward,
            has_reply_forward,
            at_users,
        )
    except Exception as exc:
        logger.error("Intent classify failed: {}", _safe_error_message(exc))
        await send_func("意图解析失败，请稍后再试。")
        return
    intent = _normalize_intent(
        intent_raw,
        has_image,
        has_reply_image,
        has_forward,
        has_reply_forward,
        at_users,
        state,
    )
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
        forward_ids=forward_ids,
        reply_forward_ids=reply_forward_ids,
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


@user_info_handler.handle()
async def handle_user_info(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    text = f"用户信息 {query}".strip()
    await _handle_command_via_intent(
        bot,
        event,
        text=text,
        send_func=user_info_handler.send,
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


@web_summary_handler.handle()
async def handle_web_summary(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    if not query:
        return
    await _handle_command_via_intent(
        bot,
        event,
        text=f"网页总结 {query}",
        send_func=web_summary_handler.send,
    )


@bangumi_handler.handle()
async def handle_bangumi(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    if not query:
        await bangumi_handler.finish(
            "请提供番剧名，例如：番剧下载 葬送的芙莉莲 最新一集 或 订阅下载 葬送的芙莉莲"
        )
    await _handle_command_via_intent(
        bot,
        event,
        text=f"番剧下载 {query}",
        send_func=bangumi_handler.send,
    )


@forward_view_handler.handle()
async def handle_forward_view(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    text = f"查看转发 {query}".strip()
    await _handle_command_via_intent(
        bot,
        event,
        text=text,
        send_func=forward_view_handler.send,
    )


@recall_view_handler.handle()
async def handle_recall_view(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    query = args.extract_plain_text().strip()
    text = f"查看撤回 {query}".strip()
    await _handle_command_via_intent(
        bot,
        event,
        text=text,
        send_func=recall_view_handler.send,
    )
