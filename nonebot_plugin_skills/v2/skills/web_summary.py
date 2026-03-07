from typing import Optional
from pydantic import BaseModel
from nonebot import get_plugin_config, logger
import re
import httpx
from urllib.parse import urlsplit

from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.http import get_http_client
from nonebot_plugin_skills.v2.core.context import SkillContext

# --- Config ---
class WebSummaryConfig(BaseModel):
    web_fetch_max_bytes: int = 2097152
    web_extract_max_chars: int = 12000
    request_timeout: float = 30.0

    class Config:
        extra = "allow"

try:
    config = get_plugin_config(WebSummaryConfig)
except Exception:
    config = WebSummaryConfig()

# --- Helper Functions ---
_HTML_DROP_BLOCK_RE = re.compile(r"<(script|style|noscript|svg|iframe|canvas)[^>]*>.*?</\1>", re.I | re.S)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

async def _fetch_web_text(url: str) -> str:
    client = get_http_client()
    try:
        resp = await client.get(url, timeout=config.request_timeout)
        resp.raise_for_status()
        html = resp.text
        
        # 简单清洗 HTML
        cleaned = _HTML_DROP_BLOCK_RE.sub("", html)
        text = _HTML_TAG_RE.sub(" ", cleaned)
        text = re.sub(r"\s+", " ", text).strip()
        
        if len(text) > config.web_extract_max_chars:
            text = text[:config.web_extract_max_chars] + "...(truncated)"
        return text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return f"获取网页失败: {str(e)}"

# --- Skill Definition ---
web_summary_schema = {
    "properties": {
        "url": {"type": "STRING", "description": "要总结的网页链接 (必须以 http:// 或 https:// 开头)"},
        "focus": {"type": "STRING", "description": "可选的总结侧重点或用户特别关注的问题"}
    },
    "required": ["url"]
}

@skill_manager.register("web_summary", "读取并提取指定网页的内容，这有助于回答需要查阅特定网页的问题", web_summary_schema)
async def web_summary(url: str, focus: Optional[str] = None, context: Optional[SkillContext] = None) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return "不支持的链接格式，仅支持 http 和 https。"

    text_content = await _fetch_web_text(url)
    
    if text_content.startswith("获取网页失败"):
        return text_content
        
    result = f"网页 {url} 的内容提取如下:\n{text_content}\n\n"
    if focus:
        result += f"请根据上述内容，重点总结或回答: {focus}"
    else:
        result += "请提炼并总结上述网页的核心内容。"
        
    return result
