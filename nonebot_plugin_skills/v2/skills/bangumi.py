import asyncio
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple, Any
from urllib.parse import quote, urlsplit
from pathlib import Path

from pydantic import BaseModel
from nonebot import get_plugin_config, logger

from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.http import get_http_client
from nonebot_plugin_skills.v2.core.context import SkillContext

# --- Config ---
class BangumiConfig(BaseModel):
    mikan_base_url: str = "https://mikanani.me"
    mikan_search_limit: int = 40
    aria2_rpc_url: str = "http://127.0.0.1:6800/jsonrpc"
    aria2_rpc_secret: str = ""
    aria2_download_dir: str = "download"
    aria2_timeout: float = 20.0
    bangumi_subscription_file: str = "data/nonebot_plugin_skills/bangumi_subscriptions.json"
    bangumi_subscription_enable: bool = True
    bangumi_watch_interval: float = 5.0

    class Config:
        extra = "allow"

try:
    config = get_plugin_config(BangumiConfig)
except Exception:
    config = BangumiConfig()

# --- Data Models ---
class BangumiRelease(BaseModel):
    title: str
    download_url: str
    guid: str

# --- XML Helpers ---
def _get_child_text_lenient(item: ET.Element, tag_name: str) -> str:
    """不计命名空间地查找子节点文本"""
    for child in list(item):
        local_name = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag
        if local_name == tag_name:
            return (child.text or "").strip()
    return ""

def _get_enclosure_url(item: ET.Element) -> str:
    """查找附件 URL"""
    for child in list(item):
        local_name = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag
        if local_name == "enclosure":
            return child.attrib.get("url", "").strip()
    return ""

# --- Mikan Search ---
async def _search_mikan_releases(keyword: str) -> List[BangumiRelease]:
    if not keyword: return []
    client = get_http_client()
    # 尝试加上 RSS 搜索路径
    url = f"{config.mikan_base_url}/RSS/Search?searchstr={quote(keyword)}"
    logger.info(f"Searching Mikan: {url}")
    
    try:
        resp = await client.get(url, timeout=15.0)
        resp.raise_for_status()
        
        # 使用 bytes 解析以自动处理编码
        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            logger.warning("Mikan RSS response has no channel node.")
            return []
            
        releases = []
        for item in channel.findall("item"):
            title = _get_child_text_lenient(item, "title")
            # 优先从 enclosure 找下载链接，其次从 torrent 或 link 找
            download_url = _get_enclosure_url(item)
            if not download_url:
                download_url = _get_child_text_lenient(item, "torrent")
            if not download_url:
                download_url = _get_child_text_lenient(item, "link")
                
            if title and download_url:
                releases.append(BangumiRelease(
                    title=title,
                    download_url=download_url,
                    guid=title
                ))
        
        logger.info(f"Found {len(releases)} releases for '{keyword}'")
        return releases[:config.mikan_search_limit]
    except Exception as e:
        logger.error(f"Mikan search failed for '{keyword}': {e}")
        return []

# --- Aria2 Helpers ---
async def _aria2_rpc_call(method: str, params: List[Any]) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": ([f"token:{config.aria2_rpc_secret}"] if config.aria2_rpc_secret else []) + params
    }
    client = get_http_client()
    resp = await client.post(config.aria2_rpc_url, json=payload, timeout=config.aria2_timeout)
    resp.raise_for_status()
    return resp.json()

async def _aria2_add_uri(uri: str) -> str:
    options = {"dir": config.aria2_download_dir} if config.aria2_download_dir else {}
    data = await _aria2_rpc_call("aria2.addUri", [[uri], options])
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Unknown error"))
    return data.get("result", "")

async def _watch_download(bot: Any, context: SkillContext, gid: str, title: str):
    while True:
        await asyncio.sleep(config.bangumi_watch_interval)
        try:
            res = await _aria2_rpc_call("aria2.tellStatus", [gid, ["status", "errorMessage"]])
            status = res.get("result", {}).get("status")
            if status == "complete":
                await _send_notification(bot, context, f"✨ 报！番剧下载好啦！\n标题: {title}")
                break
            elif status == "error":
                err = res.get("result", {}).get("errorMessage", "未知错误")
                await _send_notification(bot, context, f"❌ 下载失败了...\n标题: {title}\n原因: {err}")
                break
            elif status in ["removed", None]: break
        except: break

async def _send_notification(bot: Any, context: SkillContext, message: str):
    try:
        if "group_" in context.session_id:
            await bot.send_group_msg(group_id=int(context.session_id.replace("group_", "")), message=message)
        else:
            await bot.send_private_msg(user_id=int(context.session_id.replace("private_", "")), message=message)
    except: pass

# --- Skill Definition ---
bangumi_schema = {
    "properties": {
        "keyword": {"type": "STRING", "description": "番剧名称"},
        "episode": {"type": "INTEGER", "description": "集数 (可选)"}
    },
    "required": ["keyword"]
}

@skill_manager.register("bangumi_download", "搜索番剧并下载。只需要名字即可，嘉然会帮你搞定一切~", bangumi_schema)
async def bangumi_download(keyword: str, episode: Optional[int] = None, context: Optional[SkillContext] = None) -> str:
    if not keyword: return "请告诉我番剧名字哦~"
    
    releases = await _search_mikan_releases(keyword)
    if not releases:
        return f"呜呜... 在蜜柑上没搜到《{keyword}》，换个关键词试试？"

    selected = releases[0]
    if episode is not None:
        ep_str = str(episode).zfill(2)
        for r in releases:
            if f"[{ep_str}]" in r.title or f" {ep_str} " in r.title or f"第{episode}集" in r.title:
                selected = r
                break

    try:
        gid = await _aria2_add_uri(selected.download_url)
        if context and context.bot:
            asyncio.create_task(_watch_download(context.bot, context, gid, selected.title))
        return f"好哒！已经开始下载《{selected.title}》啦~\n任务ID: {gid}\n下载好了嘉然会叫你哒！"
    except Exception as e:
        return f"呜呜，提交到 Aria2 失败了: {e}"
