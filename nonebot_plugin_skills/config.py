from __future__ import annotations

from typing import List

from nonebot import get_plugin_config
from pydantic import BaseModel


class Config(BaseModel):
    google_api_key: str = ""
    gemini_text_model: str = "gemini-3-flash-preview"
    gemini_image_model: str = "nano-banana-pro-preview"
    request_timeout: float = 30.0
    image_timeout: float = 120.0
    image_download_max_bytes: int = 15728640
    web_fetch_max_bytes: int = 2097152
    web_extract_max_chars: int = 12000
    history_ttl_sec: int = 600
    history_max_messages: int = 60
    image_cache_max_images: int = 10
    history_reference_only: bool = True
    forward_char_threshold: int = 150
    forward_line_threshold: int = 8
    message_send_delay_sec: float = 0.6
    gemini_log_response: bool = False
    nlp_enable: bool = True
    nlp_context_history_messages: int = 60
    nlp_context_future_messages: int = 2
    nlp_context_future_wait_sec: float = 1.0
    bot_keywords: List[str] = []
    mikan_base_url: str = "https://mikanani.me"
    mikan_hosts: List[str] = []
    mikan_search_limit: int = 120
    bangumi_ai_classify_limit: int = 60
    aria2_rpc_url: str = "http://127.0.0.1:6800/jsonrpc"
    aria2_rpc_secret: str = ""
    aria2_download_dir: str = "download"
    aria2_timeout: float = 20.0
    aria2_bt_trackers: List[str] = [
        "udp://tracker.opentrackr.org:1337/announce",
        "udp://tracker.openbittorrent.com:6969/announce",
    ]
    aria2_bt_max_peers: int = 200
    aria2_disable_seed: bool = True
    bangumi_auto_send_file: bool = True
    bangumi_auto_delete_after_send: bool = True
    bangumi_send_text_reply: bool = True
    bangumi_progress_notify_interval_sec: float = 20.0
    bangumi_download_watch_timeout_sec: float = 21600.0
    bangumi_torrent_max_bytes: int = 4194304
    bangumi_torrent_illegal_keywords: List[str] = [
        "色情",
        "成人视频",
        "无码",
        "里番",
        "hentai",
        "博彩",
        "赌博",
    ]
    bangumi_torrent_block_exts: List[str] = [
        ".exe",
        ".bat",
        ".cmd",
        ".ps1",
        ".vbs",
        ".js",
        ".scr",
        ".dll",
    ]
    bangumi_subscription_enable: bool = True
    bangumi_subscription_file: str = "data/nonebot_plugin_skills/bangumi_subscriptions.json"


config = get_plugin_config(Config)
