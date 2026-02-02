from __future__ import annotations

from typing import List

from nonebot import get_plugin_config
from pydantic import BaseModel


class Config(BaseModel):
    google_api_key: str = ""
    gemini_text_model: str = "gemini-3-pro-preview"
    gemini_image_model: str = "nano-banana-pro-preview"
    request_timeout: float = 30.0
    image_timeout: float = 120.0
    history_ttl_sec: int = 600
    history_max_messages: int = 10
    history_compress_enable: bool = True
    history_compress_trigger: int = 20
    history_compress_keep: int = 6
    history_compress_min_messages: int = 6
    history_compress_max_chars: int = 600
    history_reference_only: bool = True
    forward_line_threshold: int = 8
    message_send_delay_sec: float = 0.6
    gemini_log_response: bool = False
    nlp_enable: bool = True
    bot_keywords: List[str] = []


config = get_plugin_config(Config)
