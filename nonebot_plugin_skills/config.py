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
    forward_line_threshold: int = 8
    gemini_log_response: bool = False
    nlp_enable: bool = True
    bot_keywords: List[str] = []


config = get_plugin_config(Config)
