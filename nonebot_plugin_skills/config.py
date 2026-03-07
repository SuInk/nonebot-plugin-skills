from __future__ import annotations

from typing import List
from nonebot import get_plugin_config
from pydantic import BaseModel

class Config(BaseModel):
    # --- 核心 AI 配置 ---
    google_api_key: str = ""
    
    # 基础对话模型 (使用 2.0 Flash 保证极速对话)
    gemini_text_model: str = "gemini-2.0-flash"
    
    # 统一视觉/生图模型 (集生成与修改于一体)
    gemini_image_model: str = "gemini-3.1-flash-image-preview"
    
    # --- 基础运行配置 ---
    request_timeout: float = 30.0
    nlp_enable: bool = True
    bot_keywords: List[str] = ["嘉然", "然然", "Diana"]
    chat_style_temperature: float = 0.7
    
    # 消息分段阈值 (多少字以上尝试合并为转发消息)
    combine_message_threshold: int = 80
    # 是否记录大模型原始响应日志
    gemini_log_response: bool = False
    
    # 管理员 QQ 列表 (字符串列表)
    admin_qqs: List[str] = []

    class Config:
        extra = "allow"

config = get_plugin_config(Config)
