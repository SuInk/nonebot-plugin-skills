from typing import Optional, Any
from pydantic import BaseModel
from nonebot.adapters.onebot.v11 import Bot, MessageEvent

class SkillContext(BaseModel):
    bot: Any # nonebot.adapters.onebot.v11.Bot
    event: Any # nonebot.adapters.onebot.v11.MessageEvent
    session_id: str
    user_id: str
    raw_text: str

    class Config:
        arbitrary_types_allowed = True
