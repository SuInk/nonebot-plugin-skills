from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import time
import json
import asyncio
from pathlib import Path
from nonebot import logger

class ChatMessage(BaseModel):
    role: str
    content: str
    message_id: Optional[str] = None # 新增字段
    timestamp: float = Field(default_factory=time.time)

class UserProfile(BaseModel):
    user_id: str
    username: str
    facts: List[str] = Field(default_factory=list)

class RecalledMessage(BaseModel):
    message_id: int
    user_id: str
    nickname: str
    text: str
    timestamp: float = Field(default_factory=time.time)

class MemoryCore:
    def __init__(self, db_path: str = "data/nonebot_plugin_skills/v2_memory.json"):
        self.db_path = Path(db_path)
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.profiles: Dict[str, UserProfile] = {}
        self.recalls: Dict[str, List[RecalledMessage]] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text("utf-8"))
                for uid, prof_data in data.get("profiles", {}).items():
                    self.profiles[uid] = UserProfile(**prof_data)
                for sid, history_data in data.get("sessions", {}).items():
                    self.sessions[sid] = [ChatMessage(**m) for m in history_data]
                logger.info(f"Memory loaded with IDs from {self.db_path}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")

    def _save(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "profiles": {uid: p.dict() for uid, p in self.profiles.items()},
                "sessions": {sid: [m.dict() for m in history] for sid, history in self.sessions.items()},
            }
            self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def add_message(self, session_id: str, role: str, content: Any, message_id: Optional[Any] = None, max_history: int = 20):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # 强制格式化
        mid = str(message_id) if message_id is not None else None
        self.sessions[session_id].append(ChatMessage(role=role, content=str(content), message_id=mid))
        
        if len(self.sessions[session_id]) > max_history:
            self.sessions[session_id] = self.sessions[session_id][-max_history:]
        
        asyncio.create_task(self._async_save())

    async def _async_save(self):
        self._save()

    def get_history(self, session_id: str) -> List[ChatMessage]:
        return self.sessions.get(session_id, [])

    def get_profile(self, user_id: str, username: str = "Unknown") -> UserProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id, username=username)
        return self.profiles[user_id]

    def add_fact(self, user_id: str, fact: str):
        profile = self.get_profile(user_id)
        if fact not in profile.facts:
            profile.facts.append(fact)
            self._save()

    def clear_facts(self, user_id: str):
        if user_id in self.profiles:
            self.profiles[user_id].facts = []
            self._save()

    def clear_history(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id] = []
            self._save()

    def add_recall(self, session_id: str, recall: RecalledMessage, max_recalls: int = 20):
        if session_id not in self.recalls:
            self.recalls[session_id] = []
        self.recalls[session_id].append(recall)
        if len(self.recalls[session_id]) > max_recalls:
            self.recalls[session_id] = self.recalls[session_id][-max_recalls:]

    def get_recalls(self, session_id: str) -> List[RecalledMessage]:
        return self.recalls.get(session_id, [])

    def get_context_prompt(self, session_id: str, user_id: str) -> str:
        profile = self.get_profile(user_id)
        prompt = ""
        if profile.facts:
            prompt += f"关于用户 {profile.username} 的已知记忆:\n"
            for i, fact in enumerate(profile.facts[-10:]):
                prompt += f"{i+1}. {fact}\n"
        
        # 增加对历史记录中 ID 的提示
        history = self.get_history(session_id)
        if any(m.message_id for m in history):
            prompt += "\n(你可以通过历史记录中的 [ID: xxx] 来识别用户正在回复哪条消息。)"
            
        return prompt

memory_core = MemoryCore()
