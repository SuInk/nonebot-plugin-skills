from typing import Optional
from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.memory import memory_core

history_schema = {
    "properties": {},
}

@skill_manager.register("history_clear", "清除当前用户的上下文对话历史记录，这在遇到需要重置对话状态时很有用", history_schema)
async def history_clear(context: Optional[SkillContext] = None) -> str:
    if not context:
        return "无法获取上下文信息。"
    
    # 清空该 session_id 下的短期记忆
    if context.session_id in memory_core.sessions:
        memory_core.sessions[context.session_id] = []
        return "您的对话历史已清空，我们可以重新开始了。"
    
    return "您的对话历史当前已经是空的。"
