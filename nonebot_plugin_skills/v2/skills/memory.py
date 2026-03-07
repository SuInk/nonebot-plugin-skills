from typing import Optional
from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.memory import memory_core
from nonebot_plugin_skills.config import config

memory_record_schema = {
    "properties": {
        "memory": {"type": "STRING", "description": "要记住的事项、喜好、事实等具体内容。"},
        "qq": {"type": "STRING", "description": "要记录在哪个QQ号名下。如果未提供，默认记录在当前说话人名下。"}
    },
    "required": ["memory"]
}

@skill_manager.register("memory_record", "记录一条关于用户的长期记忆（例如用户的喜好、生日、特定事实），这将在未来的对话中被AI记住", memory_record_schema)
async def memory_record(memory: str, qq: Optional[str] = None, context: Optional[SkillContext] = None) -> str:
    if not context or not context.bot or not context.event:
        return "无法获取上下文信息。"
    
    target_qq = qq or str(context.event.get_user_id())
    
    memory_core.add_fact(target_qq, memory)
    
    if target_qq == str(context.event.get_user_id()):
        return f"我记住了：{memory}"
    else:
        return f"我已将关于 {target_qq} 的记忆保存：{memory}"

memory_clear_schema = {
    "properties": {
        "qq": {"type": "STRING", "description": "要清除记忆的QQ号。如果不传，默认为清除当前发消息用户的记忆。"},
        "full_history": {"type": "BOOLEAN", "description": "是否同时清除当前会话的所有上下文聊天记录。默认为 false。"}
    }
}

@skill_manager.register("clear_memory", "清除长期记忆或会话历史。用户可以清除自己的记忆，管理员可以清除任何人的记忆或重置整个会话。", memory_clear_schema)
async def clear_memory(qq: Optional[str] = None, full_history: bool = False, context: Optional[SkillContext] = None) -> str:
    if not context: return "无法获取上下文。"

    sender_qq = str(context.user_id)
    target_qq = str(qq) if qq else sender_qq
    is_admin = sender_qq in config.admin_qqs

    # 权限校验：只能清除自己的，除非是管理员
    if target_qq != sender_qq and not is_admin:
        return "唔... 只有管理员或者本人才能清除这段记忆哦~"

    res_parts = []

    # 1. 清除长期记忆 (facts)
    if target_qq:
        memory_core.clear_facts(target_qq)
        res_parts.append(f"关于 {'你' if target_qq == sender_qq else target_qq} 的长期记忆已经全部清除啦！")

    # 2. 清除会话历史记录 (针对整个 session)
    if full_history:
        if not is_admin:
            res_parts.append("(由于你不是管理员，重置整个会话历史的请求被忽略了呢)")
        else:
            memory_core.clear_history(context.session_id)
            res_parts.append(f"当前会话 {context.session_id} 的聊天记录也已经重置了哦。")

    return "\n".join(res_parts)

