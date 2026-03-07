from typing import Optional
from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.memory import memory_core

recall_schema = {
    "properties": {
        "count": {"type": "INTEGER", "description": "查看最近几条撤回消息，默认为1"}
    }
}

import time

@skill_manager.register("recall_view", "查看群聊中最近24小时内被撤回的消息内容，包含昵称、时间和内容", recall_schema)
async def recall_view(count: int = 5, context: Optional[SkillContext] = None) -> str:
    if not context:
        return "无法获取上下文信息。"
    
    all_recalls = memory_core.get_recalls(context.session_id)
    if not all_recalls:
        return "目前还没有记录到任何撤回消息呢。"
    
    # 过滤 24 小时内的消息
    now = time.time()
    day_sec = 24 * 60 * 60
    recalls = [r for r in all_recalls if now - r.timestamp <= day_sec]
    
    if not recalls:
        return "最近 24 小时内没有发现撤回记录哦。"
    
    count = max(1, min(count, len(recalls)))
    target_recalls = recalls[-count:]
    
    lines = [f"最近 24 小时内发现了 {len(target_recalls)} 条撤回的消息："]
    for r in target_recalls:
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp))
        lines.append(f"--- 撤回记录 ---\n用户: {r.nickname} ({r.user_id})\n时间: {local_time}\n内容: {r.text}")
        
    return "\n".join(lines)
