from typing import Optional, List
from nonebot.adapters.onebot.v11 import Message
from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.v2.core.memory import memory_core, RecalledMessage

recall_schema = {
    "properties": {
        "count": {"type": "INTEGER", "description": "查看最近撤回的消息"}
    }
}

import time


def _get_recent_recalls(count: int, context: Optional[SkillContext]) -> List[RecalledMessage]:
    if not context:
        return []

    all_recalls = memory_core.get_recalls(context.session_id)
    if not all_recalls:
        return []

    now = time.time()
    day_sec = 24 * 60 * 60
    recalls = [r for r in all_recalls if now - r.timestamp <= day_sec]
    if not recalls:
        return []

    count = max(1, min(count, len(recalls)))
    return recalls[-count:]


async def _render_recall_content(text: str) -> str:
    rendered_parts = []
    parsed = Message(text)

    for seg in parsed:
        if seg.type == "text":
            rendered_parts.append(seg.data.get("text", ""))
            continue

        if seg.type == "image":
            source = seg.data.get("file") or seg.data.get("url")
            if source:
                try:
                    from nonebot_plugin_skills.v2.core.nlp import _get_image_description, _load_image_bytes

                    image_data = await _load_image_bytes(source)
                    if image_data:
                        rendered_parts.append(await _get_image_description(image_data))
                        continue
                except Exception:
                    pass
            rendered_parts.append("[图片]")
            continue

        if seg.type == "face":
            rendered_parts.append("[表情]")
            continue

        rendered_parts.append(f"[{seg.type}]")

    return "".join(rendered_parts).strip()


async def build_recall_forward_messages(count: int, context: Optional[SkillContext]) -> List[Message]:
    recalls = _get_recent_recalls(count, context)
    messages: List[Message] = []

    for r in recalls:
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp))
        rendered_text = await _render_recall_content(r.text)
        messages.append(
            Message(
                f"用户: {r.nickname} ({r.user_id})\n时间: {local_time}\n内容:\n{rendered_text}"
            )
        )

    return messages

@skill_manager.register("recall_view", "查看群聊中最近24小时内被撤回的消息内容，包含昵称、时间和内容", recall_schema)
async def recall_view(count: int = 10, context: Optional[SkillContext] = None) -> str:
    if not context:
        return "无法获取上下文信息。"

    target_recalls = _get_recent_recalls(count, context)
    if not target_recalls:
        all_recalls = memory_core.get_recalls(context.session_id)
        if not all_recalls:
            return "目前还没有记录到任何撤回消息呢。"
        return "目前还没有记录到任何撤回消息呢。"

    lines = [f"最近 24 小时内发现了 {len(target_recalls)} 条撤回的消息："]
    for r in target_recalls:
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp))
        rendered_text = await _render_recall_content(r.text)
        lines.append(f"用户: {r.nickname} ({r.user_id})\n时间: {local_time}\n内容: {rendered_text}")
        
    return "\n".join(lines)
