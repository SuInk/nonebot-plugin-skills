from typing import Optional
from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext

travel_schema = {
    "properties": {
        "destination": {"type": "STRING", "description": "旅行目的地城市或地区名称，例如 北京"},
        "days": {"type": "INTEGER", "description": "旅行天数 (可选)"},
        "nights": {"type": "INTEGER", "description": "旅行晚数 (可选)"}
    },
    "required": ["destination"]
}

@skill_manager.register("travel_plan", "根据目的地生成详细的旅行规划。如果用户没有提供天数，请根据该城市的常规游玩时长自动推荐一个方案。", travel_schema)
async def travel_plan(destination: str, days: Optional[int] = None, nights: Optional[int] = None, context: Optional[SkillContext] = None) -> str:
    if not destination:
        return "请告诉我目的地，例如：北京"
    
    summary = f"旅行规划目的地: {destination}"
    if days is not None:
        summary += f"，时长: {days}天"
        if nights is not None:
            summary += f"{nights}晚"
    else:
        summary += " (天数未指定，请根据目的地自动推荐最合适的游玩时长并生成规划)"
        
    # 直接引导 AI 生成最终内容，不再进行多余的确认。
    return f"你现在是旅行规划专家嘉然，请直接为用户生成一份详细、可落地的旅行建议。\n{summary}\n要求：结构清晰，适合QQ聊天阅读，纯文本输出。"
