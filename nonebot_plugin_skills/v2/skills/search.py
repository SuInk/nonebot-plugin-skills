import asyncio
from typing import Optional
from google import genai
from google.genai import types
from nonebot import logger

from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext
from nonebot_plugin_skills.config import config

search_schema = {
    "properties": {
        "query": {"type": "STRING", "description": "要在互联网上查询的关键词或具体问题"}
    },
    "required": ["query"]
}

def _get_client() -> Optional[genai.Client]:
    if not config.google_api_key: return None
    return genai.Client(api_key=config.google_api_key)

@skill_manager.register("google_search", "在互联网上进行深度搜索。当你遇到不懂的知识、最新的项目或不确定的事实时，请使用此工具。", search_schema)
async def google_search(query: str, context: Optional[SkillContext] = None) -> str:
    """
    通过启动一个独立的 Gemini 请求（开启内置 Google Search）来实现真正的联网搜索。
    """
    client = _get_client()
    if not client: return "未配置 API Key，搜索功能不可用。"

    logger.info(f"嘉然正在向 Google 智囊团求助搜索: {query}")

    try:
        # 启动一个独立的请求，专门用于搜索
        # 这个请求不带任何自定义 Tools，只带内置的 Google Search，从而避开 API 限制
        search_config = types.GenerateContentConfig(
            system_instruction="你是一个专业的搜索研究助手。请利用 Google Search 查阅资料，并针对用户的问题给出一份准确、详尽、客观的总结报告。请直接输出总结内容，不要解释搜索过程。",
            tools=[{"google_search": {}}], # 仅开启内置搜索
            temperature=0.2
        )

        response = await client.aio.models.generate_content(
            model=config.gemini_text_model,
            contents=query,
            config=search_config
        )

        # 提取搜索结果报告
        research_result = response.text or "搜索结果为空。"
        
        # 返回给嘉然，让她进行最后的语气转换
        return f"【搜索研究报告已送达】：\n{research_result}"

    except Exception as e:
        logger.error(f"Deep search failed: {e}")
        return f"呜呜，嘉然刚才去 Google 查资料的时候迷路了（报错: {e}）。"
