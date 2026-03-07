from typing import Optional
from nonebot import get_plugin_config
from pydantic import BaseModel, Field
import httpx

from nonebot_plugin_skills.v2.core.skills import skill_manager

# --- Skill Configuration ---
# 技能自己的配置，独立管理，不污染全局 config
class WeatherConfig(BaseModel):
    weather_api_url: str = "https://v0.yiketianqi.com/api"
    weather_app_id: str = "41249764"
    weather_app_secret: str = "r2uUAnzY"
    weather_request_timeout: float = 10.0

    class Config:
        extra = "allow"

try:
    config = get_plugin_config(WeatherConfig)
except Exception:
    # 允许在测试/未绑定 Nonebot 时使用默认配置
    config = WeatherConfig()


# --- Skill Registration ---
weather_schema = {
    "properties": {
        "city": {"type": "STRING", "description": "要查询天气的城市名称，例如 北京"},
        "days": {"type": "INTEGER", "description": "查询天数，默认为 1"}
    },
    "required": ["city"]
}

@skill_manager.register("get_weather", "获取指定城市的天气信息", weather_schema)
async def get_weather(city: str, days: int = 1) -> str:
    """真正的技能执行逻辑"""
    try:
        async with httpx.AsyncClient(timeout=config.weather_request_timeout) as client:
            resp = await client.get(
                config.weather_api_url,
                params={
                    "version": "v61",
                    "appid": config.weather_app_id,
                    "appsecret": config.weather_app_secret,
                    "city": city,
                }
            )
            data = resp.json()
            if "city" in data:
                return f"{data['city']} 当前天气: {data.get('wea')}，气温: {data.get('tem')}度，风向: {data.get('win')}，风速: {data.get('win_meter')}"
            else:
                return f"未能获取 {city} 的天气数据。"
    except Exception as e:
        return f"查询天气时发生错误: {str(e)}"
