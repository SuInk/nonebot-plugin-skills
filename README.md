# nonebot-plugin-skills

基于 Google Gemini 的头像/图片处理与聊天插件，内置上下文缓存、群/私聊隔离，并支持从聊天记录中自动获取最近头像/图片。

## 功能
- 处理头像/图片：命令内带图、@某人头像、或使用最近聊天图片
- 聊天对话：带上下文的自然语言聊天
- 上下文缓存：按群/私聊隔离，定时过期
- Skills：内置 `SKILLS.md`，可自行编辑

## 安装
在 NoneBot2 项目中安装依赖：

```bash
pip install nonebot2 nonebot-adapter-onebot httpx google-genai
```

将插件加入 `pyproject.toml`：

```toml
[tool.nonebot]
plugins = ["nonebot_plugin_skills"]
```

> 当前仓库包目录已改为 `nonebot_plugin_skills`，可直接作为可导入插件使用。

## 配置
在 `.env` 中配置：

```
GOOGLE_API_KEY=你的GoogleAPIKey
GEMINI_TEXT_MODEL=gemini-2.5-flash
GEMINI_IMAGE_MODEL=gemini-2.5-flash-image
HISTORY_TTL_SEC=600
HISTORY_MAX_MESSAGES=20
GEMINI_LOG_RESPONSE=false
IMAGE_TIMEOUT=120
NLP_ENABLE=true
BOT_KEYWORDS=["Diana","diana","嘉然"]
```

## 使用
### 指令
| 指令 | 说明 |
| --- | --- |
| 处理头像 <指令> | 处理头像/最近图片/@用户头像 |
| 聊天 <内容> | 上下文聊天 |
| 技能 <内容> | 上下文聊天 |

### 示例
- `Diana帮忙把@向晚头像变成黑白`
- `处理头像 变成赛博朋克风`
- `处理头像 @小明 变成油画风`
- `聊天 你还记得刚才的头像吗？`

> 若图片模型仅返回文本结果，插件会直接把文本回复出来（便于你确认模型是否支持图像输出）。

## 自定义 Skills
编辑 `nonebot_plugin_skills/SKILLS.md` 中的 `[chat]` 与 `[image]` 段落即可。
