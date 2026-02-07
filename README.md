# nonebot-plugin-skills

基于 Google Gemini 的头像/图片处理、聊天与网页总结插件，内置上下文缓存、群/私聊隔离，并支持从聊天记录中自动获取最近头像/图片。

## 功能
- 处理头像/图片：命令内带图、@某人头像、或使用最近聊天图片
- 聊天对话：带上下文的自然语言聊天
- 天气查询：输入城市/地区即可查询当前天气
- 网页总结：抓取并总结主流网站网页正文（支持 `github.com`、`v2ex.com`、`linux.do`、`news.ycombinator.com`、`bilibili.com`、`zhihu.com`、`x.com`、`twitter.com`）
- 上下文缓存：按群/私聊隔离，定时过期
- 图片缓存：默认缓存最近 10 张图片（按会话隔离），避免图片链接过期
- 动图/表情包：支持识别与对话（如 GIF、商城表情等）
- 历史记录压缩：超出阈值自动摘要保留关键信息

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

```env
# 必填
GOOGLE_API_KEY=你的GoogleAPIKey

# 模型（可按需替换为你账号可用的模型名）
GEMINI_TEXT_MODEL=gemini-3-pro-preview
GEMINI_IMAGE_MODEL=nano-banana-pro-preview

# 超时（秒）
REQUEST_TIMEOUT=30.0
IMAGE_TIMEOUT=120.0
IMAGE_DOWNLOAD_MAX_BYTES=15728640     # 单张下载图片大小上限（字节，默认15MB）
WEB_FETCH_MAX_BYTES=2097152            # 网页抓取大小上限（字节，默认2MB）
WEB_EXTRACT_MAX_CHARS=12000            # 提取正文后送入总结的最大字符数

# 会话历史
HISTORY_TTL_SEC=600                  # 会话状态保留时长
HISTORY_MAX_MESSAGES=10              # 最大历史条数
IMAGE_CACHE_MAX_IMAGES=10            # 图片缓存：最多缓存最近 N 张图片（用于防止图片链接过期）
HISTORY_COMPRESS_ENABLE=true         # 是否启用历史压缩摘要
HISTORY_COMPRESS_TRIGGER=20          # 触发压缩的历史条数阈值
HISTORY_COMPRESS_KEEP=6              # 压缩后保留最近 N 条原始消息
HISTORY_COMPRESS_MIN_MESSAGES=6      # 压缩最少需要的非摘要消息条数
HISTORY_COMPRESS_MAX_CHARS=600       # 摘要最大字数
HISTORY_REFERENCE_ONLY=true          # 仅把历史作为“参考文本”，避免模型继续旧话题

# 发送策略
FORWARD_CHAR_THRESHOLD=100           # 单次回复超过该字数，改用合并转发（无延迟）
FORWARD_LINE_THRESHOLD=8             # 行数超过该阈值，改用合并转发（无延迟）
MESSAGE_SEND_DELAY_SEC=0.6           # 分段发送（按空行分段）时段落间延迟

# NLP 触发
NLP_ENABLE=true
BOT_KEYWORDS=["Diana","diana","嘉然","然然"]  # 群聊中：命中关键词才触发（@/回复机器人也会触发）

# 意图识别上下文（可选）
NLP_CONTEXT_HISTORY_MESSAGES=2       # 取多少条历史拼进意图识别输入
NLP_CONTEXT_FUTURE_MESSAGES=2        # 取多少条“后续消息”（短暂等待后收集）
NLP_CONTEXT_FUTURE_WAIT_SEC=1.0      # 等待多少秒后再收集后续消息

# 调试
GEMINI_LOG_RESPONSE=false
```

## 使用
### 指令
| 指令 | 说明 |
| --- | --- |
| 处理头像 <指令> | 处理头像/最近图片/@用户头像 |
| 聊天 <内容> | 上下文聊天 |
| 技能 <内容> | 上下文聊天 |
| 天气 <城市> | 查询当前天气 |
| 网页总结 <网页链接> | 总结主流网站网页正文 |

### 示例
- `Diana帮忙把@向晚头像变成黑白`
- `处理头像 变成赛博朋克风`
- `处理头像 @小明 变成油画风`
- `聊天 你还记得刚才的头像吗？`
- `天气 上海`
- `网页总结 https://github.com/owner/repo`
- `网页总结 https://linux.do/t/topic/12345 重点看结论和待办`
- `网页总结 https://www.zhihu.com/question/123456789`

> 若图片模型仅返回文本结果，插件会直接把文本回复出来（便于你确认模型是否支持图像输出）。
