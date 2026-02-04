---
name: nonebot-plugin-skills
description: Agent instructions for this repo
---

# Agent Rules

## 必须遵守
- 所有新增或修改的功能，必须先经过自然语言处理（NLP）得到结构化 JSON，先由 JSON 明确 action/target/params，再进入业务逻辑处理。
- JSON解析完意图后，可直接在业务逻辑中根据原始文本执行具体操作；
- 发送给 QQ 的输出必须为纯文本，不使用 Markdown 格式。

## JSON 规范（建议）
输出必须是单一 JSON 对象，字段如下：
```
{
  "action": "string",          // 动作类型，例如: chat, image, weather, ignore
  "target": "string",          // 目标类型或资源，例如: message_image, city, user
  "params": { ... }            // 结构化参数，可选
}
```

## 处理流程（必须）
1. 先做 NLP 解析：将用户自然语言转为 JSON。
2. 校验 JSON 合法性与字段完整性。
3. 根据 `action` 调度对应工具或处理器。
4. 若 JSON 不完整或不可信，优先追问澄清，而不是直接执行。

## 适用范围
- 命令触发（如“天气 北京”）也必须通过 NLP 生成 JSON，再进入工具调用。
- 后续新增能力（图片、查询、分析、生成等）都必须遵守本规则。
