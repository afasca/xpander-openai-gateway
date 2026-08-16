# xpander OpenAI 兼容网关

把 **chat.xpander.ai**（网页端 AI 对话）封装成 **OpenAI 兼容 API**（FastAPI）。
协议均来自真实浏览器抓包，非猜测构造。

## 功能

- `POST /v1/chat/completions`：支持 `stream=true`（SSE 流式）与 `stream=false`
  - 正文 → `content`；工具调用 / 推理步骤 → `reasoning_content`（流式与非流式都带）
- `GET /v1/models`：模型列表**实时抓自上游** `rest/v1/model_prices`（is_active=true，400+ 个），缓存 3600s，WebUI 可手动刷新
- 多账号 Cookie 轮询：线程安全账号池、失败自动标记/临时剔除（10 分钟）、自动换号
- **token 自动续期**：Cookie 内 access_token 1 小时过期，网关用 refresh_token 自动刷新并把新 refresh_token 持久化回 `.env`（refresh_token 为一次性轮换，务必让程序可写 `.env`）
- **阅后即焚**：回答完成后自动 `DELETE` 上游会话，官网会话列表不保留聊天记录（工具开关里可关）
- SSE 心跳：上游长考时每 25s 发 `: ping` 注释帧，防代理/客户端空闲断连；流式兜底：上游偶发不吐 chunk 时用 `task_finished.result` 补发正文
- WebUI 中文管理后台（`/` 路径，免登录直接访问）：
  - 概览 / 账号池（批量添加、勾选删除、单账号启停、单账号免费体检）
  - 工具开关（深度思考 think_mode=harder、深度规划 deep_planning，默认全关，状态持久化）
  - 在线对话（流式渲染、`reasoning_content` 折叠展示、**阅后即焚**：仅存浏览器内存，刷新即清空）
  - 调用 Key 管理 / 请求日志 / Cookie 获取教程
- 请求日志：时间、账号（脱敏）、模型、输入/输出/推理、TTFT（首字延迟）、总耗时、成败与错误原因；
  内存保留 200 条 + 全量追加写入 `logs/requests.jsonl`

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，粘贴至少一个账号 Cookie（也可启动后在 WebUI「账号池管理」粘贴）
python app.py
```

- WebUI： <http://127.0.0.1:8300/> （免登录）
- 调用示例：

```bash
curl http://127.0.0.1:8300/v1/chat/completions \
  -H "Authorization: Bearer 123456" \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet-5","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

## 鉴权方式（三种任一）

网关接受以下任一凭证（默认 key `123456`，WebUI 可增删）：

1. `Authorization: Bearer <key>`（标准 OpenAI 方式）
2. `X-Api-Key: <key>` 头
3. URL 查询参数 `?key=<key>`

> 注意：某些反向代理会改写 `Authorization` 头——经过此类代理时请用 `X-Api-Key` 或 `?key=`。
> WebUI 管理后台（`/`）与 `/api/*` 免登录；仅 `/v1/*` 需要 key。

## systemd 部署（开机自启）

```ini
# /etc/systemd/system/xpander-gateway.service
[Unit]
Description=xpander OpenAI Gateway
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/xpander-openai-gateway
ExecStart=/usr/bin/python3 /opt/xpander-openai-gateway/app.py
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo mkdir -p /opt/xpander-openai-gateway
sudo cp -r app.py requirements.txt .env /opt/xpander-openai-gateway/
sudo chown -R ubuntu:ubuntu /opt/xpander-openai-gateway   # 程序需可写 .env（token 轮换持久化）
pip3 install -r /opt/xpander-openai-gateway/requirements.txt
sudo systemctl daemon-reload
sudo systemctl enable --now xpander-gateway
sudo systemctl status xpander-gateway
```

## 上游协议说明（抓包结论）

| 项目 | 端点 / 说明 |
|---|---|
| 鉴权 | Supabase Cookie `sb-svc-sb-auth-token.*`（base64 JSON：access_token + refresh_token） |
| 刷新 | `POST https://svc-sb.app.xpander.ai/auth/v1/token?grant_type=refresh_token`（Header: apikey），refresh_token 一次性轮换 |
| 发消息 | `POST https://chat-backend.xpander.ai/{agent_id}/invoke`，`id=null` 自动建会话 |
| SSE 事件 | `connected` / `task_created` / `chunk`（正文增量→content）/ `tool_call_request`（含 reasoning.title→reasoning_content）/ `tool_call_result` / `inline_card` / `context_status` / `task_finished`（result+tokens） |
| 余额/体检 | `POST /functions/v1/credits-overview` → `balance_credits`、`usd_equivalent`、`license.tier/active`、`free_grant_claimed_at` |
| 模型列表 | `GET /rest/v1/model_prices?select=provider,model_id&is_active=eq.true` |
| 官方 API key | 网页 Settings→API keys 创建的 key 仅适用于官方 REST（api.xpander.ai/v1），**不能**调 chat-backend，本网关不使用它 |

注：需求模板中的 `GET /organizations/{id}`（balanceCents）在本站不存在，对应物为上表
`credits-overview` 的 `balance_credits` / `usd_equivalent`（语义一致，WebUI 体检展示）。

## 提示词透传规则

- 单条 user 消息：**原样透传**，不加任何角色前缀；
- 多轮历史：压缩为 `System:` / `User:` / `Assistant:` 标签拼接的单条 prompt（system 用强指令包装置首）；
- 每次调用创建新会话（`id=null`），回答完自动删除（阅后即焚），官网会话列表不保留。

## 顶替 Omni 内置提示词（纯净模型通道）

Omni 智能体自带约 **5.8 万 token** 的服务端系统提示词，会锁定"我是 Omni"的人设，system 指令无法覆盖。
解决办法：在 xpander 官网（或官方 API）创建一个**无指令的自定义 Agent**，然后在 WebUI「概览 → 上游智能体」切换过去：

- 内置提示词降到约 **2.7 万 token**（残余为 xpander agent 框架自身的执行提示词，无法去除）
- `system` 角色指令完全生效（可自定义人设）
- 代价：自定义 Agent 默认**没有工具**（联网搜索等），需要工具时切回 Omni

## 常见问题

- **503 无可用账号**：账号池为空或全部停用/剔除 → WebUI「账号池管理」添加或启用。
- **401 上游鉴权失败**：Cookie 失效（Google 侧登出/改密）→ 重新获取粘贴。
- **token 自动续期不生效**：确认程序对 `.env` 有写权限（refresh_token 轮换必须落盘）。
