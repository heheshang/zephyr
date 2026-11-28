//! 帮助文本。
//!
//! 单独成文件，让 `/help` 的内容与代码逻辑分开——
//! 帮助文本是给用户看的文档，不该埋在命令分发的 match 里。

/// `/help` 与 `zephyr --help` 展示的内容。
pub const HELP: &str = "\
# 交互命令

- `/help` 显示帮助
- `/new` 开始一个新会话
- `/status` 查看模型、会话和功能状态
- `/agents` 查看 agent 树和转移目标
- `/tools` 查看每个 agent 的工具
- `/history [n]` 查看最近 n 条事件，默认 20
- `/remember` 把当前会话提取为长期记忆
- `/knowledge` 列出知识库中的文档
- `/a2a` 查看 A2A 服务端与已接入的远端 agent
- `/clear` 清屏并重画横幅
- `/quit` 退出

直接输入问题即可。输入以 `\\` 结尾时，可以继续下一行。
也可以一次性执行：`cargo run -- 你的问题`。

## 能力

- `transfer_to_agent`：把控制权交给 researcher / math / coder
- 工具审批：危险工具执行前会暂停并询问
- 长期记忆：`/remember` 保存，之后自动检索
- 文档知识库：`add_knowledge` 入库，`search_knowledge` 检索
- A2A：启动时对外暴露本助手，并把远端 A2A agent 接成工具
- Markdown：回答按终端 Markdown 渲染

## A2A 环境变量

- `ZEPHYR_A2A_HOST`：服务端监听地址，默认 `127.0.0.1`（仅本机）
- `ZEPHYR_A2A_PORT`：服务端监听端口，默认 `8899`
- `ZEPHYR_A2A_PEERS`：逗号分隔的远端 A2A agent 地址
- `ZEPHYR_A2A_TOKEN`：服务端要求的 Bearer Token；设置后 `/rpc` 需要鉴权

一次性模式（`cargo run -- 问题`）不会启动 A2A 服务端——进程退出即结束，
服务端来不及被调用。但仍会接入远端 peer，以便本次回答能跨进程委派。
";
