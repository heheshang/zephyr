## Learned User Preferences

- 始终使用简体中文回复，包括代码解释与结论总结。
- 期望对指定代码做逐段详细解释，说明每一行的作用与上下文含义，而不只是给出结论。
- 习惯用 `@文件路径:起始行-结束行` 的形式精确引用代码片段来提问。
- 提问偏向「为什么」根因分析（如「为什么不打印输出」「这个是无用的？」），回答应先给出根因再给改法。
- 倾向于让我直接动手修复并提交代码，而不是先给出方案等待确认。
- 认可并期待执行实际验证（编译、运行示例、clippy）来确认改动有效，而不只做静态推断。
- 使用 `@终端文件:行号区间` 粘贴终端输出作为上下文，期待我据此定位问题。
- 在对比不同提供方（DeepSeek、Qwen/百炼）的 API 差异时，希望明确说明兼容边界与限制。

## Learned Workspace Facts

- 本仓库是名为 `zephyr` 的 Rust crate（edition 2024），基于 `async-openai` 0.43 走 OpenAI 兼容协议访问各家模型。
- 模型与密钥配置放在根目录 `.env`（含自定义 `OPENAI_BASE_URL` 指向 OpenAI 兼容网关）；`OPENAI_BASE_URL` 不能带结尾斜杠，否则拼出双斜杠路径返回 404。
- 默认模型名集中在 `src/constant.rs` 的 `MODEL` 常量中（如 `qwen-plus`）。
- LLM 能力按文件拆分在 `src/llm/`：`complete.rs`、`complete_tools.rs`、`stream.rs`、`structured.rs`、`structured_ds.rs`、`semaphore.rs`。
- 示例程序在 `examples/`：`stream_chat`、`structured_chat`、`gaia`、`tool_call_comlete`。
- 工具（tools）统一放在 `src/tools/` 下，每个工具按 `definition.rs`（schema）、`execute.rs`（执行）、`impl.rs`（Tool trait 实现）拆分，并有一个同名 `.rs` 做模块再导出。
- 工具注册中心在 `src/tools.rs` 的 `build_toolbox()`，返回 `HashMap<String, Box<dyn Tool>>`；当前已注册 `calculator` 与 `web_search`（web_search 走 Tavily API）。
- `src/gaia/` 是 GAIA benchmark 相关模块（dataset / models / evaluatior / solver），数据集从 HuggingFace 拉取，需要 `HF_TOKEN`。
- `src/bin/` 下的 bin 是独立 crate，不能用 `crate::` 引用库代码，必须用 `zephyr::` 路径；同时 lib 侧的子模块需 `pub mod` 才会对外可见。
- DeepSeek 不支持 `response_format: json_schema`，只能走 `json_object` + 把 schema 以文本注入 system prompt + 容错解析（见 `src/llm/structured_ds.rs` 的 `extract_json`）。
- `async-openai` 在本仓库的 feature 组合下，`create_stream` 不会自动下发 `stream: true`，导致流式请求拿不到分片，需要手动构造带 `stream` 字段的请求体。
- 结构化输出使用 `schemars` 生成 JSON Schema，再由 `serde_json` 反序列化到强类型结构体。
- 示例代码中流式输出需要显式 `use futures::StreamExt;` 才能对 `impl Stream` 调用 `.next().await`。
- 仓库根目录新增了 `rustfmt.toml` 统一格式化配置。
