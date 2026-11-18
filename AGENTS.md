## Learned User Preferences

- 始终使用简体中文回复，包括代码解释与结论总结。
- 期望对指定代码做逐段详细解释，说明每一行的作用与上下文含义，而不只是给出结论。
- 习惯用 `@文件路径:起始行-结束行` 的形式精确引用代码片段来提问。
- 提问偏向「为什么」根因分析（如「为什么不打印输出」「这个是无用的？」），回答应先给出根因再给改法。
- 倾向于让我直接动手修复并提交代码，而不是先给出方案等待确认。
- 认可并期待执行实际验证（编译、运行示例、clippy）来确认改动有效，而不只做静态推断。
- 使用 `@终端文件:行号区间` 粘贴终端输出作为上下文，期待我据此定位问题。
- 在对比不同提供方（DeepSeek、Qwen/百炼）的 API 差异时，希望明确说明兼容边界与限制。
- 经常要求给指定源文件「添加详细的代码注释」，要求中文、逐段说明每个类型/函数/分支的含义。
- 期望在文件头部补充模块级文档与 ASCII 流程图，把整体调用链与关键分支写清楚。
- 接受并期待在改动中顺带修正暴露出的相邻缺陷（如字段拼写错误），前提是不改变既有行为。

## Learned Workspace Facts

- 本仓库是名为 `zephyr` 的 Rust crate（edition 2024），基于 `async-openai` 0.43 走 OpenAI 兼容协议访问各家模型；根目录有 `rustfmt.toml`；示例程序在 `examples/`（stream_chat、structured_chat、tool_call_comlete、simple_agent_loop、agent_run、agent_run_structured、file_explorer、search_web）。
- 模型与密钥配置放在根目录 `.env`（含自定义 `OPENAI_BASE_URL` 指向 OpenAI 兼容网关）；`OPENAI_BASE_URL` 不能带结尾斜杠，否则拼出双斜杠路径返回 404。
- 模型常量集中在 `src/constant.rs`：`QWEN_MODEL`（对话）、`QWEN_VL_MODEL`（视觉）、`QWEN_EMBEDDING_MODEL`（向量）。网关的 `/models` 列表可用于校验模型名是否真实存在；百炼对不存在的模型会返回误导性的 400 `url error`。
- LLM 能力按文件拆分在 `src/llm/`：`complete.rs`、`complete_tools.rs`、`complete_tools_loop.rs`、`stream.rs`、`structured.rs`、`structured_ds.rs`、`semaphore.rs`；`complete_tools.rs` 用 `MAX_TOOL_ROUNDS = 8` 的有界循环支持多轮工具调用，每轮回灌 assistant 的 `tool_calls` 与工具结果，assistant 的 `content` 需用 `ChatCompletionRequestAssistantMessageContent` 类型、有则设无则省略。
- DeepSeek 不支持 `response_format: json_schema`，只能走 `json_object` + 把 schema 以文本注入 system prompt + 容错解析（见 `src/llm/structured_ds.rs` 的 `extract_json`）。
- 结构化输出统一用 `schemars` 生成 JSON Schema：Agent 侧注入 `final_answer` 工具并设 `tool_choice = required`（比 `response_format` 更兼容各家厂商，见 `src/agent/runtime.rs`），LLM 侧再用 `serde_json` 反序列化到强类型结构体。
- `async-openai` 在本仓库的 feature 组合下，`create_stream` 不会自动下发 `stream: true`，导致流式请求拿不到分片，需要手动构造带 `stream` 字段的请求体；示例代码中流式输出还需显式 `use futures::StreamExt;` 才能调用 `.next().await`。
- 工具统一放在 `src/tools/` 下，每个工具按 `definition.rs`（schema）、`execute.rs`（执行）、`impl.rs`（Tool trait 实现）拆分，并有一个同名 `.rs` 做模块再导出。
- `src/tools.rs` 的 `build_toolbox()` 返回 `HashMap<String, Box<dyn Tool>>`，注册 `calculator` 与 `web_search`（走 Tavily API）并连接 MCP server 拉取远端工具；`build_file_explorer_toolbox()` 提供 unzip / list / read / read_image / delete 五个文件工具。
- `src/callback/` 下 `approval.rs` 的删除审批回调是交互式读 stdin；`search_compressor.rs` 会把过长的 `web_search` 结果分块压缩后重新向量检索。
- `src/gaia/` 是 GAIA benchmark 相关模块（dataset / models / evaluatior / solver），数据集从 HuggingFace 拉取，需要 `HF_TOKEN`；`src/bin/` 下的 bin 是独立 crate，不能用 `crate::` 引用库代码，必须用 `zephyr::` 路径，且 lib 侧子模块需 `pub mod` 才会对外可见。
- `src/knowledge_base/` 分 `chunk.rs`（切分）、`embed.rs`（向量化）、`search.rs`（余弦相似度 Top-K 检索）；DashScope/Qwen 兼容网关的 embedding 接口单批最多 20 条输入，超出会报 `invalid_parameter_error: batch size is invalid`，因此 `embed_texts` 需按 20 条分批发送并按原顺序拼接结果。
