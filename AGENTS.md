# AGENTS.md

## Learned User Preferences

- 使用简体中文进行提问和交流，回复也应使用简体中文。
- 常要求为具体源码文件添加详细的中文代码注释，并在合适处补充模块职责说明与 ASCII 流程图。
- 添加注释类任务应保持纯注释增量，不要改动既有代码逻辑。
- 需求描述往往简短且口语化，期望参照仓库现有代码设计与分层习惯自行补全实现。
- 期望模型名、端点等易漂移配置集中在 `src/constant.rs` 等常量处管理，不要在业务代码里硬编码字符串。
- 倾向用 git tag 记录阶段性学习/迭代进度，tag 名称采用 `epNN` 或 `NN-主题` 形式。
- 接受并期望较大的结构性改动（如按上游仓库结构整体覆盖代码树），只要有明确参照来源和验证结果。

## Learned Workspace Facts

- Rust 项目，crate 名为 `zephyr`，edition 2024；`Cargo.toml` 中 `default-run = "zephyr"`，主入口 `src/main.rs`，另有 `src/bin/gaia.rs`。
- LLM 走百炼（DashScope）OpenAI 兼容网关，`.env` 中 `OPENAI_BASE_URL` 指向 `cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1`，另有 `OPENAI_API_KEY`、`HF_TOKEN`、`TAVILY_API_KEY`。
- 该网关只接受不带厂商前缀的模型名：`qwen-plus`、`qwen-max`、`qwen-turbo`、`qwen-vl-max`、`qwen-plus-latest`，embedding 用 `text-embedding-v3`；`provider/model` 形式会返回 `Model not exist` 或 404。
- embedding 接口单批请求条目不得超过 10 条，否则报 `batch size is invalid`；常量 `EMBEDDING_BATCH_SIZE = 10` 定义在 `src/search/embedding.rs`，`embed_texts()` 内部已自动分批，调用方可传任意长度列表。（早期文档写的 20 是未实测的估计值，实际会 400。）
- 代码在启动时通过 `dotenv::dotenv()` 加载 `.env`，embedding 与向量检索模块内部也会自行加载。
- 目录结构（已对齐上游 `ep11-refactored`）：`src/tool` 为扁平工具模块、`src/llm/{ask,llm_answer,llm_request,semaphore}.rs`、`src/agent/callback/{before_llm_callback,before_tool_callback,after_tool_callback}`、`src/search`（chunking/embedding/similarity/vector_search）、`src/memory`、`src/session`。
- git remote：`origin` 为 `heheshang/zephyr`，`upstream` 为 `solenovex/rust-ai-agent`；上游以 `epNN` 形式打 tag，本地保持与其文件树逐字节对齐。
- 本地 `main` 是重写过的孤儿历史（根提交 `21a73df`），与 upstream 的 38 个提交**无共同祖先**；因此 upstream 系列的 tag（`ep04`~`ep14`、`interactive-terminal`、`refactor-after-ep10`）在 origin 上通过 `upstream-history` 分支保持可达。
- 记忆持久化使用 SQLite（`rusqlite` bundled），`src/main.rs` 使用 `memory_test.db` 作为库文件并在启动时清空以保证结果可复现。
- 记忆注入顺序：先 `MemoryInjectionCallback` 再 `ContextOptimizer`，后者带 token 阈值触发上下文压缩。
- `src/constant.rs` 内的模型常量被下游（如 `TaskMemoryManager` 的 embedding 模型）直接引用，修改时需同步确认调用点。
