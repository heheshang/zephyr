# AGENTS.md

## Learned User Preferences

- 使用简体中文进行提问和交流，回复也应使用简体中文。
- 常要求为具体源码文件添加详细的中文代码注释，并在合适处补充模块职责说明与 ASCII 流程图。
- 添加注释类任务应保持纯注释增量，不要改动既有代码逻辑。
- 需求描述往往简短且口语化，期望参照仓库现有代码设计与分层习惯自行补全实现。
- 期望模型名、端点等易漂移配置集中在 `src/constant.rs` 等常量处管理，不要在业务代码里硬编码字符串。
- 倾向用 git tag 记录阶段性学习/迭代进度，tag 名称采用 `epNN` 或 `NN-主题` 形式。
- 接受并期望较大的结构性改动（如按上游仓库结构整体覆盖代码树），只要有明确参照来源和验证结果。
- 期望 README 持续补充架构图、代码流程图、调用流程图、脑图等可视化说明，以加深对代码的理解。
- 期望「技术演进流程」章节有理有据、清晰明了，每阶段给出对应实现细节与可核对的提交信息。
- 习惯分阶段推进并要求每轮给出可验证结论（测试通过数、clippy 零警告、实际运行验证）。
- 倾向于把过大的单文件拆成子模块目录（如 `src/a2a/card/`、`src/a2a/task_store/`），并保持 import 集中在文件顶部。

## Learned Workspace Facts

- Rust 项目，crate 名为 `zephyr`，edition 2024；`Cargo.toml` 中 `default-run = "zephyr"`，主入口 `src/main.rs`，另有 `src/bin/gaia.rs`。
- LLM 走百炼（DashScope）OpenAI 兼容网关，`.env` 中 `OPENAI_BASE_URL` 指向 `cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1`，另有 `OPENAI_API_KEY`、`HF_TOKEN`、`TAVILY_API_KEY`；该网关只接受不带厂商前缀的模型名（`qwen-plus`/`qwen-max`/`qwen-turbo`/`qwen-vl-max`/`qwen-plus-latest`，embedding 用 `text-embedding-v3`），`provider/model` 形式会返回 `Model not exist` 或 404。
- embedding 接口单批请求条目不得超过 10 条，否则报 `batch size is invalid`；常量 `EMBEDDING_BATCH_SIZE = 10` 定义在 `src/search/embedding.rs`，`embed_texts()` 内部已自动分批，调用方可传任意长度列表。
- 代码在启动时通过 `dotenv::dotenv()` 加载 `.env`，embedding 与向量检索模块内部也会自行加载。
- 目录结构：`src/tool` 为扁平工具模块、`src/llm/{ask,llm_answer,llm_request,semaphore}.rs`、`src/agent/callback/*` 与 `src/agent/runtime/*`、`src/search`（chunking/embedding/similarity/vector_search）、`src/memory`、`src/session`、`src/knowledge`（文档知识库）、`src/a2a`（A2A 协议）、`src/app`（终端应用组装层，`main.rs` 只管进程生命周期）、`src/multi_agent`、`src/planning`、`src/sandbox`。
- git remote：`origin` 为 `heheshang/zephyr`，`upstream` 为 `solenovex/rust-ai-agent`；上游以 `epNN` 形式打 tag，本地保持与其文件树逐字节对齐。
- 本仓库的里程碑 tag（`ep04`~`ep15`、`interactive-terminal`、`refactor-after-ep10`）**指向本项目自己的提交**，而非 upstream 同名 tag 的提交——本地 `main` 是重写过的孤儿历史（根提交 `21a73df`），与 upstream 的 38 个提交无共同祖先。映射关系由 tree 哈希 / diff 行数 / 特征文件三重比对确定，记录在 README「技术演进流程」的阶段总览表。
- 记忆持久化使用 SQLite（`rusqlite` bundled）。主程序用 `.zephyr/memory.db` 与 `.zephyr/knowledge.db`（启动时 `create_dir_all`，**不清空**，数据跨会话保留）；`examples/memory_demo.rs` 仍用 `memory_test.db` 并在启动时清空以保证结果可复现。记忆注入顺序为先 `MemoryInjectionCallback` 再 `ContextOptimizer`（后者按 token 阈值触发上下文压缩）。
- `src/constant.rs` 内的模型常量被下游（如 `TaskMemoryManager` 的 embedding 模型）直接引用，修改时需同步确认调用点。
- `src/knowledge/` 是完整的检索链路：SQLite 落库 + sqlite-vec `vec0` 向量索引（近似最近邻，约 O(log N)）+ FTS5 BM25 关键词召回 + RRF 融合排序；`sqlite3_auto_extension` 是进程级钩子，必须在 `Connection::open` **之前**注册，否则报 `no such module: vec0`；`vec0` 建表时固定向量维度，换 embedding 模型需重建知识库。
- FTS5 自带分词器对中文不可用（`unicode61` 把整段中文当一个 token，`trigram` 要求 ≥3 字符导致两字查询搜不到），项目自实现 **bigram** 分词（`src/knowledge/tokenizer.rs`），且 bigram 切分后必须用 **OR** 而非 AND 查询；混合检索不套用 `min_similarity` 阈值（关键词路字面命中的余弦相似度可能很低），阈值只作用于 `search_vector_only()`。
- `src/a2a/` 实现 A2A 协议的 JSON-RPC over HTTP 绑定，与 MCP 分工是「MCP 调工具、A2A 调另一个 agent」；服务端配置（`A2A_DEFAULT_HOST=127.0.0.1`、`A2A_DEFAULT_PORT=8899`、`ZEPHYR_A2A_HOST/PORT/PEERS/TOKEN`）集中在 `src/constant.rs`，一次性模式（`cargo run -- "问题"`）不启动 A2A 服务端但照常接入远端 peer。
