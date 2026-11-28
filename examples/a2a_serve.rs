//! # A2A 服务端演示（`a2a_serve`）
//!
//! 启动一个**真实的 A2A 服务端**，把本系统的 agent runtime 暴露出去，
//! 让外部 A2A 客户端能够发现并调用它。
//!
//! 与 [`a2a_demo.rs`] 的区别：
//!
//! | | `a2a_demo` | `a2a_serve`（本文件） |
//! |---|---|---|
//! | 角色 | 客户端 + 服务端都在同一进程 | **只做服务端**，等外部客户端来调 |
//! | agent | 临时造的单个 agent | **复用 main.rs 的完整 agent 树**（assistant / researcher / math / coder） |
//! | 目的 | 演示协议流程 | **验证本系统 agent runtime 在 A2A 下是否正常工作** |
//!
//! ## 为什么需要这个 demo
//!
//! `a2a_demo` 证明的是"A2A 协议本身能跑通"。但真正有价值的问题是：
//!
//! > 本系统的 agent runtime（agent 树、`transfer_to_agent`、工具调用、
//! > 上下文优化、长期记忆注入……）**放到 A2A 协议下还能正常工作吗？**
//!
//! 本文件就是回答这个问题的。它启动的 agent 树与 `src/main.rs` 一致，
//! 因此凡是经过 A2A 路径的执行，都会真实走一遍 runtime。
//!
//! ## 关键设计：agent 树 + transfer
//!
//! `src/main.rs` 的根 agent（assistant）通过 `transfer_to_agent` 把活儿
//! 交给 researcher / math / coder。A2A 客户端看到的是**一个** agent，
//! 但服务端内部可能发生了多次转移。
//!
//! A2A 协议本身不暴露内部过程，因此服务端把执行轨迹写进了
//! artifact 的 `metadata`（`execution_trace` / `transfer_history` /
//! `final_agent`），让客户端**可选地**观测到真实行为。
//!
//! ## 用法
//!
//! ```bash
//! # 启动服务端（默认监听 127.0.0.1:8899）
//! cargo run --example a2a_serve
//!
//! # 指定端口与对外地址
//! cargo run --example a2a_serve -- --port 9000 --host 0.0.0.0
//! ```
//!
//! 启动后，用任意 A2A 客户端访问：
//!
//! ```bash
//! # 查看 Agent Card
//! curl http://127.0.0.1:8899/.well-known/agent-card.json
//!
//! # 发送一条消息
//! curl -X POST http://127.0.0.1:8899/rpc \
//!   -H 'Content-Type: application/json' \
//!   -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"1+1 等于几？"}]}}}'
//! ```
//!
//! ## 信任边界
//!
//! 默认**只监听 `127.0.0.1`**，仅本机可访问。当前实现尚未接入鉴权，
//! 因此用 `--host 0.0.0.0` 对外暴露前，请确保处于可信网络环境中。

use std::{net::SocketAddr, sync::Arc};

use zephyr::{
    a2a::{card::AgentCard, server::A2AServer},
    agent::{
        callback::{
            after_tool_callback::search_compressor::SearchCompressor,
            before_llm_callback::context_optimizer::ContextOptimizer,
        },
        model::Agent,
    },
    constant::GPT_6_LUNA_PRO_MODEL,
    tool::{
        GithubStarsTool, ToolBox, build_basic_toolbox, build_toolbox,
        file::list::ListFileTool, file::read::ReadFileTool,
    },
};

/// 默认监听端口。
const DEFAULT_PORT: u16 = 8899;
/// 默认监听地址：只允许本机访问。
const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    let (host, port, token) = parse_args()?;

    // ============================================================
    // STEP 1: 装配 agent 树（与 src/main.rs 保持一致的结构）
    // ============================================================
    println!("\n========== STEP 1: 装配 agent 树 ==========\n");

    let (available_tools, mcp_enabled) = match build_toolbox().await {
        Ok(toolbox) => (toolbox, true),
        Err(error) => {
            eprintln!("warning: MCP unavailable ({error}); using local tools only");
            (build_basic_toolbox(), false)
        }
    };
    let mut available_tools = available_tools;
    // 这个工具是镜像里故意失败的演示工具，对外服务不需要。
    available_tools.remove("get_wikipedia_page");

    // 工具按专长拆分到各子 agent。
    let mut math_tools = ToolBox::new();
    if let Some(calculator) = available_tools.remove("calculator") {
        math_tools.insert("calculator".to_owned(), calculator);
    }

    let mut researcher_tools = ToolBox::new();
    if let Some(web_search) = available_tools.remove("web_search") {
        researcher_tools.insert("web_search".to_owned(), web_search);
    }
    researcher_tools.insert("github_stars".to_owned(), Box::new(GithubStarsTool));

    let mut coder_tools = ToolBox::new();
    coder_tools.insert("read_file".to_owned(), Box::new(ReadFileTool));
    coder_tools.insert("list_file".to_owned(), Box::new(ListFileTool));

    let optimizer = Arc::new(ContextOptimizer::new(GPT_6_LUNA_PRO_MODEL));
    let search_compressor = Arc::new(SearchCompressor);

    // researcher：联网检索专家
    let researcher = Arc::new(
        make_agent(
            "researcher",
            "联网检索与 GitHub 数据专家",
            RESEARCHER_INSTRUCTIONS,
            researcher_tools,
            12,
        )
        .with_before_llm_callbacks(optimizer.clone())
        .with_after_tool_callbacks(search_compressor.clone()),
    );

    // math：数学计算专家
    let math = Arc::new(
        make_agent(
            "math",
            "数学计算专家",
            MATH_INSTRUCTIONS,
            math_tools,
            6,
        )
        .with_before_llm_callbacks(optimizer.clone())
        .with_after_tool_callbacks(search_compressor.clone()),
    );

    // coder：代码与沙箱专家
    let coder = Arc::new(
        make_agent(
            "coder",
            "代码执行与沙箱专家",
            CODER_INSTRUCTIONS,
            coder_tools,
            12,
        )
        .with_before_llm_callbacks(optimizer.clone())
        .with_after_tool_callbacks(search_compressor.clone()),
    );

    // assistant：根节点，负责分派
    let root = Arc::new(
        make_agent(
            "assistant",
            "通用助手，负责理解任务并分派专家",
            ROOT_INSTRUCTIONS,
            available_tools,
            24,
        )
        .with_before_llm_callbacks(optimizer)
        .with_after_tool_callbacks(search_compressor),
    );

    // 组装父子关系：这一步之后 transfer_to_agent 才会生效。
    root.set_sub_agents(vec![researcher.clone(), math.clone(), coder.clone()])?;

    println!("agent 树已装配：");
    println!("  assistant (root, 24 步)");
    println!("  ├── researcher  联网检索 + github_stars (12 步)");
    println!("  ├── math        calculator (6 步)");
    println!("  └── coder       沙箱 / 文件 (12 步)");
    println!("MCP 启用: {mcp_enabled}");

    // ============================================================
    // STEP 2: 生成 Agent Card
    // ============================================================
    println!("\n========== STEP 2: 生成 Agent Card ==========\n");

    let bind_addr = format!("{host}:{port}");
    // Agent Card 里的 url 必须是**客户端可直接调用**的 JSON-RPC 端点。
    // 对外服务时用 host 原样，不能写 127.0.0.1——否则远端客户端
    // 会去访问自己机器上的 127.0.0.1。
    let advertised_host = if host == "0.0.0.0" {
        "127.0.0.1"
    } else {
        host.as_str()
    };
    let rpc_url = format!("http://{advertised_host}:{port}/rpc");

    // Agent Card 由 agent 树自动生成：
    // - 子 agent（researcher / math / coder）→ 委派 skill，tag 含 "delegation"
    // - 工具 → 每条工具一条能力 skill
    //
    // 这样远端客户端能直接从卡片判断"该派什么活儿过来"。
    let card = AgentCard::from_agent(&root, &rpc_url, env!("CARGO_PKG_VERSION"))
        .with_streaming(true);

    // 鉴权：`--token <key>` 开启。开启后 `/rpc` 需要
    // `Authorization: Bearer <key>`，Agent Card 仍然公开。
    let card = if token.is_empty() {
        card
    } else {
        card.with_bearer_auth()
    };

    let card_json = serde_json::to_string_pretty(&card)?;
    println!("{card_json}");

    // ============================================================
    // STEP 3: 启动 A2A 服务端
    // ============================================================
    println!("\n========== STEP 3: 启动 A2A 服务端 ==========\n");

    let mut state = zephyr::a2a::server::A2AServerState::new(
        root.clone(),
        card.clone(),
        Arc::new(zephyr::a2a::task_store::InMemoryTaskStore::new()),
        root.session_manager.clone(),
    );

    if !token.is_empty() {
        state = state.with_authenticator(Arc::new(
            zephyr::a2a::auth::BearerTokenAuthenticator::new(token.clone()),
        ));
    }

    let server = A2AServer::new(Arc::new(state));

    let addr: SocketAddr = bind_addr.parse()?;
    println!("Agent Card : http://{bind_addr}{}", zephyr::a2a::types::WELL_KNOWN_AGENT_CARD_PATH);
    println!("JSON-RPC   : {rpc_url}");
    println!("监听地址   : {addr}");
    if token.is_empty() {
        println!("鉴权       : 未启用");
    } else {
        println!("鉴权       : 已启用（Bearer Token）");
    }
    println!("\n按 Ctrl+C 停止服务。\n");

    // 服务端会一直运行，直到进程被终止。
    server.serve(addr).await?;

    Ok(())
}

/// 解析命令行参数。
///
/// 支持 `--port <n>`、`--host <addr>` 与 `--token <key>`。
fn parse_args() -> anyhow::Result<(String, u16, String)> {
    let mut host = DEFAULT_HOST.to_owned();
    let mut port = DEFAULT_PORT;
    let mut token = String::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--port 需要一个端口号"))?;
                port = value.parse()?;
            }
            "--host" => {
                host = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--host 需要一个地址"))?;
            }
            "--token" => {
                token = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--token 需要一个密钥"))?;
            }
            "--help" | "-h" => {
                println!("用法: cargo run --example a2a_serve -- [--host <addr>] [--port <n>] [--token <key>]");
                println!("  --host  监听地址，默认 {DEFAULT_HOST}（仅本机）");
                println!("  --port  监听端口，默认 {DEFAULT_PORT}");
                std::process::exit(0);
            }
            other => anyhow::bail!("未知参数: {other}"),
        }
    }

    Ok((host, port, token))
}

/// 创建一个 agent，并挂上本系统的标准回调。
fn make_agent(
    name: &str,
    description: &str,
    instructions: &str,
    toolbox: ToolBox,
    max_steps: usize,
) -> Agent {
    Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec![instructions.to_owned()],
        Arc::new(toolbox),
    )
    .with_name(name.to_owned())
    .with_description(description.to_owned())
    .with_max_steps(max_steps)
}

const ROOT_INSTRUCTIONS: &str = "\
你是运行在终端里的通用 AI 助手。请用中文回答，回答使用 Markdown。

你的团队:
- researcher: 联网搜索、GitHub 数据、最新资料
- math: 数学计算
- coder: 沙箱内 Python / Bash / 文件执行，以及技能

工作规则:
1. 需要最新网络资料时，调用 transfer_to_agent(\"researcher\")。
2. 需要精确数学计算时，调用 transfer_to_agent(\"math\")。
3. 需要执行代码、使用沙箱或技能时，调用 transfer_to_agent(\"coder\")。
4. 文件、规划、反思、记忆、MCP 工具可以直接使用。
5. 能自己完成的任务不要转移；转移后由专家直接回答用户。
6. 不确定时先查证或调用工具，不要编造事实和链接。";

const RESEARCHER_INSTRUCTIONS: &str = "\
你是资料检索专家。请用中文回答。
1. 使用 web_search 查找最新资料，使用 github_stars 查询精确的 GitHub Star 数。
2. 只使用搜索结果中出现过的链接，不要编造来源。
3. 输出简洁的要点列表，并标注来源 URL。
4. 如果问题还包含数学或代码部分，转交给 math 或 coder。";

const MATH_INSTRUCTIONS: &str = "\
你是数学专家。请用中文回答。
1. 使用 calculator 工具计算，不要心算。
2. 展示简要计算步骤，最终答案要明确。
3. 不回答与数学无关的内容。";

const CODER_INSTRUCTIONS: &str = "\
你是代码与沙箱专家。请用中文回答。
1. 使用 execute_python 或 bash_tool 在沙箱中运行代码。
2. 需要读取文件时使用 read_file / list_file；需要上传文件时使用 upload_file。
3. 代码出错时先分析错误，再修正并重新运行。
4. 返回可运行的代码和真实执行结果，不要编造输出。";
