//! 组装终端应用：工具箱、agent 树、可选能力。
//!
//! # 组装顺序（顺序不可随意调换）
//!
//! ```text
//!   1. 工具箱         ── MCP 连接失败则回退到本地工具
//!   2. 可选能力       ── 记忆库 / 知识库（失败只 warning）
//   3. 工具再分配     ── calculator → math，web_search → researcher
//!   4. A2A peer 发现  ── 必须早于 agent 构造（工具箱建好就固定了）
//!   5. agent 树       ── researcher / math / coder → assistant
//!   6. A2A 服务端     ── 一次性模式跳过
//! ```
//!
//! # 失败策略
//!
//! 除了"工具箱完全不可用"之外，所有可选能力的初始化失败都只打印 warning。
//! 一个 agent 框架不该因为记忆库建不起来就整个不能用。

use std::sync::Arc;

use crate::{
    agent::{
        callback::{
            after_tool_callback::search_compressor::SearchCompressor,
            before_llm_callback::{
                context_optimizer::ContextOptimizer, memory_injection::MemoryInjectionCallback,
            },
        },
        context::ExecutionContext,
        model::Agent,
    },
    app::{
        a2a::{A2AConfig, attach_peers, discover_peers, start_server},
        prompts::{
            CODER_INSTRUCTIONS, MATH_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS, ROOT_INSTRUCTIONS,
        },
        render::styled,
        App, Features,
    },
    constant::{GPT_6_LUNA_PRO_MODEL, TEXT_EMBEDDING_3_SMALL_MODEL},
    knowledge::knowledge_base::KnowledgeBaseManager,
    memory::memory_manager::TaskMemoryManager,
    tool::{
        GithubStarsTool, ToolBox, build_basic_toolbox, build_toolbox, code::register_code_tools,
        file::list::ListFileTool, file::read::ReadFileTool,
        knowledge_tool::{AddKnowledgeTool, SearchKnowledgeTool},
        memory_tool::RecallMemoryTool,
    },
};

use uuid::Uuid;

/// 记忆库文件路径。
const MEMORY_DB: &str = ".zephyr/memory.db";
/// 知识库文件路径。
const KNOWLEDGE_DB: &str = ".zephyr/knowledge.db";

/// 组装应用。
///
/// `one_shot` 表示这是一次性模式（`cargo run -- "问题"`）：
/// 此时不启动 A2A 服务端——进程马上就退出，服务端没有存在的意义。
pub async fn build_app(one_shot: bool) -> anyhow::Result<App> {
    let model = GPT_6_LUNA_PRO_MODEL.to_owned();

    let (mut available_tools, mcp) = match build_toolbox().await {
        Ok(toolbox) => (toolbox, true),
        Err(err) => {
            eprintln!("warning: MCP unavailable ({err}); using local tools only");
            (build_basic_toolbox(), false)
        }
    };
    // 这个工具是镜像里故意失败的演示工具，交互模式不需要。
    available_tools.remove("get_wikipedia_page");

    let memory = match init_memory(&model) {
        Ok(manager) => Some(Arc::new(manager)),
        Err(err) => {
            eprintln!("warning: long-term memory disabled: {err}");
            None
        }
    };

    // 知识库是可选能力：初始化失败时只打印 warning，不影响对话。
    let knowledge = match init_knowledge() {
        Ok(manager) => Some(Arc::new(manager)),
        Err(err) => {
            eprintln!("warning: knowledge base disabled: {err}");
            None
        }
    };

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
    let sandbox_prompt = match register_code_tools(&mut coder_tools).await {
        Ok(prompt) => prompt,
        Err(err) => {
            eprintln!("warning: code execution disabled: {err}");
            String::new()
        }
    };
    coder_tools.insert("read_file".to_owned(), Box::new(ReadFileTool));
    coder_tools.insert("list_file".to_owned(), Box::new(ListFileTool));

    let code = coder_tools.contains_key("execute_python") || coder_tools.contains_key("bash_tool");
    let skills = sandbox_prompt.contains("Available Skills");

    for toolbox in [
        &mut available_tools,
        &mut researcher_tools,
        &mut math_tools,
        &mut coder_tools,
    ] {
        add_memory_tool(toolbox, &memory);
    }

    // 知识库检索给到根助手和 researcher（负责查资料的专家）；
    // 入库工具只给根助手，由它统一管理资料，避免各 Agent 重复写入。
    add_knowledge_tools(&mut available_tools, &knowledge);
    add_knowledge_tools(&mut researcher_tools, &knowledge);
    if let Some(manager) = &knowledge {
        available_tools.insert(
            "add_knowledge".to_owned(),
            Box::new(AddKnowledgeTool::new(manager.clone())),
        );
    }

    // ---- 远端 A2A peer ----
    //
    // 必须在根助手**构造之前**发现 peer：工具箱是 `Arc<HashMap>`，
    // agent 一旦建好，工具就固定了，没法再往里塞远端 agent 工具。
    let a2a_config = A2AConfig::from_env();
    let peers = discover_peers(&a2a_config.peers, &a2a_config.token).await;
    let a2a_peer_names = attach_peers(&mut available_tools, peers);

    let callbacks = Callbacks::new(memory.clone());

    let researcher = Arc::new(callbacks.decorate(make_agent(
        "researcher",
        "联网检索与 GitHub 数据专家",
        RESEARCHER_INSTRUCTIONS.to_owned(),
        researcher_tools,
        12,
    )));
    let math = Arc::new(callbacks.decorate(make_agent(
        "math",
        "数学计算专家",
        MATH_INSTRUCTIONS.to_owned(),
        math_tools,
        6,
    )));

    let mut coder_instructions = CODER_INSTRUCTIONS.to_owned();
    if !sandbox_prompt.is_empty() {
        coder_instructions.push_str("\n\n");
        coder_instructions.push_str(&sandbox_prompt);
    }
    let coder = Arc::new(callbacks.decorate(make_agent(
        "coder",
        "代码执行与沙箱专家",
        coder_instructions,
        coder_tools,
        12,
    )));

    let root = Arc::new(callbacks.decorate(make_agent(
        "assistant",
        "通用助手，负责理解任务并分派专家",
        ROOT_INSTRUCTIONS.to_owned(),
        available_tools,
        24,
    )));
    root.set_sub_agents(vec![researcher.clone(), math.clone(), coder.clone()])?;

    let agents = [
        ("assistant", root.clone()),
        ("researcher", researcher),
        ("math", math),
        ("coder", coder),
    ]
    .into_iter()
    .map(|(name, agent)| (name.to_owned(), agent))
    .collect();

    let session_id = format!("chat-{}", Uuid::new_v4());
    let session = root
        .session_manager
        .get_or_create(&session_id, None)
        .await?;
    let memory_enabled = memory.is_some();
    let knowledge_enabled = knowledge.is_some();

    // ---- A2A 服务端 ----
    //
    // 一次性模式下跳过：进程马上就要退出，服务端来不及被任何人调用，
    // 只是白占一个端口。peer 接入不受影响——那是为了让本次回答能
    // 跨进程委派。
    let a2a_addr = if one_shot {
        None
    } else {
        match start_server(&root, &a2a_config).await {
            Ok(addr) => {
                println!("{} A2A 服务端: {addr}", styled("+", "1;32"));
                Some(addr)
            }
            Err(error) => {
                eprintln!("{} A2A 服务端未启动: {error:#}", styled("!", "1;33"));
                None
            }
        }
    };

    let a2a_server_enabled = a2a_addr.is_some();
    // 鉴权状态跟随服务端：服务端没起时说"已启用鉴权"是误导。
    let a2a_auth_enabled = a2a_server_enabled && a2a_config.auth_enabled();

    let a2a_peer_count = a2a_peer_names.len();

    Ok(App {
        model,
        session_id,
        root,
        agents,
        context: ExecutionContext::new(session),
        memory,
        knowledge,
        a2a_addr,
        a2a_peer_names,
        one_shot,
        features: Features {
            mcp,
            code,
            memory: memory_enabled,
            knowledge: knowledge_enabled,
            skills,
            a2a_server: a2a_server_enabled,
            a2a_auth: a2a_auth_enabled,
            a2a_peers: a2a_peer_count,
        },
    })
}

/// 组装一组共享的 callback。
///
/// 三个 callback 分别负责：
///
/// - `ContextOptimizer`：token 超阈值时压缩上下文；
/// - `MemoryInjectionCallback`：每轮 LLM 调用前注入相关长期记忆；
/// - `SearchCompressor`：工具返回结果过长时压缩。
struct Callbacks {
    optimizer: Arc<ContextOptimizer>,
    memory_injection: Option<Arc<MemoryInjectionCallback>>,
    search_compressor: Arc<SearchCompressor>,
}

impl Callbacks {
    fn new(memory: Option<Arc<TaskMemoryManager>>) -> Self {
        Self {
            optimizer: Arc::new(ContextOptimizer::new(GPT_6_LUNA_PRO_MODEL)),
            memory_injection: memory.map(|manager| Arc::new(MemoryInjectionCallback::new(manager))),
            search_compressor: Arc::new(SearchCompressor),
        }
    }

    /// 把三个 callback 挂到一个 agent 上。
    fn decorate(&self, agent: Agent) -> Agent {
        let mut agent = agent.with_before_llm_callbacks(self.optimizer.clone());
        if let Some(callback) = &self.memory_injection {
            agent = agent.with_before_llm_callbacks(callback.clone());
        }
        agent.with_after_tool_callbacks(self.search_compressor.clone())
    }
}

/// 打开长期记忆库。
fn init_memory(chat_model: &str) -> anyhow::Result<TaskMemoryManager> {
    std::fs::create_dir_all(".zephyr")?;
    TaskMemoryManager::new(MEMORY_DB, chat_model, TEXT_EMBEDDING_3_SMALL_MODEL)
}

/// 打开文档知识库。
///
/// 与记忆库共用 `.zephyr/` 目录，但用不同的 db 文件，互不干扰。
fn init_knowledge() -> anyhow::Result<KnowledgeBaseManager> {
    std::fs::create_dir_all(".zephyr")?;
    KnowledgeBaseManager::new(KNOWLEDGE_DB, TEXT_EMBEDDING_3_SMALL_MODEL)
}

/// 给 toolbox 挂上记忆检索工具。记忆库不可用时静默跳过。
fn add_memory_tool(toolbox: &mut ToolBox, memory: &Option<Arc<TaskMemoryManager>>) {
    if let Some(manager) = memory {
        toolbox.insert(
            "recall_memory".to_owned(),
            Box::new(RecallMemoryTool::new(manager.clone())),
        );
    }
}

/// 给 toolbox 挂上知识库检索工具。知识库不可用时静默跳过。
fn add_knowledge_tools(toolbox: &mut ToolBox, knowledge: &Option<Arc<KnowledgeBaseManager>>) {
    if let Some(manager) = knowledge {
        toolbox.insert(
            "search_knowledge".to_owned(),
            Box::new(SearchKnowledgeTool::new(manager.clone())),
        );
    }
}

/// 构造一个 agent。
fn make_agent(
    name: &str,
    description: &str,
    instructions: String,
    toolbox: ToolBox,
    max_steps: usize,
) -> Agent {
    Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec![instructions],
        Arc::new(toolbox),
    )
    .with_name(name.to_owned())
    .with_description(description.to_owned())
    .with_max_steps(max_steps)
}
