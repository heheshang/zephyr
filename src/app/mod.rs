//! 终端交互式应用的**组装层**。
//!
//! # 职责划分
//!
//! `main.rs` 只负责"进程的生命周期"：解析参数 → 启动 → REPL 循环 → 退出。
//! 真正组装的逻辑放在这里，按职责拆成若干子模块：
//!
//! ```text
//!   app/
//!   ├── mod.rs        ── App / Features 数据结构，模块入口
//!   ├── builder.rs    ── 组装工具箱、agent 树、可选能力
//!   ├── prompts.rs    ── 各 agent 的系统提示词
//!   ├── a2a.rs        ── A2A 配置、服务端启动、远端 peer 发现
//!   ├── turn.rs       ── 执行一个用户回合（含工具审批）
//!   ├── commands.rs   ── 斜杠命令处理
//!   ├── render.rs     ── 终端渲染（颜色 / Markdown / 横幅）
//!   └── help.rs       ── 帮助文本
//! ```
//!
//! # 数据流
//!
//! ```text
//!   main()
//!     │
//!     ├─▶ builder::build_app()
//!     │      ├── 工具箱（MCP + 本地工具）
//!     │      ├── agent 树（assistant / researcher / math / coder）
//!     │      ├── 可选能力（记忆库、知识库）
//!     │      └── A2A（服务端 + 远端 peer 工具）
//!     │
//!     ├─▶ loop
//!     │      ├── "/xxx" ──▶ commands::handle_command()
//!     │      └── "问题"  ──▶ turn::run_turn()
//!     │
//!     └─▶ 退出
//! ```

pub mod a2a;
pub mod builder;
pub mod commands;
pub mod help;
pub mod prompts;
pub mod render;
pub mod turn;

use std::{collections::HashMap, sync::Arc};

use crate::{
    agent::{context::ExecutionContext, model::Agent},
    knowledge::knowledge_base::KnowledgeBaseManager,
    memory::memory_manager::TaskMemoryManager,
};

pub use builder::build_app;

/// 启动时探测到的能力开关。
///
/// 终端横幅与 `/status` 需要如实反映"哪些能力真的可用"，
/// 因此这里记录的是**探测结果**，而不是"用户请求了什么"。
#[derive(Debug, Clone, Copy, Default)]
pub struct Features {
    /// MCP 客户端是否连上了外部 MCP Server。
    pub mcp: bool,
    /// 沙箱代码执行是否可用。
    pub code: bool,
    /// 长期记忆库是否可用。
    pub memory: bool,
    /// 文档知识库是否可用。
    pub knowledge: bool,
    /// 沙箱里是否发现了技能（skills）。
    pub skills: bool,
    /// A2A 服务端是否已启动。
    pub a2a_server: bool,
    /// A2A 服务端是否开启了鉴权。
    pub a2a_auth: bool,
    /// 已接入的远端 A2A agent 数量。
    pub a2a_peers: usize,
}

/// 终端应用的全部运行时状态。
pub struct App {
    /// 当前使用的模型名。
    pub model: String,
    /// 当前会话 id。
    pub session_id: String,
    /// 根 agent（协调员）。
    pub root: Arc<Agent>,
    /// agent 树按名字索引，方便 `/agents` `/tools` 查询。
    pub agents: HashMap<String, Arc<Agent>>,
    /// 当前执行上下文（携带会话与事件）。
    pub context: ExecutionContext,
    /// 长期记忆库（不可用时为 `None`）。
    pub memory: Option<Arc<TaskMemoryManager>>,
    /// 文档知识库（不可用时为 `None`）。
    pub knowledge: Option<Arc<KnowledgeBaseManager>>,
    /// 能力开关。
    pub features: Features,
    /// A2A 服务端地址（未启动时为 `None`）。
    pub a2a_addr: Option<String>,
    /// 已接入的远端 A2A agent 名称（对应工具箱里的同名工具）。
    pub a2a_peer_names: Vec<String>,
    /// 是否为一次性模式（`cargo run -- "问题"`）。
    ///
    /// 一次性模式下不启动 A2A 服务端——进程马上就退出，
    /// 服务端来不及被调用。这个标记让 `/a2a` 能解释原因，
    /// 而不是含糊地说"未启动"。
    pub one_shot: bool,
}
