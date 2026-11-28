//! # A2A（Agent2Agent）协议实现
//!
//! 本模块实现 [A2A 协议](https://a2a-protocol.org) 的 **JSON-RPC over HTTP**
//! 绑定，让本项目的 `Agent` 既能作为 A2A **服务端**被别的系统发现和调用，
//! 也能作为 A2A **客户端**去调用远端的 agent。
//!
//! ## A2A 与 MCP 的关系
//!
//! 项目里已有 [`crate::tool::mcp`]（MCP 客户端）。两者的分工是：
//!
//! ```text
//!   MCP ──► agent 怎么调「工具」      （工具在 agent 外部，能力由模型调用）
//!   A2A ──► agent 怎么和「另一个 agent」对话（对方是自主体，自带模型与工具）
//! ```
//!
//! 可以类比成：
//!
//! - MCP = agent 调用 **函数**（函数是被动的、无状态的）；
//! - A2A = agent 调用 **同事**（同事有自己的判断、会多轮沟通、可能有自己的工具）。
//!
//! ## 模块结构
//!
//! ```text
//!   types.rs       ── 数据模型（Task / Message / Part / Artifact …）
//!   error.rs       ── A2A 错误类型 → JSON-RPC 错误码映射
//!   jsonrpc.rs     ── JSON-RPC 2.0 信封与方法名常量
//!   card.rs        ── Agent Card（自描述清单）
//!   auth.rs        ── 鉴权（HTTP Bearer Token）
//!   task_store.rs  ── 任务存储（内存实现 + Session 桥接实现）
//!   server.rs      ── 服务端：/.well-known/agent-card.json + POST /rpc
//!   client.rs      ── 客户端：发现 + 发消息 + 查任务
//!   client_tool.rs ── 把远端 A2A agent 包装成本地 Tool
//!   sse.rs         ── SSE 流解析（服务端与客户端共用同一套规则）
//! ```
//!
//! 其中几个目录再往下分：
//!
//! ```text
//!   types/{message,task,stream}.rs ── 数据模型按概念分文件
//!   card/{skill,build}.rs          ── skill 声明 + 从 Agent 生成 skill 的映射
//!   server/{dispatch,executor,streaming}.rs ── 分派 / 协议桥 / SSE
//!   task_store/lifecycle.rs         ── 任务状态机（纯函数）
//! ```
//!
//! ## 端到端流程
//!
//! ```text
//!   服务端 A                                     客户端 B
//!   ────────                                     ────────
//!                                                 │
//!   GET /.well-known/agent-card.json ◄────────────┤  discover()
//!         │                                       │
//!         └──► AgentCard{name, skills,           │
//!              capabilities, supportedInterfaces} │
//!                                                 │
//!   POST /rpc  {method:"SendMessage"} ◄───────────┤  send_text()
//!         │                                       │
//!         ├──▶ Agent::run_in_context()            │
//!         │                                       │
//!         └──► {result:{task:{status, artifacts}}} │
//!                                                 │
//!   POST /rpc  {method:"GetTask"}     ◄───────────┤  get_task()
//!         └──► {result:{task:{...}}}              │
//! ```
//!
//! ## 快速上手
//!
//! 服务端：
//!
//! ```ignore
//! use zephyr::a2a::server::A2AServer;
//!
//! let agent = Arc::new(Agent::new(model, instructions, toolbox));
//! let server = A2AServer::from_agent(agent, "http://127.0.0.1:8080/a2a", "0.1.0")?;
//! server.serve("127.0.0.1:8080".parse()?).await?;
//! ```
//!
//! 客户端：
//!
//! ```ignore
//! use zephyr::a2a::client::A2AClient;
//!
//! let client = A2AClient::discover("http://127.0.0.1:8080").await?;
//! let task = client.send_text("你好").await?;
//! ```
//!
//! ## 鉴权
//!
//! 服务端可以开启 HTTP Bearer Token 鉴权（规范 4.4.7 节的
//! `securitySchemes`）：
//!
//! ```ignore
//! use zephyr::a2a::auth::BearerTokenAuthenticator;
//!
//! let state = A2AServerState::new(agent, card, store, sessions)
//!     .with_authenticator(Arc::new(BearerTokenAuthenticator::new("secret")));
//! ```
//!
//! 客户端相应地带上凭据：
//!
//! ```ignore
//! let client = A2AClient::discover_with_auth("http://127.0.0.1:8080", "secret").await?;
//! ```
//!
//! 注意 **Agent Card 本身始终公开**——客户端要先拿到卡片才知道
//! 该用什么凭据（`securitySchemes` 就在卡片里）。受保护的只有 `/rpc`。
//!
//! ## 规范一致性
//!
//! 实现严格遵循规范的关键约束：
//!
//! - JSON 字段 **camelCase**，枚举值 **SCREAMING_SNAKE_CASE**（规范 5.5 节）；
//! - 时间戳 **ISO 8601 UTC**（规范 5.6.1 节）；
//! - `Part` 是 OneOf：`text` / `raw` / `url` / `data` 只能有一个（规范 4.1.6 节）；
//! - 能力未声明时，相关操作必须返回 `UnsupportedOperationError`（规范 3.3.4 节）；
//! - 服务参数通过 HTTP 头传递（规范 9.2 节）。

pub mod auth;
pub mod card;
pub mod client;
pub mod client_tool;
pub mod error;
pub mod jsonrpc;
pub mod server;
pub mod sse;
pub mod task_store;
pub mod types;

pub use auth::{AuthError, Authenticator, BearerTokenAuthenticator};
pub use card::{
    AgentCapabilities, AgentCard, AgentExtension, AgentInterface, AgentProvider, AgentSkill,
    SecurityRequirement, SecurityScheme, SecuritySchemeType,
};
pub use client::A2AClient;
pub use client_tool::A2ATool;
pub use error::{A2AError, JsonRpcError};
pub use server::{A2AServer, A2AServerState};
pub use task_store::{
    InMemoryTaskStore, SessionTaskStore, TaskLifecycle, TaskStore, artifacts_to_text,
    ensure_cancelable, merge_or_create_task, truncate_history,
};
pub use types::{
    Artifact, Message, Part, Role, SendMessageConfiguration, SendMessageRequest,
    SendMessageResponse, StreamResponse, Task, TaskArtifactUpdateEvent, TaskState, TaskStatus,
    TaskStatusUpdateEvent,
};
