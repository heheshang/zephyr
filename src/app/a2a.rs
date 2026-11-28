//! A2A 集成的配置与启动。
//!
//! # 两个方向
//!
//! ```text
//!   本机作为服务端 ──▶ 对外暴露根助手（Agent Card + /rpc）
//!   本机作为客户端 ──▶ 发现远端 A2A agent，包装成本地工具
//! ```
//!
//! 两者都是**可选能力**：任一环节失败都只打印 warning，不影响主对话。
//! 这与 MCP、记忆库、知识库的处理方式一致。
//!
//! # 为什么 peer 发现必须在 agent 构造之前
//!
//! 工具箱是 `Arc<HashMap>`，agent 一旦建好，工具就固定了，
//! 没法再往里塞远端 agent 工具。因此发现流程要早于 agent 树的组装。

use std::{net::SocketAddr, sync::Arc};

use crate::{
    a2a::{
        auth::{Authenticator, BearerTokenAuthenticator},
        card::AgentCard,
        client::A2AClient,
        client_tool::A2ATool,
        server::{A2AServer, A2AServerState},
        task_store::InMemoryTaskStore,
    },
    agent::model::Agent,
    tool::ToolBox,
};

use crate::constant::{
    A2A_DEFAULT_HOST, A2A_DEFAULT_PORT, A2A_ENV_HOST, A2A_ENV_PEERS, A2A_ENV_PORT,
    A2A_ENV_TOKEN,
};

/// A2A 集成配置。
///
/// # 配置来源
///
/// - `ZEPHYR_A2A_HOST` —— 服务端监听地址，默认 `127.0.0.1`；
/// - `ZEPHYR_A2A_PORT` —— 服务端监听端口，默认 `8899`；
/// - `ZEPHYR_A2A_PEERS` —— 逗号分隔的远端 A2A agent 地址列表；
/// - `ZEPHYR_A2A_TOKEN` —— 服务端要求的 Bearer Token（**可选**）。
#[derive(Debug, Clone)]
pub struct A2AConfig {
    /// 服务端监听地址。
    pub host: String,
    /// 服务端监听端口。
    pub port: u16,
    /// 远端 A2A agent 的基地址。
    pub peers: Vec<String>,
    /// 服务端要求的 Bearer Token。为空表示不鉴权。
    pub token: String,
}

impl A2AConfig {
    /// 从环境变量读取配置。
    pub fn from_env() -> Self {
        let host = std::env::var(A2A_ENV_HOST).unwrap_or_else(|_| A2A_DEFAULT_HOST.to_owned());

        let port = std::env::var(A2A_ENV_PORT)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(A2A_DEFAULT_PORT);

        let peers = std::env::var(A2A_ENV_PEERS)
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect();

        let token = std::env::var(A2A_ENV_TOKEN).unwrap_or_default();

        Self {
            host,
            port,
            peers,
            token,
        }
    }

    /// 是否开启了鉴权。
    pub fn auth_enabled(&self) -> bool {
        !self.token.trim().is_empty()
    }
}

/// 启动 A2A 服务端，把根助手暴露出去。
///
/// # 与会话的关系
///
/// 服务端复用根助手的 `session_manager`，因此 A2A 的 `contextId`
/// 与 agent 的 `session_id` 指向同一份会话——外部系统发起的对话
/// 与终端里的对话共享上下文。
///
/// 返回服务端实际监听的地址，失败时返回 `Err`。
pub async fn start_server(root: &Arc<Agent>, config: &A2AConfig) -> anyhow::Result<String> {
    // 对外暴露时，Agent Card 里的 url 必须写客户端能直接访问的地址。
    // 监听 0.0.0.0 时不能把 0.0.0.0 写进卡片——那是通配地址，
    // 客户端拿到它没法用。这里用 127.0.0.1 作为"本机可达"的示例地址。
    let advertised_host = if config.host == "0.0.0.0" {
        "127.0.0.1"
    } else {
        config.host.as_str()
    };
    let rpc_url = format!("http://{advertised_host}:{}/rpc", config.port);

    let card =
        AgentCard::from_agent(root, &rpc_url, env!("CARGO_PKG_VERSION")).with_streaming(true);

    // 鉴权：配置了 token 才启用。
    //
    // 启用时卡片要声明 `securitySchemes`，客户端才知道该发什么头——
    // 这是规范 4.4.7 节的用途。卡片端点本身仍然公开，
    // 只有 `/rpc` 受保护。
    let (card, authenticator) = if config.auth_enabled() {
        (
            card.with_bearer_auth(),
            Some(Arc::new(BearerTokenAuthenticator::new(
                config.token.clone(),
            )) as Arc<dyn Authenticator>),
        )
    } else {
        (card, None)
    };

    let mut state = A2AServerState::new(
        root.clone(),
        card,
        Arc::new(InMemoryTaskStore::new()),
        root.session_manager.clone(),
    );

    if let Some(authenticator) = authenticator {
        state = state.with_authenticator(authenticator);
    }

    let server = A2AServer::new(Arc::new(state));

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let display = format!("http://{}:{}", config.host, config.port);

    // 服务端在后台任务里跑，主线程继续走交互式 REPL。
    tokio::spawn(async move {
        if let Err(error) = server.serve(addr).await {
            eprintln!("{} A2A server stopped: {error:#}", crate::app::render::styled("error:", "1;31"));
        }
    });

    // 等待端口真正监听上，避免调用方立刻访问时连接被拒。
    wait_for_port(&config.host, config.port).await;

    Ok(display)
}

/// 轮询等待端口可连接。
///
/// 服务端是异步启动的，这里确认它真的 listen 上了再返回。
async fn wait_for_port(host: &str, port: u16) {
    use tokio::net::TcpStream;

    let target = format!("{host}:{port}");
    for _ in 0..50 {
        if TcpStream::connect(&target).await.is_ok() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// 发现远端 A2A agent。
///
/// # 失败策略
///
/// 单个 peer 不可达只打印 warning，其余 peer 照常接入——
/// 一个远端挂了不该拖垮整个协调员。
///
/// # 鉴权
///
/// 远端 peer 可能开启了鉴权。`token` 非空时用
/// [`A2AClient::discover_with_auth`] 带上 Bearer Token；
/// 为空时走普通发现（对方没开鉴权时也能用）。
///
/// # 工具命名
///
/// 工具名取远端 Agent Card 的 `name`，但**必须去重**：
///
/// - 远端 agent 可能与本地 agent 同名（本项目根助手就叫 `assistant`）；
/// - 多个 peer 之间也可能同名。
///
/// 同名会导致后接入的 peer 覆盖先接入的（工具箱是 `HashMap`），
/// 表现为"某个远端 agent 莫名失联"，很难排查。
/// 因此这里在冲突时加后缀，且不覆盖已有工具。
///
/// 返回 `(工具名, 客户端)` 列表。
pub async fn discover_peers(peers: &[String], token: &str) -> Vec<(String, A2AClient)> {
    let mut discovered: Vec<(String, A2AClient)> = Vec::new();

    for peer in peers {
        let result = if token.trim().is_empty() {
            A2AClient::discover(peer).await
        } else {
            A2AClient::discover_with_auth(peer, token).await
        };

        match result {
            Ok(client) => {
                let base_name = client
                    .card()
                    .map(|card| card.name.clone())
                    .unwrap_or_else(|| peer.clone());

                let name = unique_tool_name(&base_name, &discovered);
                println!(
                    "{} A2A peer 已接入: {name} ({peer})",
                    crate::app::render::styled("+", "1;32")
                );
                discovered.push((name, client));
            }
            Err(error) => {
                eprintln!(
                    "{} A2A peer 不可达 ({peer}): {error}",
                    crate::app::render::styled("!", "1;33")
                );
            }
        }
    }

    discovered
}

/// 生成一个在已接入 peer 中唯一的工具名。
///
/// 冲突时追加 `-2`、`-3` 等后缀。宁可名字难看，也不能让一个 peer
/// 覆盖另一个——那会让远端 agent 凭空消失。
fn unique_tool_name(base: &str, existing: &[(String, A2AClient)]) -> String {
    let taken = |name: &str| existing.iter().any(|(known, _)| known == name);

    if !taken(base) {
        return base.to_owned();
    }

    for suffix in 2..100 {
        let candidate = format!("{base}-{suffix}");
        if !taken(&candidate) {
            return candidate;
        }
    }

    // 理论上到不了这里；真到了也要返回一个不冲突的名字。
    format!("{base}-{}", existing.len())
}

/// 把远端 peer 挂到根助手的工具箱里。
///
/// # 冲突策略
///
/// 远端 agent 的名字可能撞上本地工具名（如 `calculator`）。
/// 这里跳过而不是覆盖——覆盖会让本地工具凭空消失。
///
/// 返回实际挂上的 peer 名称。
pub fn attach_peers(toolbox: &mut ToolBox, peers: Vec<(String, A2AClient)>) -> Vec<String> {
    let mut attached = Vec::with_capacity(peers.len());

    for (name, client) in peers {
        if toolbox.contains_key(&name) {
            eprintln!(
                "{} A2A peer 工具名 '{name}' 与本地工具冲突，已跳过",
                crate::app::render::styled("!", "1;33")
            );
            continue;
        }
        toolbox.insert(name.clone(), Box::new(A2ATool::new(client)));
        attached.push(name);
    }

    attached
}
