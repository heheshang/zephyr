//! A2A 服务端：把本项目的 `Agent` 暴露成 A2A 端点。
//!
//! # 规范依据
//!
//! - 规范 8.2 节：Agent Card 通过 `/.well-known/agent-card.json` 暴露；
//! - 规范 9.4 节：JSON-RPC 绑定下所有方法走同一个 HTTP 端点（`POST /rpc`）；
//! - 规范 9.4.2 节：流式方法用 SSE（`text/event-stream`）返回。
//!
//! # 模块结构
//!
//! ```text
//!   server.rs     ── A2AServer / A2AServerState，路由装配
//!   dispatch.rs   ── 方法分派 + 各方法的业务逻辑
//!   executor.rs   ── 在 A2A 任务上运行 agent（协议 ↔ runtime 的桥）
//!   streaming.rs  ── SSE 响应
//! ```
//!
//! # 请求流程
//!
//! ```text
//!   GET  /.well-known/agent-card.json
//!         │
//!         └──► 返回 AgentCard（客户端据此发现能力）
//!
//!   POST /rpc   Content-Type: application/json
//!         │
//!         ├──► 解析 JSON-RPC 请求
//!         ├──► 校验 jsonrpc 版本
//!         ├──► 按 method 分派
//!         │      ├── SendMessage            ──► 阻塞执行 agent
//!         │      ├── SendStreamingMessage   ──► SSE 推送事件
//!         │      ├── GetTask                ──► 查任务状态
//!         │      ├── ListTasks              ──► 列任务
//!         │      ├── CancelTask             ──► 取消任务
//!         │      ├── SubscribeToTask        ──► SSE 订阅
//!         │      └── GetExtendedAgentCard  ──► 扩展卡片
//!         │
//!         └──► 返回 JSON-RPC 响应（成功 result / 失败 error）
//! ```
//!
//! # 任务与 agent 的关系
//!
//! 服务端把一次 `SendMessage` 转成一次 [`crate::agent::model::Agent::run_in_context`]：
//!
//! - A2A 的 `contextId` 复用 agent 的 `session_id`，因此多轮对话天然共享上下文；
//! - agent 的最终输出转成 `Artifact`；
//! - agent 需要用户补充信息时（`AgentStatus::PendingConfirmation`），
//!   任务进入 `TASK_STATE_INPUT_REQUIRED`。

use std::{net::SocketAddr, sync::Arc};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};

use crate::{
    a2a::{
        auth::{Authenticator, authorization_header},
        card::AgentCard,
        error::A2AError,
        jsonrpc::{JsonRpcRequest, JsonRpcResponse, parse_service_params},
        task_store::TaskStore,
        types::WELL_KNOWN_AGENT_CARD_PATH,
    },
    agent::model::Agent,
    session::session_manager::SessionManager,
};

pub mod dispatch;
pub mod executor;
pub mod streaming;

/// 服务端共享状态。
pub struct A2AServerState {
    /// 被暴露的 agent。
    pub agent: Arc<Agent>,
    /// Agent Card。
    pub card: AgentCard,
    /// 任务存储。
    pub task_store: Arc<dyn TaskStore>,
    /// 会话管理器（与 agent 共用，保证多轮上下文一致）。
    pub session_manager: Arc<dyn SessionManager>,
    /// 鉴权器。`None` 表示不校验（默认）。
    ///
    /// 鉴权只保护 `/rpc`；Agent Card 端点保持公开，
    /// 否则客户端拿不到卡片就不知道该用什么凭据。
    pub authenticator: Option<Arc<dyn Authenticator>>,
}

impl A2AServerState {
    /// 组装服务端状态（不鉴权）。
    pub fn new(
        agent: Arc<Agent>,
        card: AgentCard,
        task_store: Arc<dyn TaskStore>,
        session_manager: Arc<dyn SessionManager>,
    ) -> Self {
        Self {
            agent,
            card,
            task_store,
            session_manager,
            authenticator: None,
        }
    }

    /// 启用鉴权。
    ///
    /// 鉴权器通过 `Arc` 持有，因此可以与 `Agent` 一样在多处共享。
    pub fn with_authenticator(mut self, authenticator: Arc<dyn Authenticator>) -> Self {
        self.authenticator = Some(authenticator);
        self
    }
}

/// A2A 服务端。
pub struct A2AServer {
    state: Arc<A2AServerState>,
}

impl A2AServer {
    /// 用已有的服务端状态创建。
    pub fn new(state: Arc<A2AServerState>) -> Self {
        Self { state }
    }

    /// 用一个 agent 与任务存储快速创建。
    ///
    /// `rpc_url` 是 **JSON-RPC 端点的完整地址**，例如
    /// `http://127.0.0.1:8080/rpc`。它会原样写进 Agent Card 的
    /// `supportedInterfaces[0].url`——规范 4.4.6 节要求这个地址就是
    /// 客户端可以直接调用的 JSON-RPC 端点，因此不能是"基础地址"。
    ///
    /// `version` 通常取 crate 版本。
    ///
    /// 服务端默认声明 `streaming: true`——它确实实现了 SSE。
    /// 如果不需要流式，用 [`A2AServer::new`] 自行组装 [`A2AServerState`]。
    pub fn from_agent(
        agent: Arc<Agent>,
        rpc_url: impl Into<String>,
        version: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let rpc_url = rpc_url.into();
        let card = AgentCard::from_agent(&agent, rpc_url, version).with_streaming(true);

        // agent 与服务端共享同一个 session manager，
        // 这样 A2A 的 contextId 与 agent 的 session_id 指向同一份会话。
        let session_manager: Arc<dyn SessionManager> = agent.session_manager.clone();

        Ok(Self::new(Arc::new(A2AServerState::new(
            agent,
            card,
            Arc::new(crate::a2a::task_store::InMemoryTaskStore::new()),
            session_manager,
        ))))
    }

    /// 构建 axum 路由。
    ///
    /// 路由包含两条：
    ///
    /// - `GET /.well-known/agent-card.json` —— Agent Card 发现；
    /// - `POST /rpc` —— JSON-RPC 方法入口。
    ///
    /// 另外也挂载了 `/a2a/rpc` 作为别名，方便把 A2A 端点放在子路径下。
    pub fn router(&self) -> Router {
        let state = self.state.clone();
        Router::new()
            .route(WELL_KNOWN_AGENT_CARD_PATH, get(agent_card_handler))
            .route("/rpc", post(rpc_handler))
            .route("/a2a/rpc", post(rpc_handler))
            .with_state(state)
    }

    /// 启动 HTTP 服务并阻塞运行。
    pub async fn serve(self, addr: SocketAddr) -> anyhow::Result<()> {
        let router = self.router();
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(
            "A2A server listening on http://{addr}{WELL_KNOWN_AGENT_CARD_PATH}"
        );
        axum::serve(listener, router).await?;
        Ok(())
    }
}

/// `GET /.well-known/agent-card.json` 的处理器。
async fn agent_card_handler(State(state): State<Arc<A2AServerState>>) -> impl IntoResponse {
    // Agent Card 是公开信息，规范 8.6 节建议用 ETag / Cache-Control 控制缓存。
    let etag = format!("W/\"{}\"", crate::a2a::types::A2A_PROTOCOL_VERSION);

    (
        StatusCode::OK,
        [(axum::http::header::CACHE_CONTROL, "public, max-age=300")],
        Json(state.card.clone()),
    )
        .into_response()
        .with_etag(etag)
}

/// 便捷方法：给响应加 ETag。
trait WithEtag {
    fn with_etag(self, etag: String) -> Response;
}

impl WithEtag for Response {
    fn with_etag(self, etag: String) -> Response {
        let mut response = self;
        if let Ok(value) = etag.parse() {
            response
                .headers_mut()
                .insert(axum::http::header::ETAG, value);
        }
        response
    }
}

/// `POST /rpc` 的处理器。
///
/// 先按方法分流：流式方法返回 SSE，其余返回普通 JSON。
async fn rpc_handler(
    State(state): State<Arc<A2AServerState>>,
    headers: HeaderMap,
    Query(_query): Query<std::collections::HashMap<String, String>>,
    body: String,
) -> Response {
    // 鉴权：先于任何业务处理。
    //
    // 失败时返回 HTTP 401 + JSON-RPC 错误——客户端既能看出是鉴权问题
    // （状态码），也能按 JSON-RPC 惯例解析错误体。
    if let Some(authenticator) = &state.authenticator
        && let Err(error) = authenticator.authenticate(authorization_header(&headers))
    {
        let failure = JsonRpcResponse::failure(
            None,
            A2AError::AuthenticationFailed(error.message()),
        );
        return dispatch::json_response(failure, StatusCode::UNAUTHORIZED);
    }

    let request: JsonRpcRequest = match serde_json::from_str(&body) {
        Ok(request) => request,
        Err(error) => {
            let failure = JsonRpcResponse::failure(
                None,
                A2AError::InvalidRequest(format!("malformed JSON: {error}")),
            );
            return dispatch::json_response(failure, StatusCode::BAD_REQUEST);
        }
    };

    // 服务参数：客户端通过 HTTP 头传递 A2A-Version / A2A-Extensions。
    let header_map: std::collections::HashMap<String, String> = headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_owned(), value.to_owned()))
        })
        .collect();
    let service_params = parse_service_params(&header_map);

    // 流式方法走 SSE。
    if streaming::is_streaming_method(&request.method) {
        return streaming::streaming_response(state, request, service_params);
    }

    let response = dispatch::dispatch(&state, request, service_params).await;

    let status = match &response {
        JsonRpcResponse::Success(_) => StatusCode::OK,
        JsonRpcResponse::Failure(error) => {
            if error.error.code == -32601 || error.error.code == -32600 {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::OK
            }
        }
    };

    dispatch::json_response(response, status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        a2a::task_store::InMemoryTaskStore, agent::model::Agent,
        session::session_manager::InMemorySessionManager, tool::ToolBox,
    };

    fn server() -> (A2AServer, Arc<A2AServerState>) {
        let agent = Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec!["You are a test agent.".to_owned()],
                Arc::new(ToolBox::new()),
            )
            .with_name("tester".to_owned())
            .with_description("A test agent".to_owned()),
        );

        let card = AgentCard::from_agent(&agent, "http://127.0.0.1:9999/a2a", "0.1.0")
            .with_streaming(true);
        let session_manager = Arc::new(InMemorySessionManager::new());
        let state = Arc::new(A2AServerState::new(
            agent,
            card,
            Arc::new(InMemoryTaskStore::new()),
            session_manager,
        ));
        (A2AServer::new(state.clone()), state)
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let (server, state) = server();
        let request = JsonRpcRequest::new("r1", "NoSuchMethod", serde_json::json!({}));
        let response = dispatch::dispatch(&state, request, Default::default()).await;

        match response {
            JsonRpcResponse::Failure(failure) => assert_eq!(failure.error.code, -32601),
            JsonRpcResponse::Success(_) => panic!("expected a failure"),
        }
        let _ = server;
    }

    #[tokio::test]
    async fn invalid_jsonrpc_version_is_rejected() {
        let (_server, state) = server();
        let request = JsonRpcRequest {
            jsonrpc: "1.0".to_owned(),
            id: Some(serde_json::json!("r1")),
            method: crate::a2a::jsonrpc::method::GET_TASK.to_owned(),
            params: None,
        };
        let response = dispatch::dispatch(&state, request, Default::default()).await;

        match response {
            JsonRpcResponse::Failure(failure) => assert_eq!(failure.error.code, -32600),
            JsonRpcResponse::Success(_) => panic!("expected a failure"),
        }
    }

    #[tokio::test]
    async fn get_task_for_unknown_id_returns_task_not_found() {
        let (_server, state) = server();
        let request = JsonRpcRequest::new(
            "r1",
            crate::a2a::jsonrpc::method::GET_TASK,
            serde_json::json!({"id": "nope"}),
        );
        let response = dispatch::dispatch(&state, request, Default::default()).await;

        match response {
            JsonRpcResponse::Failure(failure) => assert_eq!(failure.error.code, -32001),
            JsonRpcResponse::Success(_) => panic!("expected a failure"),
        }
    }

    #[tokio::test]
    async fn unsupported_version_header_is_rejected() {
        let (_server, state) = server();
        let request = JsonRpcRequest::new(
            "r1",
            crate::a2a::jsonrpc::method::GET_TASK,
            serde_json::json!({}),
        );
        let params = crate::a2a::jsonrpc::ServiceParams {
            version: Some("0.1".to_owned()),
            extensions: Vec::new(),
        };
        let response = dispatch::dispatch(&state, request, params).await;

        match response {
            JsonRpcResponse::Failure(failure) => assert_eq!(failure.error.code, -32009),
            JsonRpcResponse::Success(_) => panic!("expected a failure"),
        }
    }

    #[tokio::test]
    async fn extended_agent_card_returns_the_card() {
        let (_server, state) = server();
        let request = JsonRpcRequest::new(
            "r1",
            crate::a2a::jsonrpc::method::GET_EXTENDED_AGENT_CARD,
            serde_json::json!({}),
        );
        let response = dispatch::dispatch(&state, request, Default::default()).await;

        match response {
            JsonRpcResponse::Success(success) => {
                assert_eq!(success.result["name"], serde_json::json!("tester"));
            }
            JsonRpcResponse::Failure(failure) => panic!("unexpected failure: {:?}", failure.error),
        }
    }

    #[tokio::test]
    async fn card_is_served_at_well_known_path() {
        let (server, _state) = server();
        let router = server.router();
        let request = axum::http::Request::builder()
            .uri(WELL_KNOWN_AGENT_CARD_PATH)
            .body(axum::body::Body::empty())
            .unwrap();

        let response = tower::ServiceExt::oneshot(router, request)
            .await
            .expect("router should respond");
        assert_eq!(response.status(), StatusCode::OK);

        let card: AgentCard = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(card.name, "tester");
        assert!(card.supports_streaming());
    }
}
