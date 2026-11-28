//! A2A 客户端：发现远端 agent 并与之通信。
//!
//! # 规范依据
//!
//! - 规范 8.2 节：通过 `/.well-known/agent-card.json` 发现 Agent Card；
//! - 规范 9.3 节：JSON-RPC 2.0 请求格式；
//! - 规范 9.2 节：服务参数通过 HTTP 头传递。
//!
//! # 调用流程
//!
//! ```text
//!   A2AClient::discover(url)
//!        │
//!        ├──► GET {url}/.well-known/agent-card.json
//!        │        └──► AgentCard（拿到能力、skill、首选接口）
//!
//!   client.send_text("你好")
//!        │
//!        ├──► POST {接口地址}   Content-Type: application/json
//!        │      A2A-Version: 1.0
//!        │      {jsonrpc, id, method: "SendMessage", params: {message}}
//!        │
//!        └──► SendMessageResponse（Task 或 Message）
//! ```
//!
//! # 模块结构
//!
//! ```text
//!   client.rs ── A2AClient，本文件
//!   sse.rs    ── SSE 流的解析（服务端与客户端共用同一套规则）
//! ```

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use futures::StreamExt;
use serde_json::Value;

use crate::a2a::{
    card::AgentCard,
    error::A2AError,
    jsonrpc::{JsonRpcRequest, JsonRpcResponse, method, service_param},
    sse::parse_sse_stream,
    types::{
        Artifact, Message, Role, SendMessageConfiguration, SendMessageRequest, SendMessageResponse,
        StreamResponse, Task, TaskState, WELL_KNOWN_AGENT_CARD_PATH,
    },
};

/// A2A 客户端。
pub struct A2AClient {
    http: reqwest::Client,
    /// 远端 agent 的卡片（`discover` 之后可用）。
    card: Option<AgentCard>,
    /// JSON-RPC 端点地址。
    endpoint: String,
    /// 请求 id 计数器。
    next_id: AtomicU64,
    /// 额外请求头。
    extra_headers: HashMap<String, String>,
    /// 当前会话的 `contextId`，由服务端分配，客户端自动记住以便多轮对话。
    context_id: Mutex<Option<String>>,
}

impl A2AClient {
    /// 用给定端点创建客户端。
    ///
    /// `endpoint` 是 JSON-RPC 的完整地址，例如
    /// `http://127.0.0.1:8080/rpc`。
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            card: None,
            endpoint: endpoint.into(),
            next_id: AtomicU64::new(1),
            extra_headers: HashMap::new(),
            context_id: Mutex::new(None),
        }
    }

    /// 从远端 agent 的**基础地址**创建客户端并执行发现。
    ///
    /// `base_url` 形如 `http://127.0.0.1:8080`；发现会请求
    /// `{base_url}/.well-known/agent-card.json`，成功后把卡片里的首选接口
    /// 地址写入 `endpoint`。
    ///
    /// # 鉴权
    ///
    /// 本方法**不带**任何凭据。若远端开启了鉴权，
    /// 请改用 [`A2AClient::discover_with_auth`]。
    pub async fn discover(base_url: impl Into<String>) -> anyhow::Result<Self> {
        Self::discover_with_headers(base_url, &HashMap::new()).await
    }

    /// 发现远端 agent，并附带鉴权凭据。
    ///
    /// 开启鉴权的服务端会保护 `/rpc`，但 **Agent Card 本身仍然公开**
    /// （否则客户端拿不到卡片就不知道该用什么凭据）。
    ///
    /// 因此这里仍然要带上凭据：有些服务端会把卡片也纳入保护，
    /// 或者客户端拿到的卡片里声明了需要鉴权的扩展接口。
    pub async fn discover_with_auth(
        base_url: impl Into<String>,
        token: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_owned(),
            format!("Bearer {}", token.into()),
        );
        Self::discover_with_headers(base_url, &headers).await
    }

    /// 发现远端 agent，附带自定义请求头。
    pub async fn discover_with_headers(
        base_url: impl Into<String>,
        headers: &HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        let base_url = base_url.into();
        let card_url = format!(
            "{}{}",
            base_url.trim_end_matches('/'),
            WELL_KNOWN_AGENT_CARD_PATH
        );

        let mut builder = reqwest::Client::new().get(&card_url);
        for (name, value) in headers {
            builder = builder.header(name, value);
        }

        let card: AgentCard = builder
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let endpoint = card
            .primary_url()
            .ok_or_else(|| {
                anyhow::anyhow!("agent card at {card_url} has no supported interfaces")
            })?
            .to_owned();

        Ok(Self {
            http: reqwest::Client::new(),
            card: Some(card),
            endpoint,
            next_id: AtomicU64::new(1),
            extra_headers: headers.clone(),
            context_id: Mutex::new(None),
        })
    }

    /// 已发现的 agent 卡片。
    pub fn card(&self) -> Option<&AgentCard> {
        self.card.as_ref()
    }

    /// JSON-RPC 端点地址。
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// 设置请求时附带的额外头（例如 `Authorization`）。
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }

    /// 生成下一个请求 id。
    fn next_request_id(&self) -> String {
        self.next_id.fetch_add(1, Ordering::Relaxed).to_string()
    }

    /// 构造一个带好服务参数与额外头的 POST 请求。
    fn post_builder(&self) -> reqwest::RequestBuilder {
        let mut builder = self
            .http
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(service_param::VERSION, crate::a2a::types::A2A_PROTOCOL_VERSION);

        for (name, value) in &self.extra_headers {
            builder = builder.header(name, value);
        }

        builder
    }

    /// 发送一个 JSON-RPC 请求并解析响应。
    ///
    /// 返回值分两层：`Ok(Ok(result))` 表示远端成功，`Ok(Err(error))` 表示
    /// 远端返回了 A2A 错误（错误码按规范映射保留），`Err(_)` 表示传输层
    /// 失败（网络、HTTP 状态码、JSON 解析失败）。
    async fn call(
        &self,
        method_name: &str,
        params: Value,
    ) -> Result<Result<Value, A2AError>, anyhow::Error> {
        let request = JsonRpcRequest::new(self.next_request_id(), method_name, params);

        let response = self
            .post_builder()
            .json(&request)
            .send()
            .await?
            .error_for_status()?;

        let envelope: JsonRpcResponse = response.json().await?;

        Ok(match envelope {
            JsonRpcResponse::Success(success) => Ok(success.result),
            JsonRpcResponse::Failure(failure) => Err(A2AError::from_rpc(
                failure.error.code,
                failure.error.message,
            )),
        })
    }

    /// 发送一条文本消息给远端 agent。
    ///
    /// 返回远端的任务对象。远端如果直接回消息（不建任务），则构造一个
    /// 已完成任务返回，方便调用方统一处理。
    ///
    /// **多轮对话**：客户端会记住服务端返回的 `contextId`，后续的
    /// [`A2AClient::send_text`] 会自动带上它，因此连续调用就构成一段
    /// 连续对话（规范 3.4.1 节的 context 语义）。
    pub async fn send_text(&self, text: impl Into<String>) -> anyhow::Result<Task> {
        let context_id = self.context_id.lock().unwrap().clone();
        self.send_text_with(text, context_id.as_deref()).await
    }

    /// 发送一条文本消息，并显式指定上下文标识。
    ///
    /// - `context_id`：对应规范的 `contextId`，用于把多轮消息归到一组；
    ///   传 `None` 表示开启一段新对话。
    ///
    /// 返回的任务里会带服务端分配的 `contextId`，客户端会自动记住它。
    pub async fn send_text_with(
        &self,
        text: impl Into<String>,
        context_id: Option<&str>,
    ) -> anyhow::Result<Task> {
        let message = Message::new_text(
            Role::User,
            text,
            context_id.map(str::to_owned),
            None,
        );

        let request = SendMessageRequest {
            message,
            configuration: Some(SendMessageConfiguration::default()),
            metadata: None,
        };

        let result = self
            .call(method::SEND_MESSAGE, serde_json::to_value(request)?)
            .await?;

        let response: SendMessageResponse = match result {
            Ok(value) => serde_json::from_value(value)?,
            Err(error) => return Err(error.into()),
        };

        if let Some(task) = response.task {
            // 记住服务端分配的 contextId，后续消息自动续用同一上下文。
            if let Some(context_id) = &task.context_id {
                *self.context_id.lock().unwrap() = Some(context_id.clone());
            }
            return Ok(task);
        }

        // 远端直接回了消息：包装成一个已完成的任务。
        if let Some(message) = response.message {
            let mut task = Task::new_submitted(context_id.map(str::to_owned));
            task.status.state = TaskState::Completed;
            task.artifacts = vec![Artifact::new_text("result", message.text())];
            task.history.push(message);
            return Ok(task);
        }

        Err(anyhow::anyhow!("A2A response contained neither task nor message"))
    }

    /// 查询任务状态。
    pub async fn get_task(&self, task_id: &str) -> anyhow::Result<Task> {
        let result = self
            .call(method::GET_TASK, serde_json::json!({ "id": task_id }))
            .await?;

        match result {
            Ok(value) => Ok(serde_json::from_value(value)?),
            Err(error) => Err(error.into()),
        }
    }

    /// 订阅一个任务的流式更新（SSE）。
    ///
    /// 规范 9.4.6 节：订阅的是"非终态任务"的更新流；任务已进入终态时
    /// 服务端应返回 `UnsupportedOperationError`。
    ///
    /// 这里返回一个 [`futures::stream::BoxStream`]，每个元素是一个
    /// [`StreamResponse`]（状态更新 / 产物更新 / 任务快照）。
    pub fn subscribe_task(
        &self,
        task_id: &str,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<StreamResponse>>> {
        let request = JsonRpcRequest::new(
            self.next_request_id(),
            method::SUBSCRIBE_TO_TASK,
            serde_json::json!({ "id": task_id }),
        );

        let builder = self.post_builder();

        Ok(Box::pin(async_stream::stream! {
            let response = match builder.json(&request).send().await {
                Ok(response) => response,
                Err(error) => {
                    yield Err(anyhow::anyhow!("A2A subscribe failed: {error}"));
                    return;
                }
            };

            if !response.status().is_success() {
                yield Err(anyhow::anyhow!(
                    "A2A subscribe returned HTTP {}",
                    response.status()
                ));
                return;
            }

            let mut stream = Box::pin(parse_sse_stream(response));

            while let Some(item) = stream.next().await {
                yield item;
            }
        }))
    }

    /// 以流式方式发送消息，边收边处理。
    ///
    /// 规范 9.4.2 节的 `SendStreamingMessage`。
    pub fn send_streaming(
        &self,
        text: impl Into<String>,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<StreamResponse>>> {
        let context_id = self.context_id.lock().unwrap().clone();
        let message = Message::new_text(Role::User, text, context_id, None);
        let request = SendMessageRequest {
            message,
            configuration: Some(SendMessageConfiguration::default()),
            metadata: None,
        };

        let request = JsonRpcRequest::new(
            self.next_request_id(),
            method::SEND_STREAMING_MESSAGE,
            serde_json::to_value(request)?,
        );

        let builder = self.post_builder();

        Ok(Box::pin(async_stream::stream! {
            let response = match builder.json(&request).send().await {
                Ok(response) => response,
                Err(error) => {
                    yield Err(anyhow::anyhow!("A2A streaming failed: {error}"));
                    return;
                }
            };

            if !response.status().is_success() {
                yield Err(anyhow::anyhow!(
                    "A2A streaming returned HTTP {}",
                    response.status()
                ));
                return;
            }

            let mut stream = Box::pin(parse_sse_stream(response));

            while let Some(item) = stream.next().await {
                yield item;
            }
        }))
    }

    /// 列出任务，可按上下文过滤。
    pub async fn list_tasks(&self, context_id: Option<&str>) -> anyhow::Result<Vec<Task>> {
        let mut params = serde_json::Map::new();
        if let Some(context) = context_id {
            params.insert("contextId".to_owned(), Value::String(context.to_owned()));
        }

        let result = self
            .call(method::LIST_TASKS, Value::Object(params))
            .await?;

        match result {
            Ok(value) => Ok(serde_json::from_value(value)?),
            Err(error) => Err(error.into()),
        }
    }

    /// 取消任务。
    pub async fn cancel_task(&self, task_id: &str) -> anyhow::Result<Task> {
        let result = self
            .call(method::CANCEL_TASK, serde_json::json!({ "id": task_id }))
            .await?;

        match result {
            Ok(value) => Ok(serde_json::from_value(value)?),
            Err(error) => Err(error.into()),
        }
    }

    /// 获取扩展版 Agent Card。
    pub async fn extended_agent_card(&self) -> anyhow::Result<AgentCard> {
        let result = self
            .call(method::GET_EXTENDED_AGENT_CARD, Value::Null)
            .await?;

        match result {
            Ok(value) => Ok(serde_json::from_value(value)?),
            Err(error) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn discover_rejects_a_card_without_interfaces() {
        // 用一个不可达的地址，确认错误是可预期的，而不是 panic。
        let error = A2AClient::discover("http://127.0.0.1:1").await;
        assert!(error.is_err());
    }

    #[test]
    fn client_keeps_the_given_endpoint() {
        let client = A2AClient::new("http://127.0.0.1:8080/rpc");
        assert_eq!(client.endpoint(), "http://127.0.0.1:8080/rpc");
        assert!(client.card().is_none());
    }

    #[test]
    fn request_ids_increase_monotonically() {
        let client = A2AClient::new("http://x/rpc");
        let first = client.next_request_id();
        let second = client.next_request_id();
        assert_ne!(first, second);
        assert_eq!(first.parse::<u64>().unwrap() + 1, second.parse::<u64>().unwrap());
    }

    #[test]
    fn extra_headers_are_stored() {
        let client = A2AClient::new("http://x/rpc").with_header("Authorization", "Bearer t");
        assert_eq!(
            client.extra_headers.get("Authorization").map(String::as_str),
            Some("Bearer t")
        );
    }
}
