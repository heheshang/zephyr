//! A2A 服务端的方法分派。
//!
//! # 分派流程
//!
//! ```text
//!   JSON-RPC 请求
//!     │
//!     ├──▶ 校验 jsonrpc 版本（规范 9.3）
//!     ├──▶ 校验 A2A-Version 头（规范 3.6）
//!     │
//!     └──▶ 按 method 匹配
//!            ├── SendMessage           ──▶ handle_send_message
//!            ├── GetTask               ──▶ handle_get_task
//!            ├── ListTasks             ──▶ handle_list_tasks
//!            ├── CancelTask            ──▶ handle_cancel_task
//!            ├── GetExtendedAgentCard  ──▶ 直接回卡片
//!            └── 其他                  ──▶ MethodNotFound（-32601）
//! ```
//!
//! 业务逻辑按方法拆成独立函数，公共部分（把任务存回 store、拼装响应）
//! 收敛在少数几个 helper 里。

use std::sync::Arc;

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;

use crate::{
    a2a::{
        error::A2AError,
        jsonrpc::{JsonRpcRequest, JsonRpcResponse, method, ServiceParams},
        server::A2AServerState,
        task_store::{
            TaskLifecycle, merge_or_create_task, truncate_history,
        },
        types::{SendMessageRequest, SendMessageResponse, TaskState},
    },
};

/// 把 JSON-RPC 响应包成 HTTP 响应。
pub fn json_response(response: JsonRpcResponse, status: StatusCode) -> Response {
    (status, Json(response)).into_response()
}

/// 按 `method` 分派到具体处理逻辑。
pub async fn dispatch(
    state: &Arc<A2AServerState>,
    request: JsonRpcRequest,
    service_params: ServiceParams,
) -> JsonRpcResponse {
    let id = request.id.clone();

    if let Err(error) = request.validate_version() {
        return JsonRpcResponse::failure(id, error);
    }

    // 版本协商：规范 3.6 节要求服务端在收到不兼容版本时明确报错。
    if let Some(version) = &service_params.version
        && !version.starts_with(crate::a2a::types::A2A_PROTOCOL_VERSION)
    {
        return JsonRpcResponse::failure(id, A2AError::VersionNotSupported(version.clone()));
    }
    let params = request.params.clone().unwrap_or(serde_json::Value::Null);

    match request.method.as_str() {
        method::SEND_MESSAGE => handle_send_message(state, id, params).await,
        method::GET_TASK => handle_get_task(state, id, params).await,
        method::LIST_TASKS => handle_list_tasks(state, id, params).await,
        method::CANCEL_TASK => handle_cancel_task(state, id, params).await,
        method::GET_EXTENDED_AGENT_CARD => {
            JsonRpcResponse::success(id, serde_json::to_value(&state.card).unwrap_or_default())
        }
        other => JsonRpcResponse::failure(id, A2AError::MethodNotFound(other.to_owned())),
    }
}

/// `SendMessage` 的处理逻辑。
pub async fn handle_send_message(
    state: &Arc<A2AServerState>,
    id: Option<serde_json::Value>,
    params: serde_json::Value,
) -> JsonRpcResponse {
    let request: SendMessageRequest = match serde_json::from_value(params) {
        Ok(request) => request,
        Err(error) => {
            return JsonRpcResponse::failure(
                id,
                A2AError::InvalidParams(format!("invalid SendMessage params: {error}")),
            );
        }
    };

    // 消息内容校验：规范 4.1.2 节要求消息至少包含一个 part，
    // 否则 agent 无从知道要处理什么。
    if request.message.parts.is_empty() {
        return JsonRpcResponse::failure(
            id,
            A2AError::InvalidParams("message must contain at least one part".to_owned()),
        );
    }

    let message = request.message;
    let context_id = message.context_id.clone();

    // 查找同一上下文下的已有任务。
    //
    // 规范 3.4.2 节：同一 `contextId` 下的多轮消息构成一段连续对话，
    // 后续消息应当**延续**之前的任务（继承其历史），而不是另起一段。
    //
    // 因此这里取该 context 下**最新的**任务（不论是否终态）：
    // 若只找非终态任务，已完成的任务会被丢弃，历史随之丢失。
    let existing = match &context_id {
        Some(context) => state
            .task_store
            .list_by_context(context)
            .await
            .into_iter()
            .max_by_key(|task| task.history.len()),
        None => None,
    };

    let mut task = merge_or_create_task(existing, message.clone(), context_id.clone());
    crate::a2a::server::executor::ensure_context_id(&mut task);

    task.status = TaskLifecycle::working();

    let return_immediately = request
        .configuration
        .as_ref()
        .and_then(|config| config.return_immediately)
        .unwrap_or(false);

    if let Err(error) = state.task_store.save(task.clone()).await {
        return JsonRpcResponse::failure(id, A2AError::Internal(error.to_string()));
    }

    // 立即返回模式：任务已经在工作状态，客户端可以随后轮询。
    if return_immediately {
        let response = SendMessageResponse::from_task(task);
        return JsonRpcResponse::success(id, serde_json::to_value(response).unwrap_or_default());
    }

    // 阻塞模式：跑完 agent 再返回。
    let outcome = crate::a2a::server::executor::run_agent(state, &task, &message).await;

    match outcome {
        Ok(result) => {
            task.artifacts = result.artifacts;
            task.status = result.status;
            if let Some(history_length) = request
                .configuration
                .as_ref()
                .and_then(|config| config.history_length)
            {
                truncate_history(&mut task, Some(history_length));
            }
            if let Err(error) = state.task_store.save(task.clone()).await {
                return JsonRpcResponse::failure(id, A2AError::Internal(error.to_string()));
            }
            let response = SendMessageResponse::from_task(task);
            JsonRpcResponse::success(id, serde_json::to_value(response).unwrap_or_default())
        }
        Err(error) => {
            task.status = TaskLifecycle::failed(error.to_string());
            let _ = state.task_store.save(task.clone()).await;
            JsonRpcResponse::failure(id, A2AError::Internal(error.to_string()))
        }
    }
}

/// `GetTask` 的处理逻辑。
pub async fn handle_get_task(
    state: &Arc<A2AServerState>,
    id: Option<serde_json::Value>,
    params: serde_json::Value,
) -> JsonRpcResponse {
    #[derive(Deserialize)]
    struct GetTaskParams {
        id: String,
        #[serde(default)]
        history_length: Option<usize>,
    }

    let params: GetTaskParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return JsonRpcResponse::failure(
                id,
                A2AError::InvalidParams(format!("invalid GetTask params: {error}")),
            );
        }
    };

    match state.task_store.get(&params.id).await {
        Some(mut task) => {
            truncate_history(&mut task, params.history_length);
            JsonRpcResponse::success(id, serde_json::to_value(task).unwrap_or_default())
        }
        None => JsonRpcResponse::failure(id, A2AError::TaskNotFound(params.id)),
    }
}

/// `ListTasks` 的处理逻辑。
pub async fn handle_list_tasks(
    state: &Arc<A2AServerState>,
    id: Option<serde_json::Value>,
    params: serde_json::Value,
) -> JsonRpcResponse {
    #[derive(Debug, Default, Deserialize)]
    struct ListTasksParams {
        #[serde(default)]
        context_id: Option<String>,
        #[serde(default)]
        status: Option<String>,
    }

    let params: ListTasksParams = serde_json::from_value(params).unwrap_or_default();

    let mut tasks = match &params.context_id {
        Some(context) => state.task_store.list_by_context(context).await,
        None => state.task_store.list_all().await,
    };

    if let Some(status) = params.status {
        tasks.retain(|task| task.status.state.as_str() == status);
    }

    JsonRpcResponse::success(id, serde_json::to_value(tasks).unwrap_or_default())
}

/// `CancelTask` 的处理逻辑。
pub async fn handle_cancel_task(
    state: &Arc<A2AServerState>,
    id: Option<serde_json::Value>,
    params: serde_json::Value,
) -> JsonRpcResponse {
    #[derive(Deserialize)]
    struct CancelTaskParams {
        id: String,
    }

    let params: CancelTaskParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return JsonRpcResponse::failure(
                id,
                A2AError::InvalidParams(format!("invalid CancelTask params: {error}")),
            );
        }
    };

    let task = match state.task_store.get(&params.id).await {
        Some(task) => task,
        None => return JsonRpcResponse::failure(id, A2AError::TaskNotFound(params.id)),
    };

    if let Err(error) = crate::a2a::task_store::ensure_cancelable(&task) {
        return JsonRpcResponse::failure(id, error);
    }

    let mut canceled = task;
    canceled.status.state = TaskState::Canceled;
    if let Err(error) = state.task_store.save(canceled.clone()).await {
        return JsonRpcResponse::failure(id, A2AError::Internal(error.to_string()));
    }

    JsonRpcResponse::success(id, serde_json::to_value(canceled).unwrap_or_default())
}
