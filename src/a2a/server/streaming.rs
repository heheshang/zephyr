//! A2A 服务端的流式（SSE）响应。
//!
//! # 规范依据
//!
//! 规范 9.4.2 节：流式方法用 SSE 返回，`Content-Type: text/event-stream`，
//! 每个事件形如 `data: {jsonrpc envelope}`。
//!
//! # 两层信封
//!
//! ```text
//!   SSE 帧                JSON-RPC 信封            内层
//!   ──────                ─────────────            ────
//!   data: {"jsonrpc":     {"jsonrpc":"2.0",
//!         "2.0",   ──►    "id":"1",          ──►    "statusUpdate":{…}}
//!         "id":"1",
//!         "result":{…}}
//! ```
//!
//! 信封在发送端就已包好，客户端只需剥两层。

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc;

use crate::{
    a2a::{
        error::A2AError,
        jsonrpc::{JsonRpcRequest, JsonRpcResponse, method, ServiceParams},
        server::{A2AServerState, dispatch, executor},
        task_store::{TaskLifecycle, merge_or_create_task},
        types::{
            SendMessageRequest, StreamResponse, TaskArtifactUpdateEvent, TaskStatusUpdateEvent,
        },
    },
};

/// 判断一个方法是否走 SSE。
pub fn is_streaming_method(method_name: &str) -> bool {
    method_name == method::SEND_STREAMING_MESSAGE || method_name == method::SUBSCRIBE_TO_TASK
}

/// 流式方法的响应：SSE。
pub fn streaming_response(
    state: Arc<A2AServerState>,
    request: JsonRpcRequest,
    service_params: ServiceParams,
) -> Response {
    let id = request.id.clone();

    // 能力校验：规范 3.3.4 节，未声明 streaming 就必须报错。
    if !state.card.capabilities.streaming.unwrap_or(false) {
        let failure =
            JsonRpcResponse::failure(id, A2AError::UnsupportedOperation("streaming".to_owned()));
        return dispatch::json_response(failure, StatusCode::BAD_REQUEST);
    }

    let (tx, mut rx) = mpsc::channel::<JsonRpcResponse>(16);

    tokio::spawn(async move {
        // 先发一个任务快照，客户端立刻知道 taskId。
        let params = request.params.clone().unwrap_or(serde_json::Value::Null);
        let task_id = params
            .get("message")
            .and_then(|message| message.get("taskId"))
            .and_then(|value| value.as_str())
            .map(str::to_owned);

        if let Some(task_id) = task_id
            && let Some(task) = state.task_store.get(&task_id).await
        {
            let _ = tx
                .send(JsonRpcResponse::success(
                    id.clone(),
                    serde_json::to_value(StreamResponse::Task { task }).unwrap_or_default(),
                ))
                .await;
        }

        // 再跑 agent，把状态与产物作为事件推送。
        match serde_json::from_value::<SendMessageRequest>(params) {
            Ok(send_request) => {
                let message = send_request.message;
                let context_id = message.context_id.clone();
                let existing = match &context_id {
                    Some(context) => state
                        .task_store
                        .list_by_context(context)
                        .await
                        .into_iter()
                        .find(|task| !task.is_terminal()),
                    None => None,
                };
                let mut task = merge_or_create_task(existing, message.clone(), context_id.clone());
                executor::ensure_context_id(&mut task);

                let working = TaskStatusUpdateEvent {
                    task_id: task.id.clone(),
                    context_id: task.context_id.clone().unwrap_or_default(),
                    status: TaskLifecycle::working(),
                    metadata: None,
                };
                let _ = tx
                    .send(JsonRpcResponse::success(
                        id.clone(),
                        serde_json::to_value(StreamResponse::StatusUpdate {
                            status_update: working,
                        })
                        .unwrap_or_default(),
                    ))
                    .await;

                match executor::run_agent(&state, &task, &message).await {
                    Ok(outcome) => {
                        task.artifacts = outcome.artifacts;
                        task.status = outcome.status;

                        for artifact in task.artifacts.clone() {
                            let event = StreamResponse::ArtifactUpdate {
                                artifact_update: TaskArtifactUpdateEvent {
                                    task_id: task.id.clone(),
                                    context_id: task.context_id.clone().unwrap_or_default(),
                                    artifact,
                                    metadata: None,
                                },
                            };
                            let _ = tx
                                .send(JsonRpcResponse::success(
                                    id.clone(),
                                    serde_json::to_value(&event).unwrap_or_default(),
                                ))
                                .await;
                        }

                        let final_event = StreamResponse::StatusUpdate {
                            status_update: TaskStatusUpdateEvent {
                                task_id: task.id.clone(),
                                context_id: task.context_id.clone().unwrap_or_default(),
                                status: task.status.clone(),
                                metadata: None,
                            },
                        };
                        let _ = tx
                            .send(JsonRpcResponse::success(
                                id.clone(),
                                serde_json::to_value(&final_event).unwrap_or_default(),
                            ))
                            .await;

                        let _ = state.task_store.save(task).await;
                    }
                    Err(error) => {
                        let event = StreamResponse::StatusUpdate {
                            status_update: TaskStatusUpdateEvent {
                                task_id: task.id.clone(),
                                context_id: task.context_id.clone().unwrap_or_default(),
                                status: TaskLifecycle::failed(error.to_string()),
                                metadata: None,
                            },
                        };
                        let _ = tx
                            .send(JsonRpcResponse::success(
                                id.clone(),
                                serde_json::to_value(&event).unwrap_or_default(),
                            ))
                            .await;
                    }
                }
            }
            Err(error) => {
                let failure = JsonRpcResponse::failure(
                    id.clone(),
                    A2AError::InvalidParams(format!("invalid streaming params: {error}")),
                );
                let _ = tx.send(failure).await;
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            // 规范 9.4.2 节：SSE 的每个事件都是完整的 JSON-RPC 信封，
            // 内层的 result 才是 StreamResponse。信封在发送端就已包好。
            let text = serde_json::to_string(&event).unwrap_or_default();

            yield Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from(format!("data: {text}\n\n")),
            );
        }
    };

    let _ = service_params;

    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .header(axum::http::header::CONNECTION, "keep-alive")
        .body(axum::body::Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}
