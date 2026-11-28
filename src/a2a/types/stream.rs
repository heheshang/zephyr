//! 流式事件与 `SendMessage` 的请求/响应模型。
//!
//! 规范 3.2 节（请求配置）与 4.2 节（流式事件）。

use serde::{Deserialize, Serialize};

use super::{
    message::Message,
    task::{Artifact, Task, TaskStatus},
};

/// 流式事件：任务状态发生变化。
///
/// 规范 4.2.1 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatusUpdateEvent {
    /// 任务标识。
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// 上下文标识。
    #[serde(rename = "contextId")]
    pub context_id: String,
    /// 新状态。
    pub status: TaskStatus,
    /// 扩展元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// 流式事件：产生了新的产物。
///
/// 规范 4.2.2 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskArtifactUpdateEvent {
    /// 任务标识。
    #[serde(rename = "taskId")]
    pub task_id: String,
    /// 上下文标识。
    #[serde(rename = "contextId")]
    pub context_id: String,
    /// 产物增量。
    pub artifact: Artifact,
    /// 扩展元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// 流式响应的包装类型。
///
/// 规范 3.2.3 节：`task` / `message` / `statusUpdate` / `artifactUpdate`
/// 是 OneOf，**必须且只能**有一个被设置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamResponse {
    /// 任务快照。
    Task {
        /// 任务快照。
        task: Task,
    },
    /// agent 直接返回的一条消息。
    Message {
        /// agent 消息。
        message: Message,
    },
    /// 状态更新事件。
    StatusUpdate {
        /// 状态更新。
        #[serde(rename = "statusUpdate")]
        status_update: TaskStatusUpdateEvent,
    },
    /// 产物更新事件。
    ArtifactUpdate {
        /// 产物更新。
        #[serde(rename = "artifactUpdate")]
        artifact_update: TaskArtifactUpdateEvent,
    },
}

/// `SendMessage` 的配置对象。
///
/// 规范 3.2.2 节。`returnImmediately` 决定是阻塞等待终态还是立即返回。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageConfiguration {
    /// 客户端可接受的输出媒体类型。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepted_output_modes: Vec<String>,
    /// 返回的历史消息条数上限。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_length: Option<usize>,
    /// 是否在创建任务后立即返回。
    ///
    /// `false`（默认）表示阻塞，直到任务进入终态或中断态。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_immediately: Option<bool>,
}

/// `SendMessage` 的请求参数。
///
/// 规范 3.2.1 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageRequest {
    /// 要发送的消息。
    pub message: Message,
    /// 配置对象。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<SendMessageConfiguration>,
    /// 扩展元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// `SendMessage` 的响应。
///
/// 规范 9.4.1 节：响应包含 `task` 或 `message` 之一。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageResponse {
    /// 任务对象。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<Task>,
    /// agent 直接回复的消息。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
}

impl SendMessageResponse {
    /// 构造一个任务响应。
    pub fn from_task(task: Task) -> Self {
        Self {
            task: Some(task),
            message: None,
        }
    }

    /// 构造一个消息响应。
    pub fn from_message(message: Message) -> Self {
        Self {
            task: None,
            message: Some(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_message_configuration_uses_camel_case() {
        let config = SendMessageConfiguration {
            accepted_output_modes: vec!["text/plain".into()],
            history_length: Some(5),
            return_immediately: Some(true),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"acceptedOutputModes\""), "{json}");
        assert!(json.contains("\"historyLength\""), "{json}");
        assert!(json.contains("\"returnImmediately\""), "{json}");
    }

    #[test]
    fn stream_response_is_one_of() {
        let task = Task::new_submitted(None);
        let response = StreamResponse::Task { task };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.starts_with('{'));
        assert!(json.contains("\"task\""), "{json}");
    }
}
