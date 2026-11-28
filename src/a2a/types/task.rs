//! 任务与任务状态。
//!
//! 规范 4.1.1 ~ 4.1.3 节。

use serde::{Deserialize, Serialize};

use super::message::{Message, Role};

/// 任务的生命周期状态。
///
/// 规范 4.1.3 节。终态（terminal）有 `Completed` / `Failed` / `Canceled` /
/// `Rejected`；中断态（interrupted）有 `InputRequired` / `AuthRequired`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// 未指定。
    #[serde(rename = "TASK_STATE_UNSPECIFIED")]
    Unspecified,
    /// 已提交，尚未开始处理。
    #[serde(rename = "TASK_STATE_SUBMITTED")]
    Submitted,
    /// 正在处理。
    #[serde(rename = "TASK_STATE_WORKING")]
    Working,
    /// 已完成。
    #[serde(rename = "TASK_STATE_COMPLETED")]
    Completed,
    /// 执行失败。
    #[serde(rename = "TASK_STATE_FAILED")]
    Failed,
    /// 已被取消。
    #[serde(rename = "TASK_STATE_CANCELED")]
    Canceled,
    /// 需要用户补充输入才能继续。
    #[serde(rename = "TASK_STATE_INPUT_REQUIRED")]
    InputRequired,
    /// 被拒绝。
    #[serde(rename = "TASK_STATE_REJECTED")]
    Rejected,
    /// 需要用户完成认证才能继续。
    #[serde(rename = "TASK_STATE_AUTH_REQUIRED")]
    AuthRequired,
}

impl TaskState {
    /// 是否为终态。
    ///
    /// 终态任务不会再变化，客户端可以停止轮询或订阅。
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
        )
    }

    /// 是否为中断态。
    ///
    /// 中断态表示任务需要额外输入或认证才能继续，例如 agent 反问用户。
    pub fn is_interrupted(&self) -> bool {
        matches!(self, TaskState::InputRequired | TaskState::AuthRequired)
    }

    /// 按规范要求序列化为 SCREAMING_SNAKE_CASE 字符串。
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskState::Unspecified => "TASK_STATE_UNSPECIFIED",
            TaskState::Submitted => "TASK_STATE_SUBMITTED",
            TaskState::Working => "TASK_STATE_WORKING",
            TaskState::Completed => "TASK_STATE_COMPLETED",
            TaskState::Failed => "TASK_STATE_FAILED",
            TaskState::Canceled => "TASK_STATE_CANCELED",
            TaskState::InputRequired => "TASK_STATE_INPUT_REQUIRED",
            TaskState::Rejected => "TASK_STATE_REJECTED",
            TaskState::AuthRequired => "TASK_STATE_AUTH_REQUIRED",
        }
    }
}

/// 任务的状态快照。
///
/// 规范 4.1.2 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    /// 当前状态。
    pub state: TaskState,
    /// 与该状态关联的消息（例如 agent 反问用户时的那条）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    /// 状态记录时间，ISO 8601 UTC。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl TaskStatus {
    /// 构造一个只带状态的时间戳快照。
    pub fn new(state: TaskState) -> Self {
        Self {
            state,
            message: None,
            timestamp: Some(super::now_iso8601()),
        }
    }

    /// 构造一个带消息的状态快照。
    pub fn with_message(state: TaskState, message: Message) -> Self {
        Self {
            state,
            message: Some(message),
            timestamp: Some(super::now_iso8601()),
        }
    }

    /// 构造一个带 agent 消息的状态快照。
    pub fn with_agent_note(state: TaskState, text: impl Into<String>) -> Self {
        Self::with_message(state, Message::new_text(Role::Agent, text, None, None))
    }
}

/// 任务的产物。
///
/// 规范 4.1.7 节。`artifactId` 在同一个任务内必须唯一。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// 产物唯一标识。
    #[serde(rename = "artifactId")]
    pub artifact_id: String,
    /// 人类可读名称。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 人类可读描述。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 产物内容，至少一个片段。
    pub parts: Vec<crate::a2a::types::message::Part>,
    /// 扩展元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Artifact {
    /// 构造一个文本产物。
    pub fn new_text(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            artifact_id: uuid::Uuid::new_v4().to_string(),
            name: Some(name.into()),
            description: None,
            parts: vec![crate::a2a::types::message::Part::new_text(text)],
            metadata: None,
        }
    }
}

/// A2A 的核心单元：任务。
///
/// 规范 4.1.1 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// 任务唯一标识，由服务端生成。
    pub id: String,
    /// 上下文标识。
    #[serde(rename = "contextId", skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// 当前状态。
    pub status: TaskStatus,
    /// 产物列表。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
    /// 交互历史。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Message>,
    /// 扩展元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Task {
    /// 创建一个刚提交、尚未开始处理的任务。
    pub fn new_submitted(context_id: Option<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            context_id,
            status: TaskStatus::new(TaskState::Submitted),
            artifacts: Vec::new(),
            history: Vec::new(),
            metadata: None,
        }
    }

    /// 是否已进入终态。
    pub fn is_terminal(&self) -> bool {
        self.status.state.is_terminal()
    }

    /// 任务当前状态。
    pub fn state(&self) -> TaskState {
        self.status.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_values_use_screaming_snake_case() {
        assert_eq!(
            serde_json::to_string(&TaskState::InputRequired).unwrap(),
            "\"TASK_STATE_INPUT_REQUIRED\""
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Working).unwrap(),
            "\"TASK_STATE_WORKING\""
        );
    }

    #[test]
    fn task_state_terminal_and_interrupted_classification() {
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(TaskState::Canceled.is_terminal());
        assert!(TaskState::Rejected.is_terminal());
        assert!(!TaskState::Working.is_terminal());

        assert!(TaskState::InputRequired.is_interrupted());
        assert!(TaskState::AuthRequired.is_interrupted());
        assert!(!TaskState::Submitted.is_interrupted());
    }

    #[test]
    fn task_serializes_with_camel_case_and_omits_empty_history() {
        let mut task = Task::new_submitted(Some("ctx".into()));
        task.status.state = TaskState::Completed;
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("\"contextId\""), "{json}");
        assert!(json.contains("\"TASK_STATE_COMPLETED\""), "{json}");
        assert!(!json.contains("\"history\""), "empty history should be omitted: {json}");
    }
}
