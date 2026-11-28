//! A2A 任务的生命周期状态机。
//!
//! # 状态机
//!
//! ```text
//!   Submitted ──► Working ──┬─► Completed
//!                            ├─► Failed
//!                            ├─► Canceled
//!                            └─► InputRequired
//! ```
//!
//! 规范 4.1.3 节定义了这几个状态。这里把它们收敛成 [`TaskLifecycle`]
//! 上的几个构造器，避免各处随手写 `TaskState::…`，也避免漏掉时间戳。
//!
//! 本模块只放**纯函数**——不碰存储，不碰 I/O。这样状态机的规则可以被
//! 单独推演与测试。

use crate::a2a::{
    error::A2AError,
    types::{Artifact, Message, Role, Task, TaskState, TaskStatus},
};

/// 任务执行结果的记录助手。
///
/// 把 A2A 的状态机收敛到几个明确的转移上，避免各处随手写 `TaskState`。
pub struct TaskLifecycle;

impl TaskLifecycle {
    /// 任务创建时的初始状态。
    pub fn initial() -> TaskStatus {
        TaskStatus::new(TaskState::Submitted)
    }

    /// 进入工作状态。
    pub fn working() -> TaskStatus {
        TaskStatus::new(TaskState::Working)
    }

    /// 进入完成状态。
    pub fn completed() -> TaskStatus {
        TaskStatus::new(TaskState::Completed)
    }

    /// 进入失败状态，并附带错误说明。
    pub fn failed(reason: impl Into<String>) -> TaskStatus {
        TaskStatus::with_message(
            TaskState::Failed,
            Message::new_text(Role::Agent, reason, None, None),
        )
    }

    /// 进入需要输入的状态，并附带向用户提出的问题。
    pub fn input_required(question: impl Into<String>) -> TaskStatus {
        TaskStatus::with_message(
            TaskState::InputRequired,
            Message::new_text(Role::Agent, question, None, None),
        )
    }

    /// 把一段历史消息追加到任务上。
    pub fn append_history(task: &mut Task, message: Message) {
        task.history.push(message);
    }
}

/// 把客户端发来的消息合并进已有任务，或新建任务。
///
/// 规范 3.4.2 节讲的是"任务标识语义"：客户端可以在后续消息里带上
/// `taskId` 继续同一个任务；不带 `taskId` 但带 `contextId` 时，表示在同一个
/// 上下文里开启一个新任务。
pub fn merge_or_create_task(
    existing: Option<Task>,
    message: Message,
    context_id: Option<String>,
) -> Task {
    match existing {
        Some(mut task) => {
            TaskLifecycle::append_history(&mut task, message);
            task
        }
        None => {
            let mut task = Task::new_submitted(context_id.or(message.context_id.clone()));
            TaskLifecycle::append_history(&mut task, message);
            task
        }
    }
}

/// 按 `historyLength` 语义裁剪历史。
///
/// 规范 3.2.4 节：
///
/// - `None`：不施加限制，返回服务端默认量；
/// - `Some(0)`：不返回历史；
/// - `Some(n)`：最多返回最近 n 条。
pub fn truncate_history(task: &mut Task, history_length: Option<usize>) {
    match history_length {
        Some(0) => task.history.clear(),
        Some(length) if task.history.len() > length => {
            let start = task.history.len() - length;
            task.history = task.history.split_off(start);
        }
        _ => {}
    }
}

/// 校验任务是否可以取消。
///
/// 规范 3.1.5 节：只有非终态任务可以取消。
pub fn ensure_cancelable(task: &Task) -> Result<(), A2AError> {
    if task.status.state.is_terminal() {
        Err(A2AError::TaskNotCancelable(task.id.clone()))
    } else {
        Ok(())
    }
}

/// 把任务的产物拼成一段纯文本。
///
/// 客户端工具用它把远端 agent 的回复交给本地协调员。
pub fn artifacts_to_text(artifacts: &[Artifact]) -> String {
    artifacts
        .iter()
        .map(|artifact| {
            artifact
                .parts
                .iter()
                .map(|part| part.collect_text())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(text: &str) -> Message {
        Message::new_text(Role::User, text, None, None)
    }

    #[test]
    fn merge_or_create_appends_to_existing_task() {
        let mut existing = Task::new_submitted(Some("ctx".into()));
        existing.history.push(message("first"));

        let merged = merge_or_create_task(Some(existing), message("second"), None);
        assert_eq!(merged.history.len(), 2);
        assert_eq!(merged.history[1].text(), "second");
    }

    #[test]
    fn merge_or_create_builds_new_task_when_absent() {
        let created = merge_or_create_task(None, message("hello"), Some("ctx".into()));
        assert_eq!(created.state(), TaskState::Submitted);
        assert_eq!(created.context_id.as_deref(), Some("ctx"));
        assert_eq!(created.history.len(), 1);
    }

    #[test]
    fn truncate_history_respects_length_semantics() {
        let mut task = Task::new_submitted(None);
        for index in 0..5 {
            task.history.push(message(&format!("m{index}")));
        }

        let mut unlimited = task.clone();
        truncate_history(&mut unlimited, None);
        assert_eq!(unlimited.history.len(), 5);

        let mut zero = task.clone();
        truncate_history(&mut zero, Some(0));
        assert!(zero.history.is_empty());

        let mut two = task.clone();
        truncate_history(&mut two, Some(2));
        assert_eq!(two.history.len(), 2);
        assert_eq!(two.history[1].text(), "m4");
    }

    #[test]
    fn terminal_tasks_cannot_be_canceled() {
        let mut task = Task::new_submitted(None);
        assert!(ensure_cancelable(&task).is_ok());

        task.status.state = TaskState::Completed;
        let error = ensure_cancelable(&task).unwrap_err();
        assert_eq!(error.code(), -32002);
    }
}
