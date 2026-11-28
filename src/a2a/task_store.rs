//! A2A 任务存储。
//!
//! # 为什么需要单独的存储层
//!
//! A2A 的 `Task` 是**跨请求、跨进程**的生命周期实体：客户端发一条消息创建任务，
//! 之后可以用 `GetTask` 查状态、用 `CancelTask` 取消它。而项目里
//! [`crate::session::model::Session`] 是"一次对话的上下文"，语义相近但不完全一致：
//!
//! - Session 的 `events` 是内部的 `AgentEvent`，包含工具调用等内部细节；
//! - A2A 的 `Task` 对外暴露 `history`（消息序列）和 `artifacts`（产物）。
//!
//! 所以这里做一层**桥接**：A2A 的 `contextId` 直接复用 `Session` 的
//! `session_id`，任务状态与历史保存在 [`TaskStore`] 里，而 agent 的实际运行
//! 上下文（工具调用、审批状态）保存在 Session 里。两者用同一个 id 关联。
//!
//! # 生命周期
//!
//! ```text
//!   SendMessage
//!       │
//!       ├──▶ 查 store：contextId 对应的任务存在？
//!       │        │
//!       │        ├── 是 ──▶ 追加消息到 history，继续同一个任务
//!       │        └── 否 ──▶ 创建新 Task（TASK_STATE_SUBMITTED）
//!       │
//!       ├──▶ 交给 Agent 执行（后台或阻塞）
//!       │
//!       └──▶ 结果写回：状态 + artifacts + history
//! ```
//!
//! # 模块结构
//!
//! ```text
//!   task_store.rs        ── 本文件：trait 定义 + 存储实现
//!   task_store/lifecycle.rs ── 任务状态机与历史裁剪等纯函数
//! ```
//!
//! # 存储实现
//!
//! 提供两种实现：
//!
//! - [`InMemoryTaskStore`]：进程内 `HashMap`，重启即丢，适合开发与测试；
//! - [`SessionTaskStore`]：把任务状态存进 `Session.state`，复用项目已有的
//!   `SessionManager` 持久化能力（项目里已有 SQLite 实现）。

pub mod lifecycle;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    a2a::types::Task,
    session::{model::Session, session_manager::SessionManager},
};

pub use lifecycle::{
    TaskLifecycle, artifacts_to_text, ensure_cancelable, merge_or_create_task, truncate_history,
};

/// 任务存储的抽象。
///
/// 规范 3.1 节定义的核心操作是 `GetTask` / `ListTasks` / `CancelTask`，
/// 这里把它们抽象成 trait，方便替换后端。
#[async_trait::async_trait]
pub trait TaskStore: Send + Sync {
    /// 按任务 id 取任务。
    async fn get(&self, task_id: &str) -> Option<Task>;

    /// 按上下文 id 列出任务。
    async fn list_by_context(&self, context_id: &str) -> Vec<Task>;

    /// 保存（新建或覆盖）任务。
    async fn save(&self, task: Task) -> anyhow::Result<()>;

    /// 列出全部任务。
    async fn list_all(&self) -> Vec<Task>;

    /// 删除任务。
    async fn remove(&self, task_id: &str) -> anyhow::Result<()>;
}

/// 进程内任务存储。
///
/// 适合开发、演示与单元测试。生产环境建议换成持久化实现。
#[derive(Debug, Default)]
pub struct InMemoryTaskStore {
    tasks: Mutex<HashMap<String, Task>>,
}

impl InMemoryTaskStore {
    /// 创建一个空的存储。
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl TaskStore for InMemoryTaskStore {
    async fn get(&self, task_id: &str) -> Option<Task> {
        self.tasks.lock().unwrap().get(task_id).cloned()
    }

    async fn list_by_context(&self, context_id: &str) -> Vec<Task> {
        self.tasks
            .lock()
            .unwrap()
            .values()
            .filter(|task| task.context_id.as_deref() == Some(context_id))
            .cloned()
            .collect()
    }

    async fn save(&self, task: Task) -> anyhow::Result<()> {
        self.tasks.lock().unwrap().insert(task.id.clone(), task);
        Ok(())
    }

    async fn list_all(&self) -> Vec<Task> {
        self.tasks.lock().unwrap().values().cloned().collect()
    }

    async fn remove(&self, task_id: &str) -> anyhow::Result<()> {
        self.tasks.lock().unwrap().remove(task_id);
        Ok(())
    }
}

/// 把任务状态存进 [`Session`] 的 `state` 字段的存储。
///
/// 复用项目已有的 `SessionManager`。这样一来，A2A 任务的生命周期就和
/// 项目已有的会话持久化（例如 SQLite）绑在一起，不需要引入新的存储依赖。
///
/// # 存储布局
///
/// `session.state` 里的键：
///
/// - `a2a.tasks`：该 context 下的全部任务（JSON 数组）
/// - `a2a.task_index`：`taskId → contextId` 的映射（JSON 对象）
///
/// 之所以需要索引，是因为 `SessionManager` 的 `get` 需要先知道 `session_id`，
/// 而 A2A 的 `GetTask` 只给 `taskId`。索引让"按任务 id 找上下文"变成 O(1)。
pub struct SessionTaskStore {
    session_manager: Arc<dyn SessionManager>,
    /// `taskId → contextId` 的内存索引，进程重启后可从 session 中重建。
    index: Mutex<HashMap<String, String>>,
}

impl SessionTaskStore {
    /// 用一个会话管理器创建存储。
    pub fn new(session_manager: Arc<dyn SessionManager>) -> Self {
        Self {
            session_manager,
            index: Mutex::new(HashMap::new()),
        }
    }

    /// `Session.state` 中存放任务列表的键。
    const TASKS_KEY: &'static str = "a2a.tasks";
    /// `Session.state` 中存放任务索引的键。
    const INDEX_KEY: &'static str = "a2a.task_index";

    /// 从 session 中读出任务列表。
    fn load_tasks(session: &Session) -> Vec<Task> {
        session
            .state
            .get(Self::TASKS_KEY)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default()
    }

    /// 从 session 中读出任务索引。
    fn load_index(session: &Session) -> HashMap<String, String> {
        session
            .state
            .get(Self::INDEX_KEY)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default()
    }

    /// 把任务列表写回 session。
    fn store_tasks(session: &mut Session, tasks: &[Task]) {
        if let Ok(value) = serde_json::to_value(tasks) {
            session.state.insert(Self::TASKS_KEY.to_owned(), value);
        }
    }
}

#[async_trait::async_trait]
impl TaskStore for SessionTaskStore {
    async fn get(&self, task_id: &str) -> Option<Task> {
        // 先查内存索引定位 context。
        let context_id = self.index.lock().unwrap().get(task_id).cloned()?;

        // 索引可能因为进程重启而丢失，回退到扫描已知 session。
        let tasks = match self.session_manager.get(&context_id).await {
            Ok(Some(session)) => Self::load_tasks(&session),
            _ => Vec::new(),
        };

        tasks.into_iter().find(|task| task.id == task_id)
    }

    async fn list_by_context(&self, context_id: &str) -> Vec<Task> {
        match self.session_manager.get(context_id).await {
            Ok(Some(session)) => Self::load_tasks(&session),
            _ => Vec::new(),
        }
    }

    async fn save(&self, task: Task) -> anyhow::Result<()> {
        let context_id = task
            .context_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("task '{}' has no context id", task.id))?;

        let mut session = self
            .session_manager
            .get_or_create(&context_id, None)
            .await?;
        let mut tasks = Self::load_tasks(&session);

        match tasks.iter_mut().find(|item| item.id == task.id) {
            Some(existing) => *existing = task.clone(),
            None => tasks.push(task.clone()),
        }

        Self::store_tasks(&mut session, &tasks);

        // 索引随 session 一起持久化，保证重启后仍能按 id 定位。
        let mut index = Self::load_index(&session);
        index.insert(task.id.clone(), context_id.clone());
        if let Ok(value) = serde_json::to_value(&index) {
            session.state.insert(Self::INDEX_KEY.to_owned(), value);
        }

        self.index
            .lock()
            .unwrap()
            .insert(task.id.clone(), context_id);

        self.session_manager.save(session).await
    }

    async fn list_all(&self) -> Vec<Task> {
        // 只能列出内存索引里已知 context 的任务。
        let contexts: Vec<String> = self.index.lock().unwrap().values().cloned().collect();
        let mut all = Vec::new();
        for context in contexts {
            all.extend(self.list_by_context(&context).await);
        }
        all
    }

    async fn remove(&self, task_id: &str) -> anyhow::Result<()> {
        let context_id = self.index.lock().unwrap().get(task_id).cloned();
        let Some(context_id) = context_id else {
            return Ok(());
        };

        if let Ok(Some(mut session)) = self.session_manager.get(&context_id).await {
            let mut tasks = Self::load_tasks(&session);
            let before = tasks.len();
            tasks.retain(|task| task.id != task_id);
            if tasks.len() != before {
                Self::store_tasks(&mut session, &tasks);
                self.session_manager.save(session).await?;
            }
        }

        self.index.lock().unwrap().remove(task_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a2a::types::{Artifact, Message, Role};

    fn message(text: &str) -> Message {
        Message::new_text(Role::User, text, None, None)
    }

    #[tokio::test]
    async fn in_memory_store_roundtrips_tasks() {
        let store = InMemoryTaskStore::new();
        let mut task = Task::new_submitted(Some("ctx".into()));
        task.history.push(message("hello"));

        store.save(task.clone()).await.unwrap();

        let fetched = store.get(&task.id).await.unwrap();
        assert_eq!(fetched.id, task.id);
        assert_eq!(fetched.history.len(), 1);

        let by_context = store.list_by_context("ctx").await;
        assert_eq!(by_context.len(), 1);

        store.remove(&task.id).await.unwrap();
        assert!(store.get(&task.id).await.is_none());
    }

    #[tokio::test]
    async fn session_store_persists_tasks_and_can_find_them_by_id() {
        let manager: Arc<dyn SessionManager> =
            Arc::new(crate::session::session_manager::InMemorySessionManager::new());
        let store = SessionTaskStore::new(manager);

        let mut task = Task::new_submitted(Some("ctx-1".into()));
        task.history.push(message("hi"));
        store.save(task.clone()).await.unwrap();

        let listed = store.list_by_context("ctx-1").await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, task.id);

        // 按 id 也能取回（依赖索引）。
        let by_id = store.get(&task.id).await.unwrap();
        assert_eq!(by_id.id, task.id);
    }

    #[test]
    fn artifacts_are_flattened_to_text() {
        let artifacts = vec![
            Artifact::new_text("a", "first"),
            Artifact::new_text("b", "second"),
        ];
        assert_eq!(artifacts_to_text(&artifacts), "first\nsecond");
    }
}
