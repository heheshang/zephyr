//! 在 A2A 任务上运行 agent —— 协议与 runtime 之间的桥。
//!
//! # 桥接关系
//!
//! ```text
//!   A2A 侧                          项目 runtime 侧
//!   ─────                          ────────────────
//!   Task.contextId        ──────►   Session.session_id
//!   Message.parts[text]   ──────►   AgentEvent(user message)
//!   AgentStatus::Complete ──────► TaskStatus::COMPLETED + Artifact
//!   AgentStatus::PendingConfirmation ──► TaskStatus::INPUT_REQUIRED
//! ```
//!
//! # 为什么把执行轨迹写进 artifact
//!
//! A2A 协议本身**不暴露** agent 的内部过程——这正是它与 MCP 的差别：
//! 远端客户端只能看到最终产物，看不到转移、工具调用这些中间步骤。
//!
//! 但这会让 A2A 端到端测试变成"黑盒猜谜"：无法判断 agent 到底是
//! 自己答的，还是转移给了子 agent；也无法确认工具是否真的执行了。
//!
//! 因此这里把轨迹写进 artifact 的 `metadata`（规范 4.1.7 节允许
//! artifact 带自定义元数据），让调用方**可选地**观测到执行细节。
//! 这不改变协议的语义，只是把本已存在的信息显式化。

use std::sync::Arc;

use crate::{
    a2a::{
        server::A2AServerState,
        task_store::TaskLifecycle,
        types::{Artifact, Message, Task, TaskStatus},
    },
    agent::{
        context::ExecutionContext,
        event::ContentItem,
        model::{AgentStatus},
        runtime::push_user_message,
    },
};

/// agent 一次执行的结果。
pub struct AgentOutcome {
    /// 产出的产物。
    pub artifacts: Vec<Artifact>,
    /// 映射后的任务状态。
    pub status: TaskStatus,
}

/// 确保任务带有 `contextId`，并把它回填到最后一条历史消息上。
///
/// 规范 4.1.1 节：`contextId` 是"一组相关交互的集合标识"。
/// 客户端首条消息通常不带它，此时由服务端分配；否则后续多轮消息
/// 无法关联到同一个上下文，多轮对话就会断成若干独立回合。
///
/// 消息本身也要带上 `contextId`——客户端据此在下一轮消息里带回它。
pub fn ensure_context_id(task: &mut Task) {
    if task.context_id.is_none() {
        task.context_id = Some(uuid::Uuid::new_v4().to_string());
    }

    if let Some(context_id) = task.context_id.clone()
        && let Some(message) = task.history.last_mut()
    {
        message.context_id = Some(context_id);
    }
}

/// 从 agent 的执行上下文里提取可观测的执行轨迹。
///
/// 见模块文档：轨迹写进 artifact 的 `metadata`，让远端调用方可选地观测。
pub fn extract_execution_trace(context: &ExecutionContext) -> Vec<serde_json::Value> {
    let mut trace = Vec::new();

    for event in &context.session.events {
        for item in &event.content {
            match item {
                ContentItem::ToolCall(call) => {
                    trace.push(serde_json::json!({
                        "kind": "tool_call",
                        "agent": event.author,
                        "name": call.name,
                    }));
                }
                ContentItem::ToolResult(result) => {
                    trace.push(serde_json::json!({
                        "kind": "tool_result",
                        "agent": event.author,
                        "name": result.name,
                        "status": format!("{:?}", result.status),
                    }));
                }
                ContentItem::Message(_) => {}
            }
        }
    }

    trace
}

/// 在给定的 A2A 任务上运行 agent。
///
/// 上下文衔接：agent 的 `session_id` 用 A2A 的 `contextId`，
/// 因此同一个 context 下的多轮消息会共享同一份对话历史。
pub async fn run_agent(
    state: &Arc<A2AServerState>,
    task: &Task,
    message: &Message,
) -> anyhow::Result<AgentOutcome> {
    let context_id = task.context_id.clone().unwrap_or_else(|| task.id.clone());

    let session = state
        .session_manager
        .get_or_create(&context_id, None)
        .await?;
    let mut context = ExecutionContext::new(session);

    // 把 A2A 消息的文本部分作为用户消息推进 agent。
    let text = message.text();
    push_user_message(&mut context, &text);

    let result = state.agent.run_in_context(context).await?;

    // 提取执行轨迹：转移路径、工具调用。这些信息本已存在于会话事件里，
    // 这里显式化，好让 A2A 客户端（尤其是测试）能观测到 agent 的真实行为。
    let trace = extract_execution_trace(&result.context);

    match result.status {
        AgentStatus::Complete => {
            let output = match result.output {
                Some(serde_json::Value::String(text)) => text,
                Some(other) => other.to_string(),
                None => String::new(),
            };

            let mut artifact = Artifact::new_text("result", output);

            // 把执行轨迹写进 artifact 的 metadata（规范 4.1.7 节允许
            // artifact 带自定义元数据）。
            let mut metadata = serde_json::Map::new();
            metadata.insert(
                "execution_trace".to_owned(),
                serde_json::Value::Array(trace.clone()),
            );
            metadata.insert(
                "transfer_history".to_owned(),
                serde_json::to_value(&result.context.transfer_history).unwrap_or_default(),
            );
            metadata.insert(
                "final_agent".to_owned(),
                serde_json::Value::String(
                    result
                        .context
                        .session
                        .events
                        .last()
                        .map_or_else(|| state.agent.name.clone(), |event| event.author.clone()),
                ),
            );
            artifact.metadata = Some(metadata);

            Ok(AgentOutcome {
                artifacts: vec![artifact],
                status: TaskLifecycle::completed(),
            })
        }
        AgentStatus::PendingConfirmation => {
            // agent 需要人工确认（例如危险工具）。对 A2A 客户端来说，
            // 这等价于"我需要你补充信息"，映射成 INPUT_REQUIRED。
            Ok(AgentOutcome {
                artifacts: Vec::new(),
                status: TaskLifecycle::input_required(
                    "The agent requires additional input or confirmation to proceed.",
                ),
            })
        }
        AgentStatus::Error => Err(anyhow::anyhow!("agent execution failed")),
    }
}
