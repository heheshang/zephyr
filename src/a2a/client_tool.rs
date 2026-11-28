//! 把远端 A2A agent 包装成本地 [`Tool`]。
//!
//! # 定位
//!
//! 项目里已经有两条"跨 agent"的路径：
//!
//! - [`crate::tool::agent_tool::AgentTool`]：把**进程内**的 `Agent` 变成工具；
//! - 本模块：把**远端**的 A2A agent 变成工具。
//!
//! 两者的差别正好对应 A2A 与 MCP 的差别：
//!
//! ```text
//!   AgentTool（进程内）      A2ATool（跨进程 / 跨组织）
//!   ─────────────────      ──────────────────────────
//!   共享 Rust 进程          走 HTTP + JSON-RPC
//!   直接调 run_in_context    先发现 Agent Card，再发 SendMessage
//!   能看到中间事件          只看到最终产物（远端把过程封装了）
//! ```
//!
//! 这也正是 A2A 存在的意义：**A2A 管的是 agent 之间怎么对话，
//! MCP 管的是 agent 怎么调工具**。
//!
//! # 使用方式
//!
//! ```ignore
//! let client = A2AClient::discover("http://remote-agent:8080").await?;
//! let tool = A2ATool::new(client);
//! toolbox.insert("remote_researcher".to_owned(), Box::new(tool));
//! ```

use std::sync::Arc;

use serde_json::{Value, json};

use crate::{
    a2a::{client::A2AClient, types::TaskState},
    agent::context::ExecutionContext,
    tool::Tool,
};

/// 把远端 A2A agent 包装成可调用的工具。
pub struct A2ATool {
    client: Arc<A2AClient>,
    name: String,
    description: String,
    parameters: Value,
    /// 透传的租户标识（规范 3.2.1 节）。
    tenant: Option<String>,
}

impl A2ATool {
    /// 用一个已完成发现的客户端创建工具。
    ///
    /// 工具名与描述优先取远端 Agent Card 里的 `name` / `description`，
    /// 这样协调员看到的就是远端 agent 自己声明的能力。
    pub fn new(client: A2AClient) -> Self {
        let client = Arc::new(client);

        let (name, description) = match client.card() {
            Some(card) => {
                let description = if card.description.trim().is_empty() {
                    format!("Delegate a task to the remote '{}' A2A agent.", card.name)
                } else {
                    card.description.clone()
                };
                (card.name.clone(), description)
            }
            None => (
                "remote_agent".to_owned(),
                "Delegate a task to a remote A2A agent.".to_owned(),
            ),
        };

        Self {
            client,
            name,
            description,
            parameters: json!({
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task or question to delegate to the remote agent"
                    }
                },
                "required": ["request"]
            }),
            tenant: None,
        }
    }

    /// 覆盖工具名。
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 覆盖工具描述。
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// 设置租户标识。
    ///
    /// 规范 4.4.6 节：多个 agent 共用一个 A2A 端点时，用 `tenant` 区分路由。
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// 底层的客户端。
    pub fn client(&self) -> &A2AClient {
        &self.client
    }
}

#[async_trait::async_trait]
impl Tool for A2ATool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let arguments: Value = serde_json::from_str(args_json)?;
        let request = arguments
            .get("request")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "tool '{}' expects a string parameter named 'request'",
                    self.name
                )
            })?
            .to_owned();

        tracing::info!("A2ATool '{}' delegating: {request}", self.name);

        let task = self.client.send_text(request).await?;

        match task.state() {
            TaskState::Completed => {
                // 把所有产物拼成最终答案交给协调员。
                let text = task
                    .artifacts
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
                    .join("\n");

                if text.trim().is_empty() {
                    // 没有产物时退回最后一条消息。
                    Ok(task
                        .history
                        .last()
                        .map(|message| message.text())
                        .unwrap_or_default())
                } else {
                    Ok(text)
                }
            }
            TaskState::InputRequired => Ok(format!(
                "The remote agent needs more input: {}",
                task.status
                    .message
                    .as_ref()
                    .map(|message| message.text())
                    .unwrap_or_default()
            )),
            TaskState::Failed => Err(anyhow::anyhow!(
                "remote agent '{}' failed: {}",
                self.name,
                task.status
                    .message
                    .as_ref()
                    .map(|message| message.text())
                    .unwrap_or_default()
            )),
            other => Ok(format!(
                "Remote agent '{}' returned task state {} (task id: {})",
                self.name,
                other.as_str(),
                task.id
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a2a::card::AgentCard;

    #[test]
    fn tool_uses_the_remote_card_name_and_description() {
        let client = A2AClient::new("http://x/rpc");
        // 没有卡片时用兜底名。
        let tool = A2ATool::new(client);
        assert_eq!(tool.name(), "remote_agent");
        assert!(tool.description().contains("remote A2A agent"));
    }

    #[test]
    fn tool_parameters_request_a_single_string() {
        let tool = A2ATool::new(A2AClient::new("http://x/rpc"));
        let parameters = tool.parameters();
        assert_eq!(parameters["required"], json!(["request"]));
        assert_eq!(parameters["properties"]["request"]["type"], "string");
    }

    #[test]
    fn name_and_description_can_be_overridden() {
        let tool = A2ATool::new(A2AClient::new("http://x/rpc"))
            .with_name("researcher")
            .with_description("Researches remotely");
        assert_eq!(tool.name(), "researcher");
        assert_eq!(tool.description(), "Researches remotely");
    }

    #[tokio::test]
    async fn execute_rejects_missing_request_parameter() {
        let tool = A2ATool::new(A2AClient::new("http://x/rpc"));
        let error = tool
            .execute(r#"{"wrong": 1}"#, &ExecutionContext::default())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("request"));
    }

    #[test]
    fn card_description_drives_tool_description() {
        let card = AgentCard::new("remote_agent", "Remote specialist", "http://x", "1.0");
        assert_eq!(card.name, "remote_agent");
    }
}
