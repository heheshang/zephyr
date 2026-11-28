//! 执行一个用户回合。
//!
//! # 一个回合是什么
//!
//! 一次用户输入 → agent 循环（想 → 做 → 看结果）→ 最终答案。
//! 期间可能夹杂**工具审批**：agent 请求执行危险工具时暂停，
//! 等用户确认后继续。
//!
//! # 状态机
//!
//! ```text
//!   用户输入
//!     │
//!     ▼
//!   run_in_context()
//!     │
//!     ├── Complete ──────────────▶ 打印答案，保存会话，结束
//!     │
//!     ├── PendingConfirmation ───▶ 询问用户 → 写确认结果 → 继续跑
//!     │        ▲                          │
//!     │        └──────────────────────────┘
//!     │
//!     └── Error ─────────────────▶ 中断
//! ```

use std::{collections::HashMap, io, io::Write, sync::Arc};

use crate::{
    agent::{
        context::ExecutionContext,
        model::{Agent, AgentStatus},
        runtime::push_user_message,
        tool_confirmation::ToolConfirmation,
    },
    app::{render::styled, App},
};

/// 审批轮次上限。
///
/// 防止"确认一次又请求一次"的无限循环把会话卡死。
const MAX_APPROVAL_ROUNDS: usize = 5;

/// 执行一个回合。
pub async fn run_turn(app: &mut App, input: String) -> anyhow::Result<()> {
    push_user_message(&mut app.context, &input);
    // 每个用户回合都是新的一次执行：步数、转移次数和转移路径重新计数。
    app.context.current_step = 0;
    app.context.transfer_count = 0;
    app.context.transfer_history.clear();

    let mut result = app.root.run_in_context(app.context.clone()).await?;
    let mut approval_rounds = 0;

    loop {
        match result.status {
            AgentStatus::Complete => {
                let speaker = result
                    .context
                    .transfer_history
                    .last()
                    .and_then(|hop| hop.rsplit_once(" -> ").map(|(_, to)| to))
                    .unwrap_or(&app.root.name)
                    .to_owned();

                if let Some(output) = result.output {
                    let text = match output {
                        serde_json::Value::String(text) => text,
                        other => format!(
                            "```json\n{}\n```",
                            serde_json::to_string_pretty(&other).unwrap_or_default()
                        ),
                    };
                    println!();
                    println!("{}", styled(&format!("{speaker}>"), "1;32"));
                    crate::app::render::render_markdown(&text);
                }

                app.context = result.context;
                app.root
                    .session_manager
                    .save(app.context.session.clone())
                    .await?;
                return Ok(());
            }
            AgentStatus::PendingConfirmation => {
                approval_rounds += 1;
                if approval_rounds > MAX_APPROVAL_ROUNDS {
                    anyhow::bail!("too many tool approval rounds");
                }

                let active = active_agent(&result.context, &app.root, &app.agents);
                let confirmations = prompt_confirmations(&result.context)?;
                let mut context = result.context;
                context.session.state.insert(
                    "tool_confirmations".to_owned(),
                    serde_json::to_value(&confirmations)?,
                );

                result = active.run_in_context(context).await?;
            }
            AgentStatus::Error => anyhow::bail!("agent stopped with an error"),
        }
    }
}

/// 找出当前实际在跑的 agent。
///
/// 转移发生后，最后一条转移路径的终点就是当前执行者。
fn active_agent(
    context: &ExecutionContext,
    root: &Arc<Agent>,
    agents: &HashMap<String, Arc<Agent>>,
) -> Arc<Agent> {
    context
        .transfer_history
        .last()
        .and_then(|hop| hop.rsplit_once(" -> ").map(|(_, to)| to))
        .and_then(|name| agents.get(name))
        .cloned()
        .unwrap_or_else(|| root.clone())
}

/// 逐个询问用户是否批准待执行的工具调用。
///
/// 支持 `[y]` 批准、`[n]` 拒绝、`[a]` 全部批准。
fn prompt_confirmations(context: &ExecutionContext) -> anyhow::Result<Vec<ToolConfirmation>> {
    let pending = &context.session.pending_tool_calls;
    if pending.is_empty() {
        return Ok(Vec::new());
    }

    println!("\n{}", styled("tool approval required", "1;33"));
    let mut approve_all = false;
    let mut confirmations = Vec::with_capacity(pending.len());

    for call in pending {
        let approved = if approve_all {
            true
        } else {
            println!("{}", call.confirmation_message);
            print!("[y] approve  [n] deny  [a] approve all: ");
            io::stdout().flush()?;

            let mut reply = String::new();
            if io::stdin().read_line(&mut reply)? == 0 {
                false
            } else {
                match reply.trim().to_lowercase().as_str() {
                    "y" | "yes" => true,
                    "a" | "all" => {
                        approve_all = true;
                        true
                    }
                    _ => false,
                }
            }
        };

        confirmations.push(ToolConfirmation {
            tool_call_id: call.tool_call.id.clone(),
            reason: Some(if approved {
                "approved in terminal".to_owned()
            } else {
                "denied in terminal".to_owned()
            }),
            approved,
            modified_arguments: None,
        });
    }

    Ok(confirmations)
}
