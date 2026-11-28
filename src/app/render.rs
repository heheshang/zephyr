//! 终端渲染：颜色、Markdown、横幅。
//!
//! 所有"往终端上画东西"的逻辑集中在这里，方便统一处理
//! "输出是不是终端"这个判断——管道重定向时不该吐出 ANSI 转义序列。

use std::io::{self, IsTerminal, Write};

use termimad::MadSkin;

use crate::{
    app::App,
    constant::A2A_CARD_PATH,
};

/// 给文本包裹 ANSI 样式。
///
/// 非终端环境（管道、文件重定向）下返回原文，避免转义序列污染输出。
pub fn styled(text: &str, code: &str) -> String {
    if io::stdout().is_terminal() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

/// 按终端 Markdown 渲染一段文本。
pub fn render_markdown(text: &str) {
    if io::stdout().is_terminal() {
        let skin = MadSkin::default();
        skin.print_text(text.trim_end());
        println!();
    } else {
        println!("{}", text.trim_end());
    }
}

/// 打印启动横幅。
pub fn print_banner(app: &App) {
    let memory = flag(app.features.memory);
    let knowledge = flag(app.features.knowledge);
    let code = flag(app.features.code);
    let mcp = flag(app.features.mcp);
    let skills = flag(app.features.skills);
    let a2a = flag(app.features.a2a_server);

    println!();
    println!("{}  terminal assistant", styled("zephyr", "1;36"));
    println!("model   : {}", app.model);
    println!("session : {}", app.session_id);
    println!(
        "features: mcp={mcp}  code={code}  memory={memory}  knowledge={knowledge}  skills={skills}  a2a={a2a}  transfer=on"
    );
    let mut agent_names = app.agents.keys().cloned().collect::<Vec<_>>();
    agent_names.sort();
    println!("agents  : {}", agent_names.join(", "));
    if let Some(addr) = &app.a2a_addr {
        let auth = if app.features.a2a_auth { "  [鉴权开启]" } else { "" };
        println!("a2a     : {addr}{auth}");
        println!("         Agent Card: {addr}{A2A_CARD_PATH}");
    }
    if app.features.a2a_peers > 0 {
        println!("peers   : {} 个远端 A2A agent 已接入", app.features.a2a_peers);
    }
    println!("commands: /help  /new  /status  /agents  /tools  /history  /remember  /knowledge  /a2a  /quit");
    println!();
}

/// 把布尔能力开关渲染成 `on` / `off`。
fn flag(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

/// 打印 `/status` 的纯文本状态。
pub fn status_text(app: &App) -> String {
    let transfer = if app.context.transfer_history.is_empty() {
        "(none)".to_owned()
    } else {
        app.context.transfer_history.join(", ")
    };
    format!(
        "model: {}\nsession: {}\nevents: {}\nlast step: {}\ntransfers: {}\nfeatures: mcp={} code={} memory={} knowledge={} skills={} a2a_server={} a2a_peers={}\na2a: {}",
        app.model,
        app.session_id,
        app.context.session.events.len(),
        app.context.current_step,
        transfer,
        app.features.mcp,
        app.features.code,
        app.features.memory,
        app.features.knowledge,
        app.features.skills,
        app.features.a2a_server,
        app.features.a2a_peers,
        app.a2a_addr.as_deref().unwrap_or("(disabled)"),
    )
}

/// 渲染 agent 树为 Markdown。
pub fn agents_markdown(app: &App) -> String {
    fn walk(agent: &crate::agent::model::Agent, depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        out.push_str(&format!(
            "{indent}- **{}** — {}\n",
            agent.name, agent.description
        ));

        let targets = agent
            .transfer_targets()
            .iter()
            .map(|target| target.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        if !targets.is_empty() {
            out.push_str(&format!("{indent}  transfer: `{targets}`\n"));
        }

        for child in agent.sub_agents() {
            walk(child, depth + 1, out);
        }
    }

    let mut out = String::from("# Agents\n\n");
    walk(&app.root, 0, &mut out);
    out
}

/// 渲染各 agent 的工具清单为 Markdown。
pub fn tools_markdown(app: &App) -> String {
    let mut names = app.agents.keys().cloned().collect::<Vec<_>>();
    names.sort();

    let mut out = String::from("# Tools\n\n");
    for name in names {
        let Some(agent) = app.agents.get(&name) else {
            continue;
        };
        out.push_str(&format!("## {name}\n\n"));

        let mut tools = agent.toolbox.keys().cloned().collect::<Vec<_>>();
        tools.sort();
        for tool_name in tools {
            let description = agent
                .toolbox
                .get(&tool_name)
                .map(|tool| tool.description())
                .unwrap_or_default();
            out.push_str(&format!(
                "- `{tool_name}` — {}\n",
                description.replace('\n', " ")
            ));
        }
        out.push('\n');
    }
    out
}

/// 打印最近的历史事件。
pub fn print_history(app: &App, limit: usize) {
    let events = &app.context.session.events;
    let start = events.len().saturating_sub(limit);
    if start == events.len() {
        println!("(no history)");
        return;
    }

    for event in &events[start..] {
        println!("\n[{}]", event.author);
        for item in &event.content {
            match item {
                crate::agent::event::ContentItem::Message(message) => {
                    println!("{}: {}", message.role, preview(&message.content, 500));
                }
                crate::agent::event::ContentItem::ToolCall(call) => {
                    println!("tool_call {} {}", call.name, call.arguments);
                }
                crate::agent::event::ContentItem::ToolResult(result) => {
                    let status = match result.status {
                        crate::agent::event::ToolResultStatus::Success => "ok",
                        crate::agent::event::ToolResultStatus::Error => "error",
                    };
                    println!(
                        "tool_result {} ({status}): {}",
                        result.name,
                        preview(&result.content, 200)
                    );
                }
            }
        }
    }
}

/// 截断过长文本用于预览。
pub fn preview(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    let mut out = text.chars().take(max_chars).collect::<String>();
    if count > max_chars {
        out.push('…');
    }
    out.replace('\n', " ")
}

/// 刷新标准输出缓冲。
pub fn flush() -> anyhow::Result<()> {
    io::stdout().flush()?;
    Ok(())
}
