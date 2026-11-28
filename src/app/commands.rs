//! 斜杠命令处理。
//!
//! 命令分三类：
//!
//! - **查询类**（`/status` `/agents` `/tools` `/history` `/knowledge` `/a2a`）——只读；
//! - **变更类**（`/new` `/remember` `/clear`）——改会话状态；
//! - **控制类**（`/quit`）——返回 `false` 让主循环退出。

use crate::{
    app::{
        render::{
            agents_markdown, print_banner, print_history, render_markdown, status_text,
            tools_markdown,
        },
        App,
    },
    constant::A2A_CARD_PATH,
};

/// 默认展示的历史条数。
const HISTORY_LIMIT: usize = 20;

/// 处理一条斜杠命令。
///
/// 返回 `false` 表示"退出 REPL"。
pub async fn handle_command(app: &mut App, input: &str) -> anyhow::Result<bool> {
    let mut parts = input.split_whitespace();
    let command = parts.next().unwrap_or_default();

    match command {
        "/quit" | "/exit" | "/q" => return Ok(false),
        "/help" => render_markdown(crate::app::help::HELP),
        "/new" => {
            let session_id = format!("chat-{}", uuid::Uuid::new_v4());
            let session = app
                .root
                .session_manager
                .get_or_create(&session_id, None)
                .await?;
            app.context = crate::agent::context::ExecutionContext::new(session);
            app.session_id = session_id;
            println!("new session: {}", app.session_id);
        }
        "/status" => println!("{}", status_text(app)),
        "/agents" => render_markdown(&agents_markdown(app)),
        "/tools" => render_markdown(&tools_markdown(app)),
        "/history" => {
            let limit = parts
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(HISTORY_LIMIT);
            print_history(app, limit);
        }
        "/remember" | "/save" => remember(app).await?,
        "/knowledge" => print_knowledge(app)?,
        "/a2a" => print_a2a(app)?,
        "/clear" => {
            print!("\x1b[2J\x1b[H");
            crate::app::render::flush()?;
            print_banner(app);
        }
        other => println!("unknown command: {other}. Type /help for help."),
    }

    Ok(true)
}

/// 把当前会话提取为长期记忆。
async fn remember(app: &mut App) -> anyhow::Result<()> {
    let Some(memory) = &app.memory else {
        println!("long-term memory is disabled");
        return Ok(());
    };

    println!("extracting long-term memory...");
    match memory.save(&app.context.session.events).await {
        Ok(Some(id)) => println!("saved memory: {id}"),
        Ok(None) => println!("nothing saved (duplicate, empty, or extraction failed)"),
        Err(err) => println!("memory save failed: {err:#}"),
    }
    Ok(())
}

/// 列出知识库里的文档。
///
/// 知识库可以被 Agent 通过 `add_knowledge` 写入，运维视角需要能确认
/// "库里现在到底有哪些资料"。
fn print_knowledge(app: &App) -> anyhow::Result<()> {
    let Some(knowledge) = &app.knowledge else {
        println!("knowledge base is disabled");
        return Ok(());
    };

    let documents = knowledge.list_documents()?;
    let total_chunks = knowledge.total_chunks()?;
    if documents.is_empty() {
        println!("knowledge base is empty ({total_chunks} chunks)");
        return Ok(());
    }

    println!("knowledge base: {} documents, {total_chunks} chunks", documents.len());
    for doc in documents {
        let source = if doc.source.is_empty() {
            String::new()
        } else {
            format!(" — {}", doc.source)
        };
        println!(
            "  [{}] {}{} ({} chars)",
            &doc.id[..doc.id.len().min(8)],
            doc.title,
            source,
            doc.content.chars().count()
        );
    }

    Ok(())
}

/// 打印 A2A 集成状态。
///
/// 运维视角需要知道两件事：
/// - 本机对外暴露了什么（服务端地址 + Agent Card 地址）；
/// - 本机能向谁委派活儿（已接入的远端 peer）。
fn print_a2a(app: &App) -> anyhow::Result<()> {
    println!("== A2A 服务端 ==");
    match &app.a2a_addr {
        Some(addr) => {
            println!("  状态      : 运行中");
            println!("  监听      : {addr}");
            println!("  Agent Card: {addr}{A2A_CARD_PATH}");
            println!("  JSON-RPC  : {addr}/rpc");
            println!(
                "  鉴权      : {}",
                if app.features.a2a_auth {
                    "已启用（Bearer Token）"
                } else {
                    "未启用"
                }
            );
        }
        None => {
            if app.one_shot {
                println!("  状态      : 未启动（一次性模式：进程退出即结束，无需对外暴露）");
            } else {
                println!("  状态      : 未启动");
            }
        }
    }

    println!("\n== 已接入的远端 A2A agent ==");
    if app.a2a_peer_names.is_empty() {
        println!("  (无)");
    } else {
        for name in &app.a2a_peer_names {
            println!("  · {name}");
        }
    }

    println!("\n配置: ZEPHYR_A2A_HOST / ZEPHYR_A2A_PORT / ZEPHYR_A2A_PEERS");

    Ok(())
}
