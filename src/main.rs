//! 终端交互式 AI Agent —— 进程入口。
//!
//! 这是一个行式 REPL，不进入 alternate screen：
//! - 正常使用终端滚动和复制粘贴；
//! - 助手回答用 Markdown 渲染；
//! - 支持 transfer、工具审批、长期记忆、沙箱、MCP 和上下文优化。
//!
//! # 这里只做什么
//!
//! `main.rs` 只负责**进程生命周期**：
//!
//! ```text
//!   解析参数 ──▶ 组装应用 ──▶ REPL 循环 ──▶ 退出
//! ```
//!
//! 具体逻辑在 [`crate::app`] 下按职责拆分：
//! 组装（`builder`）、A2A 集成（`a2a`）、回合执行（`turn`）、
//! 命令处理（`commands`）、渲染（`render`）、提示词（`prompts`）。

use std::io::{self, IsTerminal, Write};

use zephyr::app::{
    build_app, commands::handle_command, help::HELP, render::render_markdown, turn::run_turn,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("zephyr=info,rmcp=warn"));
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_writer(io::stderr)
        .with_env_filter(filter)
        .init();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if matches!(args.first().map(String::as_str), Some("-h" | "--help")) {
        render_markdown(HELP);
        return Ok(());
    }

    // 一次性模式：带着问题直接跑一次就退出。
    //
    // 这种模式下**不启动 A2A 服务端**——进程马上就结束了，
    // 外部系统根本来不及发现或调用它，启动只是白占一个端口。
    // 但**仍然要接入远端 peer**：那是为了让本次回答能跨进程委派，
    // 与服务端无关。
    let one_shot = args.join(" ");
    let one_shot = if one_shot.trim().is_empty() {
        None
    } else {
        Some(one_shot)
    };

    let mut app = build_app(one_shot.is_some()).await?;
    zephyr::app::render::print_banner(&app);

    if let Some(question) = one_shot {
        run_turn(&mut app, question).await?;
        return Ok(());
    }

    loop {
        let Some(input) = read_input()? else {
            break;
        };
        let input = input.trim().to_owned();
        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            if !handle_command(&mut app, &input).await? {
                break;
            }
            continue;
        }

        if let Err(err) = run_turn(&mut app, input).await {
            eprintln!("{} {err:#}", zephyr::app::render::styled("error:", "1;31"));
        }
    }

    println!("\nbye.");
    Ok(())
}

/// 读取一行用户输入。
///
/// 以 `\` 结尾表示续行（多行输入拼成一次提问）。
fn read_input() -> anyhow::Result<Option<String>> {
    let mut buffer = String::new();
    let terminal = io::stdout().is_terminal();

    loop {
        if terminal {
            if buffer.is_empty() {
                print!("\x1b[1;36myou>\x1b[0m ");
            } else {
                print!("\x1b[1;36m...>\x1b[0m ");
            }
            io::stdout().flush()?;
        }

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            if buffer.is_empty() {
                return Ok(None);
            }
            break;
        }

        let line = line.trim_end_matches(['\r', '\n']);
        if let Some(prefix) = line.strip_suffix('\\') {
            buffer.push_str(prefix);
            buffer.push('\n');
            continue;
        }

        buffer.push_str(line);
        break;
    }

    Ok(Some(buffer))
}
