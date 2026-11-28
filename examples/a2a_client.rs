//! # A2A 端到端测试客户端（`a2a_client`）
//!
//! 与 [`a2a_serve.rs`] 配套：它扮演**外部 A2A 客户端**的角色，
//! 用本仓库的 [`A2AClient`] 去连接 `a2a_serve` 启动的服务端。
//!
//! ## 定位
//!
//! `a2a_demo.rs` 在同一进程里同时扮演客户端和服务端，验证的是
//! **协议实现的正确性**。本文件扮演**纯外部客户端**，验证的是
//! **本系统 agent runtime 在 A2A 协议下的可用性**——
//! 也就是外部第三方系统能否真的把活儿派给本项目。
//!
//! 两者回答的问题不同，需要都跑。
//!
//! ## 测试覆盖
//!
//! | 场景 | 验证点 |
//! |---|---|
//! | 能力发现 | `/.well-known/agent-card.json` 可解析，技能清单完整 |
//! | 单轮问答 | 端到端产出可用答案 |
//! | agent 转移 | 根助手确实把活儿派给了子 agent（轨迹可观测） |
//! | 工具调用 | 子 agent 真实执行了工具，不是模型编的 |
//! | 多轮上下文 | 同一 context 下历史延续，agent 记得上文 |
//! | 任务查询 | `GetTask` 能取回已完成任务 |
//! | 任务列表 | 按 context 列出任务 |
//! | 任务取消 | 对已完成任务取消，语义正确 |
//! | 错误处理 | 不存在的任务 → 规范的错误码 |
//!
//! ## 用法
//!
//! ```bash
//! # 终端 1：启动服务端
//! cargo run --example a2a_serve -- --port 8899
//!
//! # 终端 2：跑客户端测试
//! cargo run --example a2a_client -- --url http://127.0.0.1:8899
//! ```

use zephyr::a2a::{
    client::A2AClient,
    types::{Task, TaskState},
};

/// 测试服务端地址。
const DEFAULT_URL: &str = "http://127.0.0.1:8899";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let (url, token) = parse_args();
    println!("\n========== A2A 端到端测试：目标 {url} ==========\n");

    let mut passed = 0usize;
    let mut failed = 0usize;

    // --------------------------------------------------------
    // TEST 1: 能力发现
    // --------------------------------------------------------
    println!("[TEST 1] 能力发现");
    let client = match discover(&url, token.as_deref()).await {
        Ok(client) => {
            println!("  ✓ 客户端发现成功");
            passed += 1;
            client
        }
        Err(error) => {
            println!("  ✗ 发现失败: {error}");
            failed += 1;
            return report(passed, failed);
        }
    };

    let Some(card) = client.card() else {
        println!("  ✗ 客户端未持有 Agent Card");
        failed += 1;
        return report(passed, failed);
    };
    println!("  · agent 名称    : {}", card.name);
    println!("  · 描述          : {}", card.description);
    println!("  · 技能数量      : {}", card.skills.len());
    for skill in &card.skills {
        println!("    - {} ({})", skill.id, skill.name);
    }
    println!("  · 流式支持      : {:?}", card.capabilities.streaming);

    // 期望：agent 树的关键能力都被登记为 skill，
    // 否则远端客户端根本不知道能派什么活儿。
    //
    // 新版卡片规则：子 agent → `{root}-{子agent}`，工具 → `{root}-tool-{工具名}`。
    let skill_ids: Vec<&str> = card.skills.iter().map(|s| s.id.as_str()).collect();
    for expected in ["assistant-researcher", "assistant-math", "assistant-coder"] {
        if skill_ids.contains(&expected) {
            println!("  ✓ 包含委派技能 {expected}");
            passed += 1;
        } else {
            println!("  ✗ 缺少委派技能 {expected}");
            failed += 1;
        }
    }

    // 技能描述不应是内部系统提示词——那是给模型看的，不是给客户端看的。
    let leaked = card.skills.iter().any(|skill| {
        skill.description.contains("工作规则") || skill.description.contains("transfer_to_agent")
    });
    check(
        !leaked,
        "技能描述未泄露内部提示词",
        &mut passed,
        &mut failed,
    );

    // --------------------------------------------------------
    // TEST 2: 单轮问答
    // --------------------------------------------------------
    println!("\n[TEST 2] 单轮问答");
    let task = client.send_text("用一句话介绍你能做什么").await?;
    println!("  · 任务状态      : {:?}", task.status.state);
    println!(
        "  · 答案          : {}",
        first_text(&task).chars().take(80).collect::<String>()
    );
    check(
        matches!(task.status.state, TaskState::Completed),
        "任务已完成",
        &mut passed,
        &mut failed,
    );

    // --------------------------------------------------------
    // TEST 3: agent 转移 + 工具调用（核心验证）
    // --------------------------------------------------------
    println!("\n[TEST 3] agent 转移与工具调用（需联网）");
    let task = client
        .send_text("Rust crate 的最新稳定版本是多少？请联网搜索确认并给出来源")
        .await?;

    let trace = execution_trace(&task);
    println!("  · 最终执行 agent: {}", final_agent(&task));
    println!("  · 转移历史      : {:?}", transfer_history(&task));
    for entry in &trace {
        println!("  · 轨迹          : {}", entry);
    }

    check(
        !transfer_history(&task).is_empty(),
        "发生了 agent 转移",
        &mut passed,
        &mut failed,
    );
    check(
        trace.iter().any(|entry| entry.contains("web_search")),
        "真实调用了 web_search 工具",
        &mut passed,
        &mut failed,
    );
    check(
        first_text(&task).contains("http"),
        "答案包含真实来源链接（未编造）",
        &mut passed,
        &mut failed,
    );

    // --------------------------------------------------------
    // TEST 4: 多轮上下文
    // --------------------------------------------------------
    println!("\n[TEST 4] 多轮上下文延续");
    let first = client.send_text("记住我的名字叫李四，别忘了").await?;
    let context_id = first.context_id.clone();
    println!("  · contextId     : {context_id:?}");

    let second = client
        .send_text_with("我叫什么名字？", Some(context_id.as_deref().unwrap_or("")))
        .await?;
    let answer = first_text(&second);
    println!(
        "  · 答案          : {}",
        answer.chars().take(60).collect::<String>()
    );

    check(
        second.context_id.as_deref() == context_id.as_deref(),
        "多轮沿用同一 contextId",
        &mut passed,
        &mut failed,
    );
    check(
        second.history.len() >= 2,
        "任务历史已累积",
        &mut passed,
        &mut failed,
    );
    check(
        answer.contains("李四"),
        "agent 记住了上一轮的内容",
        &mut passed,
        &mut failed,
    );

    // --------------------------------------------------------
    // TEST 5: 任务查询
    // --------------------------------------------------------
    println!("\n[TEST 5] 任务查询与列表");
    let task_id = second.id.clone();
    let fetched = client.get_task(&task_id).await?;
    println!("  · 取回任务      : {}", fetched.id);
    check(
        fetched.id == task_id,
        "GetTask 取回正确任务",
        &mut passed,
        &mut failed,
    );

    let listed = client.list_tasks(context_id.as_deref()).await?;
    println!("  · 该 context 任务数: {}", listed.len());
    check(
        !listed.is_empty(),
        "ListTasks 返回任务",
        &mut passed,
        &mut failed,
    );

    // --------------------------------------------------------
    // TEST 6: 任务取消语义
    // --------------------------------------------------------
    println!("\n[TEST 6] 任务取消");
    match client.cancel_task(&task_id).await {
        Ok(cancelled) => {
            println!("  · 取消后状态    : {:?}", cancelled.status.state);
            passed += 1;
        }
        Err(error) => {
            // 已完成任务不可取消是**正确**语义（规范 6.3 节）。
            let message = error.to_string();
            println!("  · 取消被拒绝    : {message}");
            check(
                message.contains("not in a cancelable state"),
                "已完成任务拒绝取消（规范语义）",
                &mut passed,
                &mut failed,
            );
        }
    }

    // --------------------------------------------------------
    // TEST 7: 错误处理
    // --------------------------------------------------------
    println!("\n[TEST 7] 错误处理");
    match client.get_task("does-not-exist").await {
        Ok(_) => {
            println!("  ✗ 不存在的任务竟然返回成功");
            failed += 1;
        }
        Err(error) => {
            println!("  · 错误          : {error}");
            check(
                error.to_string().contains("not found"),
                "不存在的任务返回规范错误",
                &mut passed,
                &mut failed,
            );
        }
    }

    report(passed, failed)
}

/// 解析命令行里的 `--url` 与 `--token`。
fn parse_args() -> (String, Option<String>) {
    let mut url = DEFAULT_URL.to_owned();
    let mut token: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--url" => {
                if let Some(value) = args.next() {
                    url = value;
                }
            }
            "--token" => {
                token = args.next();
            }
            "--help" | "-h" => {
                println!("用法: cargo run --example a2a_client -- [--url <服务端地址>] [--token <密钥>]");
                std::process::exit(0);
            }
            _ => {}
        }
    }
    (url, token)
}

/// 发现远端 agent；有密钥时带上 Bearer Token。
async fn discover(url: &str, token: Option<&str>) -> anyhow::Result<A2AClient> {
    match token {
        Some(token) => A2AClient::discover_with_auth(url, token).await,
        None => A2AClient::discover(url).await,
    }
}

/// 记录一条检查结果。
fn check(condition: bool, label: &str, passed: &mut usize, failed: &mut usize) {
    if condition {
        println!("  ✓ {label}");
        *passed += 1;
    } else {
        println!("  ✗ {label}");
        *failed += 1;
    }
}

/// 汇总测试结果。
fn report(passed: usize, failed: usize) -> anyhow::Result<()> {
    println!("\n========== 测试结果 ==========");
    println!("通过: {passed}");
    println!("失败: {failed}");
    if failed == 0 {
        println!("\n全部通过 ✓");
    } else {
        println!("\n存在失败项 ✗");
    }
    Ok(())
}

/// 取任务的第一个文本 part。
fn first_text(task: &Task) -> String {
    task.artifacts
        .iter()
        .flat_map(|artifact| artifact.parts.iter())
        .filter_map(|part| part.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// 从 artifact 的 metadata 里读出执行轨迹。
///
/// metadata 是服务端放进去的可观测信息，键缺失时返回空列表。
fn execution_trace(task: &Task) -> Vec<String> {
    task.artifacts
        .iter()
        .filter_map(|artifact| artifact.metadata.as_ref())
        .filter_map(|metadata| metadata.get("execution_trace"))
        .filter_map(|value| value.as_array())
        .flat_map(|entries| {
            entries.iter().map(|entry| {
                let agent = entry.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
                let kind = entry.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
                let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                format!("{agent} · {kind} · {name}")
            })
        })
        .collect()
}

/// 从 artifact 的 metadata 里读出转移历史。
fn transfer_history(task: &Task) -> Vec<String> {
    task.artifacts
        .iter()
        .filter_map(|artifact| artifact.metadata.as_ref())
        .filter_map(|metadata| metadata.get("transfer_history"))
        .filter_map(|value| value.as_array())
        .flat_map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str())
                .map(str::to_owned)
        })
        .collect()
}

/// 从 artifact 的 metadata 里读出最终执行 agent。
fn final_agent(task: &Task) -> String {
    task.artifacts
        .iter()
        .filter_map(|artifact| artifact.metadata.as_ref())
        .filter_map(|metadata| metadata.get("final_agent"))
        .filter_map(|value| value.as_str())
        .map(str::to_owned)
        .next()
        .unwrap_or_else(|| "未知".to_owned())
}
