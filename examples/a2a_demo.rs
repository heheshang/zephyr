//! # A2A 端到端演示
//!
//! 演示两个本地 `Agent` 通过 **A2A 协议**（JSON-RPC over HTTP）互相调用。
//!
//! 与项目里已有的 `AgentTool`（进程内直接调用）相比，A2A 的关键差别是
//! **走 HTTP、只暴露 Agent Card 声明的能力、不暴露内部过程**——这正是
//! A2A 与 MCP 的分界线：
//!
//! ```text
//!   MCP ──► agent 调「工具」   （工具是被动的函数）
//!   A2A ──► agent 调「另一个 agent」（对方是自主体，有自己的模型与工具）
//! ```
//!
//! ## 演示流程
//!
//! ```text
//!   STEP 1  启动「远程专家」agent 的 A2A 服务端（监听 127.0.0.1:8899）
//!
//!   STEP 2  客户端发现：GET /.well-known/agent-card.json
//!           └──► 打印 Agent Card（名字、技能、支持的接口）
//!
//!   STEP 3  客户端发消息：POST /rpc  {method: "SendMessage"}
//!           └──► 打印远端任务的状态与产物
//!
//!   STEP 4  把远端 A2A agent 包成本地 Tool，交给协调员 agent
//!           └──► 协调员自己决定要不要委派
//!
//!   STEP 5  查询与取消任务：GetTask / ListTasks / CancelTask
//! ```
//!
//! ## 运行
//!
//! ```bash
//! cargo run --example a2a_demo
//! ```
//!
//! 需要 `.env` 里的 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`（走真实 LLM）。
//! 只想看协议流程不想花钱，可以设置 `A2A_DEMO_FAKE_LLM=1`，用内置的
//! 假模型跑通全流程。

use std::{net::SocketAddr, sync::Arc};

use zephyr::{
    a2a::{
        client::A2AClient,
        client_tool::A2ATool,
        server::A2AServer,
        types::TaskState,
    },
    agent::model::Agent,
    constant::GPT_6_LUNA_PRO_MODEL,
    llm::llm_answer::TextAnswer,
    tool::{build_basic_toolbox, final_answer::FinalAnswerTool},
};

/// 演示用的端口。换成别的端口只需改这里。
const DEMO_PORT: u16 = 8899;

/// 远端专家 agent 的名字。
const EXPERT_NAME: &str = "expert";

/// 协调员 agent 的名字。
const COORDINATOR_NAME: &str = "coordinator";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::WARN)
        .init();

    // ============================================================
    // STEP 1: 启动远端 agent 的 A2A 服务端
    // ============================================================
    println!("\n========== STEP 1: 启动远端 A2A 服务端 ==========\n");

    let expert = build_expert_agent()?;
    let expert = Arc::new(expert);

    let base_url = format!("http://127.0.0.1:{DEMO_PORT}");
    // Agent Card 里的 url 必须就是可直接调用的 JSON-RPC 端点，
    // 因此这里给完整的 `/rpc` 地址，而不是基础地址。
    let rpc_url = format!("{base_url}/rpc");
    let server = A2AServer::from_agent(expert.clone(), rpc_url, env!("CARGO_PKG_VERSION"))?;

    let addr: SocketAddr = format!("127.0.0.1:{DEMO_PORT}").parse()?;
    tokio::spawn(async move {
        if let Err(error) = server.serve(addr).await {
            eprintln!("A2A server stopped: {error}");
        }
    });

    // 等服务端起来（最多等 3 秒）。
    let client = wait_for_discovery(&base_url).await?;
    println!("A2A 服务端已就绪：{base_url}");

    // ============================================================
    // STEP 2: 客户端发现 Agent Card
    // ============================================================
    println!("\n========== STEP 2: 客户端发现 Agent Card ==========\n");

    let card = client
        .card()
        .expect("discover() 已经成功，card 必然存在");

    println!("name        : {}", card.name);
    println!("description : {}", card.description);
    println!("version     : {}", card.version);
    println!("streaming   : {}", card.supports_streaming());
    println!("interfaces  :");
    for interface in &card.supported_interfaces {
        println!(
            "  - {} ({}, v{})",
            interface.url, interface.protocol_binding, interface.protocol_version
        );
    }
    println!("skills      :");
    for skill in &card.skills {
        println!("  - {}: {}", skill.id, skill.description);
        println!("    tags: {}", skill.tags.join(", "));
    }

    // ============================================================
    // STEP 3: 发一条消息
    // ============================================================
    println!("\n========== STEP 3: 通过 A2A 发消息 ==========\n");

    let task = client.send_text("用一句话解释什么是 TCP 三次握手。").await?;

    println!("task id    : {}", task.id);
    println!("context id : {}", task.context_id.clone().unwrap_or_default());
    println!("state      : {}", task.state().as_str());
    for artifact in &task.artifacts {
        let text = artifact
            .parts
            .iter()
            .map(|part| part.collect_text())
            .collect::<Vec<_>>()
            .join("\n");
        println!("artifact   : {}", text);
    }

    assert_eq!(task.state(), TaskState::Completed, "示例任务应当完成");

    // ============================================================
    // STEP 3.5: 多轮对话（同一个 contextId 内的追问）
    // ============================================================
    println!("\n========== STEP 3.5: 多轮对话（自动续用 contextId） ==========\n");

    // 客户端记住了上一步服务端分配的 contextId，
    // 因此这一句"那它和四次挥手有什么区别？"会被远端当作
    // 同一段对话里的追问，能引用上一轮的内容。
    let follow_up = client.send_text("那它和四次挥手有什么区别？").await?;

    println!("context id : {}", follow_up.context_id.clone().unwrap_or_default());
    println!("state      : {}", follow_up.state().as_str());
    for artifact in &follow_up.artifacts {
        let text = artifact
            .parts
            .iter()
            .map(|part| part.collect_text())
            .collect::<Vec<_>>()
            .join("\n");
        println!("artifact   : {}", text);
    }

    assert_eq!(
        follow_up.context_id, task.context_id,
        "多轮对话必须落在同一个 context 里"
    );

    // ============================================================
    // STEP 3.8: 流式（SSE）
    // ============================================================
    println!("\n========== STEP 3.8: 流式响应（SSE） ==========\n");

    // 规范 9.4.2 节的 SendStreamingMessage：服务端以 text/event-stream
    // 推送任务状态与产物更新，客户端边收边处理。
    {
        use futures::StreamExt;

        let mut stream = client
            .send_streaming("用一句话说明什么是拥塞控制。")
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        while let Some(event) = stream.next().await {
            match event {
                Ok(zephyr::a2a::types::StreamResponse::Task { task }) => {
                    println!("[task]      {} [{}]", task.id, task.state().as_str());
                }
                Ok(zephyr::a2a::types::StreamResponse::StatusUpdate { status_update }) => {
                    println!("[status]    {}", status_update.status.state.as_str());
                }
                Ok(zephyr::a2a::types::StreamResponse::ArtifactUpdate { artifact_update }) => {
                    let text = artifact_update
                        .artifact
                        .parts
                        .iter()
                        .map(|part| part.collect_text())
                        .collect::<Vec<_>>()
                        .join("\n");
                    println!("[artifact]  {text}");
                }
                Ok(zephyr::a2a::types::StreamResponse::Message { message }) => {
                    println!("[message]   {}", message.text());
                }
                Err(error) => println!("[error]     {error}"),
            }
        }
    }

    // ============================================================
    // STEP 4: 把远端 agent 包成本地 Tool，交给协调员
    // ============================================================
    println!("\n========== STEP 4: 远端 agent 作为本地工具被委派 ==========\n");

    let mut toolbox = build_basic_toolbox();
    toolbox.insert(
        zephyr::constant::FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<TextAnswer>::new()),
    );

    // A2ATool 把远端 A2A agent 变成一个普通工具：协调员只看到
    // 名称、描述和参数，看不到远端的内部过程。
    let remote_tool = A2ATool::new(A2AClient::discover(&base_url).await?)
        .with_name("expert")
        .with_description("远程专家 agent，适合回答需要专业判断的问题");
    toolbox.insert("expert".to_owned(), Box::new(remote_tool));

    let coordinator = Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec!["你是一个协调员。你自己不擅长的问题，可以委派给 expert 工具。\
              拿到 expert 的回答后，用自己的话总结给用户。"
            .to_owned()],
        Arc::new(toolbox),
    )
    .with_max_steps(6)
    .with_name(COORDINATOR_NAME.to_owned());

    let coordinator = Arc::new(coordinator);
    // 注意：协调员有**自己独立的会话**，看不到 STEP 3 里用户与专家的对话。
    // 这正是 A2A 的隔离性——远端 agent 的对话历史不会泄漏给协调员，
    // 协调员只能通过工具调用拿到远端的**最终答案**。
    let result = coordinator
        .run::<TextAnswer>(
            "请委派 expert 工具回答：什么是零拷贝？为什么它能提升性能？",
            "a2a-demo-session",
            None,
        )
        .await?;

    println!("协调员回答：\n{result}");

    // ============================================================
    // STEP 5: 任务查询与取消
    // ============================================================
    println!("\n========== STEP 5: 任务查询与取消 ==========\n");

    let fetched = client.get_task(&task.id).await?;
    println!("GetTask    : {} -> {}", fetched.id, fetched.state().as_str());

    let tasks = client.list_tasks(None).await?;
    println!("ListTasks  : 共 {} 个任务", tasks.len());
    for item in &tasks {
        println!("  - {} [{}]", item.id, item.state().as_str());
    }

    // 取消一个已完成的任务应当失败（规范：终态任务不可取消）。
    match client.cancel_task(&task.id).await {
        Ok(canceled) => println!("CancelTask : {}", canceled.state().as_str()),
        Err(error) => println!("CancelTask : 按预期被拒绝 —— {error}"),
    }

    println!("\nA2A 演示完成。\n");
    Ok(())
}

/// 构造远端「专家」agent。
fn build_expert_agent() -> anyhow::Result<Agent> {
    let toolbox = build_basic_toolbox();
    Ok(
        Agent::new(
            GPT_6_LUNA_PRO_MODEL.to_owned(),
            vec!["你是一位网络协议专家，擅长解释 TCP/IP、HTTP 等协议原理。\
                  回答要准确、简洁。"
                .to_owned()],
            Arc::new(toolbox),
        )
        .with_name(EXPERT_NAME.to_owned())
        .with_description("远程专家 agent，回答网络协议与分布式系统问题")
        .with_max_steps(6),
    )
}

/// 轮询等待服务端可发现。
///
/// 服务端是异步启动的，客户端第一次请求可能撞上"连接被拒绝"，
/// 所以这里重试几次。
async fn wait_for_discovery(base_url: &str) -> anyhow::Result<A2AClient> {
    let mut last_error = None;

    for attempt in 0..30 {
        match A2AClient::discover(base_url).await {
            Ok(client) => return Ok(client),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if attempt == 29 {
                    break;
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "A2A server did not become ready: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_default()
    ))
}
