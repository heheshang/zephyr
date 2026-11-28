//! 端到端验证：Agent 通过 `search_knowledge` 工具检索文档知识库。
//!
//! 与 `memory_demo.rs` 的区别：那个演示的是**长期记忆**（解题经验的自动注入），
//! 这个演示的是**文档知识库**（Agent 主动调工具检索资料）。
//!
//! 运行：`cargo run --example knowledge_tool_demo`

use std::sync::Arc;

use uuid::Uuid;
use zephyr::{
    agent::model::Agent,
    constant::{
        FINAL_ANSWER_TOOL_NAME, GPT_6_LUNA_PRO_MODEL, TEXT_EMBEDDING_3_SMALL_MODEL,
    },
    knowledge::knowledge_base::KnowledgeBaseManager,
    llm::llm_answer::TextAnswer,
    tool::{
        Tool, ToolBox, final_answer::FinalAnswerTool,
        knowledge_tool::{AddKnowledgeTool, SearchKnowledgeTool},
    },
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    // 用内存库：演示不依赖磁盘状态，跑完即弃
    let manager = Arc::new(
        KnowledgeBaseManager::new(":memory:", TEXT_EMBEDDING_3_SMALL_MODEL)?
            .with_chunking(200, 30),
    );

    // ============================================================
    // STEP 1: 预置资料
    // ============================================================
    println!("\n========== STEP 1: 预置资料 ==========\n");

    manager
        .add_document(
            "TCP 与 UDP",
            "https://example.com/network",
            "TCP 是面向连接的可靠传输协议，提供重传、排序、流量控制和拥塞控制。\
             UDP 是无连接的，一次发送一个数据报，不保证到达和顺序，但延迟更低、开销更小。\
             QUIC 在 UDP 之上实现了可靠性，相当于加密版的 TCP。",
        )
        .await?;
    manager
        .add_document(
            "Rust 所有权",
            "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html",
            "所有权是 Rust 特有的内存管理机制。每个值都有一个所有者(owner)。\
             当所有者离开作用域时，值会被丢弃。Rust 通过编译期检查确保没有内存泄漏、\
             没有悬垂指针、没有数据竞争。借用规则规定同一时刻要么有任意多个不可变借用，\
             要么有一个可变借用。",
        )
        .await?;

    println!("知识库共 {} 个切片", manager.total_chunks()?);

    // ============================================================
    // STEP 2: 装配工具
    // ============================================================
    let mut toolbox = ToolBox::new();
    toolbox.insert(
        FINAL_ANSWER_TOOL_NAME.to_owned(),
        Box::new(FinalAnswerTool::<TextAnswer>::new()),
    );
    toolbox.insert(
        "search_knowledge".to_owned(),
        Box::new(SearchKnowledgeTool::new(manager.clone())),
    );
    toolbox.insert(
        "add_knowledge".to_owned(),
        Box::new(AddKnowledgeTool::new(manager.clone())),
    );

    let agent = Agent::new(
        GPT_6_LUNA_PRO_MODEL.to_owned(),
        vec!["你是一个乐于助人的助手。请用中文回答。\
              回答涉及资料细节的问题前，先用 search_knowledge 工具查找依据，\
              再基于检索到的内容回答，并注明来源。"
            .to_owned()],
        Arc::new(toolbox),
    )
    .with_max_steps(8);

    // ============================================================
    // STEP 3: 问一个只有资料里才有答案的问题
    // ============================================================
    println!("\n========== STEP 2: Agent 检索知识库回答 ==========\n");

    let session = Uuid::new_v4().to_string();
    let result = agent
        .run::<TextAnswer>(
            "UDP 为什么比 TCP 延迟低?请说明理由。",
            &session,
            None,
        )
        .await?;

    println!("\n最终回答：\n{result}");

    // ============================================================
    // STEP 4: 验证 Agent 确实调用了工具、且入库工具可用
    // ============================================================
    println!("\n========== STEP 3: 校验 ==========\n");

    let used_search = result
        .context
        .session
        .events
        .iter()
        .flat_map(|event| event.content.iter())
        .any(|item| {
            matches!(
                item,
                zephyr::agent::event::ContentItem::ToolCall(call) if call.name == "search_knowledge"
            )
        });
    println!("Agent 调用了 search_knowledge: {used_search}");

    // 直接调用入库工具，验证 Agent 侧也能扩充知识库
    let add_tool = AddKnowledgeTool::new(manager.clone());
    let output = add_tool
        .execute(
            r#"{"title": "DNS", "source": "https://example.com/dns", "content": "DNS 把域名解析成 IP 地址。"}"#,
            &zephyr::agent::context::ExecutionContext::default(),
        )
        .await?;
    println!("add_knowledge 输出：{output}");
    println!("入库后共 {} 个切片", manager.total_chunks()?);

    Ok(())
}
