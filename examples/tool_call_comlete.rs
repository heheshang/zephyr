use anyhow::Result;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use zephyr::{constant::QWEN_MODEL, llm::complete_tools::chat_complete, tools::build_toolbox};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let toolbox = build_toolbox().await?;

    let system_prompt = concat!(
        "你是一个全能助手。",
        "当你需要当前信息时，请使用搜索工具。",
        "当你需要计算时，请使用计算器工具。",
        "工具返回结果后，请直接使用这些结果回答，不要说不知道。"
    );

    // 测试 1：需要实时信息（触发 web_search）
    println!("\n=== 测试 1：实时信息查询 ===");
    let result = chat_complete(
        QWEN_MODEL,
        Some(system_prompt),
        "2026年世界杯决赛的比分是多少？",
        &toolbox,
        "1234567890",
    )
    .await?;
    println!("回答: {result}");

    // 测试 2：需要计算（触发 calculator）
    println!("\n=== 测试 2：计算任务 ===");
    let result = chat_complete(
        QWEN_MODEL,
        Some(system_prompt),
        "123456789 乘以 987654321 等于多少？",
        &toolbox,
        "1234567890",
    )
    .await?;
    println!("回答: {result}");

    Ok(())
}
