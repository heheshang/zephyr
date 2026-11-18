use std::sync::Arc;

use zephyr::{
    agent::runtime::Agent, callback::search_compressor::SearchCompressorCallback,
    constant::QWEN_MODEL, tools::build_toolbox,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let toolbox = Arc::new(build_toolbox().await?);

    let agent = Agent::new(
        QWEN_MODEL,
        Some("你是一个善用网页搜索的助手".to_string()),
        toolbox,
    )
    .with_max_steps(5)
    .with_after_tool_callback(Arc::new(SearchCompressorCallback));

    let result = agent.run("2026世界人工智能大会 WAIC 有什么亮点？").await?;
    println!("\n最终回答: {}", result.output);

    Ok(())
}
