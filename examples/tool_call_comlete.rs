use anyhow::Result;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use zephyr::{constant::MODEL, llm::complete_tools::chat_completion, tools::get_tools};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    let tools = get_tools();
    let response = chat_completion(
        MODEL,
        Some("你是一个全能助手"),
        "尼泊尔的首都是哪里",
        tools.clone(),
    )
    .await?;
    tracing::info!("response: {:#?}", response);

    let response = chat_completion(
        MODEL,
        Some("你是一个全能助手"),
        "5876乘以 675等于多少",
        tools.clone(),
    )
    .await?;
    tracing::info!("response: {:#?}", response);

    Ok(())
}
