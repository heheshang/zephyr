use anyhow::Result;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use zephyr::{constant::QWEN_MODEL, llm::chat_completion_structured};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    let response = chat_completion_structured(
        QWEN_MODEL,
        Some("You are a helpful assistant that generates action plans for the user."),
        "I want to learn how to code in Rust.",
    )
    .await?;
    tracing::info!("response: {:#?}", response);
    Ok(())
}
