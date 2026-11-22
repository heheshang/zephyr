use std::sync::Arc;

use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use zephyr::{agent::runtime::Agent, constant::QWEN_MODEL, tools::build_toolbox};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv()?;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let toolbox = Arc::new(build_toolbox().await?);
    let agent = Agent::new(QWEN_MODEL, Some("你是一个全能助手"), toolbox);

    let result = agent
        .run(
            Some("我的名字叫 kiven是一个软件开发工程师"),
            "multiple_run_agent",
            None,
        )
        .await?;

    println!("\n回答1: {:?}", result.output);

    println!("\n事件: {:?}", result.context.session.events);

    let result = agent
        .run(
            Some("我叫什么名字,我的职业是什么"),
            "multiple_run_agent",
            None,
        )
        .await?;

    println!("\n回答2: {:?}", result.output);

    println!("\n事件: {:?}", result.context.session.events);

    Ok(())
}
