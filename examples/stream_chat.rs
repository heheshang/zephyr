use anyhow::Result;
use tokio::task::JoinSet;
use tracing::{Instrument, Level};
use tracing_subscriber::FmtSubscriber;
use zephyr::llm::get_semaphore;
use zephyr::{constant::MODEL, llm::chat_stream_with_retry};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv()?;
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let prompts = vec![
        "用三句话解释 Rust 的所有权机制",
        "用三句话解释 Rust 的并发机制",
        "用三句话解释 Rust 的错误处理机制",
        "用三句话解释 Rust 的泛型机制",
        "用三句话解释 Rust 的宏机制",
        "用三句话解释 Rust 的异步机制",
    ];

    let mut set = JoinSet::new();
    for prompt in prompts {
        let span = tracing::info_span!("chat", prompt = prompt);
        set.spawn(
            async move {
                tracing::info!("\n{prompt}");
                let permit = get_semaphore().acquire().await?;
                let output = chat_stream_with_retry(
                    MODEL,
                    Some("你是一个全能助理，请根据用户的问题给出回答。"),
                    prompt,
                )
                .await?;
                tracing::info!("output: {output:#?}");
                drop(permit);
                Ok::<_, anyhow::Error>((prompt, output))
            }
            .instrument(span),
        );
    }

    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok((prompt, output))) => {
                tracing::info!("prompt: {prompt}, output: {output}");
            },
            Ok(Err(e)) => {
                tracing::error!("error: {e}");
            },
            Err(e) => {
                tracing::error!("error: {e}");
            },
        }
    }

    Ok(())
}
