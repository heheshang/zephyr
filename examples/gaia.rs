use anyhow::Result;
use std::collections::HashMap;
use tokio::task::JoinSet;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use zephyr::{
    gaia::{GaiaEvalResult, evaluate_gaia_single, load_gaia_level1},
    llm::get_semaphore,
};

const MODELS: &[&str] = &["qwen-plus"];

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    gaia_level1_experiment().await?;

    Ok(())
}

pub async fn gaia_level1_experiment() -> Result<()> {
    let problems = load_gaia_level1().await?;
    tracing::info!("loaded {} problems", problems.len());
    let mut set = JoinSet::new();
    for model in MODELS {
        for problem in problems.clone().into_iter() {
            set.spawn(async move {
                let permit = get_semaphore().acquire().await;
                let eval = evaluate_gaia_single(problem.clone(), model).await;
                tracing::info!("{eval:#?}");
                drop(permit);
                Ok::<_, anyhow::Error>(eval)
            });
        }
    }

    let mut results: HashMap<String, Vec<GaiaEvalResult>> = HashMap::new();
    while let Some(result) = set.join_next().await {
        let result = result?;
        match result {
            Ok(eval) => {
                results.entry(eval.model.clone()).or_default().push(eval);
            },
            Err(e) => {
                tracing::error!("error evaluating problem: {e}");
            },
        }
    }

    for (model, evals) in results.iter() {
        let correct = evals.iter().filter(|eval| eval.correct).count();
        let total = evals.len();
        tracing::info!(
            "model: {model}, correct: {correct}, total: {total}, accuracy: {:.2}%",
            correct as f64 / total as f64 * 100.0
        );
    }

    Ok(())
}
