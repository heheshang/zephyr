use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use tokio::task::JoinSet;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use zephyr::{
    constant::QWEN_MODEL,
    gaia::{
        GaiaEvalResult, evaluate_gaia_single, evaluatior::evaluate_gaia_single_with_tools,
        load_gaia_level1,
    },
    llm::get_semaphore,
    tools::build_toolbox,
};

// const MODELS: &[&str] = &["qwen-plus"];

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
    let toolbox = Arc::new(build_toolbox().await?);
    for problem in problems.clone().into_iter() {
        set.spawn(async move {
            let permit = get_semaphore().acquire().await;
            // let eval = evaluate_gaia_single(problem.clone(), QWEN_MODEL).await;
            let eval = evaluate_gaia_single(problem.clone(), QWEN_MODEL).await;
            tracing::info!("{eval:#?}");
            drop(permit);
            Ok::<_, anyhow::Error>(("without_tools", eval))
        });
    }

    for problem in problems.clone().into_iter() {
        let toolbox = toolbox.clone();
        set.spawn(async move {
            let permit = get_semaphore().acquire().await;
            // let eval = evaluate_gaia_single(problem.clone(), QWEN_MODEL).await;
            let eval =
                evaluate_gaia_single_with_tools(problem.clone(), QWEN_MODEL, toolbox, "1234567890")
                    .await;
            tracing::info!("{eval:#?}");
            drop(permit);
            Ok::<_, anyhow::Error>(("with_tools", eval))
        });
    }

    let mut results: HashMap<String, Vec<GaiaEvalResult>> = HashMap::new();
    while let Some(result) = set.join_next().await {
        let result = result?;
        match result {
            Ok((group, eval)) => {
                tracing::info!("group: {group}, eval: {eval:#?}");
                results.entry(group.to_string()).or_default().push(eval);
            },
            Err(e) => {
                tracing::error!("error evaluating problem: {e}");
            },
        }
    }
    tracing::info!("====带工具 vs 不带工具=====");
    for group in ["without_tools", "with_tools"] {
        let evals = results.get(group).unwrap();
        let correct = evals.iter().filter(|eval| eval.correct).count();
        let total = evals.len();
        tracing::info!(
            "group: {group}, correct: {correct}, total: {total}, accuracy: {:.2}%",
            correct as f64 / total as f64 * 100.0
        );
    }

    Ok(())
}
