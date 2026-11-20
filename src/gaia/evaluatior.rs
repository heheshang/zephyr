use std::sync::Arc;

use crate::{
    gaia::{
        GaiaOutput,
        models::{GaiaEvalResult, GaiaRow},
        solver::{GAIA_PROMPT, solve_problem_with_retry, solve_problem_with_tools},
    },
    tools::ToolBox,
};

fn is_correct(prediction: &str, answer: &str) -> bool {
    if prediction.is_empty() {
        false
    } else {
        prediction.trim().to_lowercase() == answer.trim().to_lowercase()
    }
}

fn to_eval_result(
    problem: GaiaRow,
    model: &str,
    result: Result<GaiaOutput, anyhow::Error>,
) -> GaiaEvalResult {
    match result {
        Ok(output) => GaiaEvalResult {
            task_id: problem.task_id,
            model: String::from(model),
            correct: is_correct(&output.final_answer, &problem.final_answer),
            is_solvable: Some(output.is_solvable),
            prediction: Some(output.final_answer),
            answer: problem.final_answer,
            unsolvable_reason: Some(output.unsolvable_reason),
            error: None,
        },

        Err(e) => GaiaEvalResult {
            task_id: problem.task_id,
            model: String::from(model),
            correct: false,
            is_solvable: None,
            prediction: None,
            answer: problem.final_answer,
            unsolvable_reason: None,
            error: Some(e.to_string()),
        },
    }
}

pub async fn evaluate_gaia_single(problem: GaiaRow, model: &str) -> GaiaEvalResult {
    let result = solve_problem_with_retry(model, GAIA_PROMPT, &problem.question).await;
    to_eval_result(problem, model, result)
}

pub async fn evaluate_gaia_single_with_tools(
    problem: GaiaRow,
    model: &str,
    toolbox: Arc<ToolBox>,
    session_id: &str,
) -> GaiaEvalResult {
    let result =
        solve_problem_with_tools(model, GAIA_PROMPT, &problem.question, toolbox, session_id).await;
    to_eval_result(problem, model, result)
}
