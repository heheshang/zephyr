use crate::gaia::models::GaiaOutput;
use async_openai::types::chat::ChatCompletionRequestSystemMessageArgs;
use async_openai::types::chat::ChatCompletionRequestUserMessageArgs;
use async_openai::types::chat::CreateChatCompletionRequestArgs;
use async_openai::types::chat::FinishReason;
use async_openai::types::chat::ResponseFormat;
use async_openai::types::chat::ResponseFormatJsonSchema;
use backon::ExponentialBuilder;
use backon::Retryable;

pub const GAIA_PROMPT: &str = r#"You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."#;

pub async fn solve_problem_with_retry(
    model: &str,
    system: &str,
    prompt: &str,
) -> anyhow::Result<GaiaOutput> {
    let op = || async { solve_problem(model, system, prompt).await };
    op.retry(ExponentialBuilder::default().with_max_times(3))
        .await
}

async fn solve_problem(model: &str, system: &str, prompt: &str) -> anyhow::Result<GaiaOutput> {
    let schema = schemars::schema_for!(GaiaOutput);
    let schema_json = serde_json::to_value(&schema)?;
    let formatted_schema = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some("GAIA problem solving output".into()),
            name: "gaia_output".into(),
            schema: schema_json,
            strict: Some(true),
        },
    };

    let client = async_openai::Client::new();
    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into(),
        ])
        .response_format(formatted_schema)
        .build()?;

    let response = client.chat().create(request).await?;
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No response from model"))?;

    if choice.finish_reason == Some(FinishReason::ContentFilter) {
        return Ok(GaiaOutput {
            is_solvable: false,
            unsolvable_reason: "Content filter".to_string(),
            final_answer: "".to_string(),
        });
    }
    let content = choice
        .message
        .content
        .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

    let gaia_output: GaiaOutput = serde_json::from_str(&content)?;

    Ok(gaia_output)
}
