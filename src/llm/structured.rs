use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
};

use crate::models::action_plan::ActionPlan;

pub async fn chat_completion_structured(
    model: &str,
    system: Option<&str>,
    prompt: &str,
) -> anyhow::Result<ActionPlan> {
    let client = async_openai::Client::new();
    let mut messages = Vec::new();

    if let Some(system) = system {
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()?
                .into(),
        );
    };

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    );
    let schema = schemars::schema_for!(ActionPlan);
    let schema_json = schema.as_value().clone();
    let formatted_schema = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some("The action plan for the user".to_string()),
            name: "action_plan".into(),
            schema: schema_json,
            strict: Some(true),
        },
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages)
        .max_tokens(2048u32)
        .response_format(formatted_schema)
        .build()?;

    let response = client.chat().create(request).await?;
    tracing::info!("response: {:#?}", response);

    let plan: ActionPlan = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| anyhow::anyhow!("No content found"))
        .and_then(|content| {
            serde_json::from_str(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse content: {}", e))
        })
        .map_err(|e| anyhow::anyhow!("Failed to parse content: {}", e))?;

    Ok(plan)
}
