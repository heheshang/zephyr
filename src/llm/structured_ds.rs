use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, ResponseFormat,
};

use crate::models::action_plan::ActionPlan;

pub async fn chat_completion_structured_ds(
    model: &str,
    system: Option<&str>,
    prompt: &str,
) -> anyhow::Result<ActionPlan> {
    let client = async_openai::Client::new();
    let mut messages = Vec::new();

    // DeepSeek 不支持 response_format=json_schema，只能用 json_object，
    // 因此把 schema 通过提示词注入，并要求模型只输出 JSON。
    let schema = schemars::schema_for!(ActionPlan);
    let schema_json = serde_json::to_string_pretty(&schema.as_value())?;

    let mut system_prompt = String::new();
    if let Some(system) = system {
        system_prompt.push_str(system);
        system_prompt.push('\n');
    }
    system_prompt.push_str(
        "You must reply with a single valid JSON object and nothing else. \
         Do not wrap the JSON in Markdown code fences and do not add any commentary. \
         The JSON object must strictly conform to the following JSON Schema:\n",
    );
    system_prompt.push_str(&schema_json);
    system_prompt.push_str("\n\nNote: the word \"json\" appears above; output raw JSON only.");

    messages.push(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt.as_str())
            .build()?
            .into(),
    );

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    );

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages)
        .max_tokens(2048u32)
        // DeepSeek 只支持 json_object（不支持 json_schema / strict）
        .response_format(ResponseFormat::JsonObject)
        .build()?;

    let response = client.chat().create(request).await?;
    tracing::info!("response: {:#?}", response);

    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or_else(|| anyhow::anyhow!("No content found"))?;

    let plan: ActionPlan = serde_json::from_str(&extract_json(&content))
        .map_err(|e| anyhow::anyhow!("Failed to parse content: {e}\nraw: {content}"))?;

    Ok(plan)
}

/// 从模型返回的文本中提取 JSON 对象。
/// 兼容模型偶尔返回 ```json ... ``` 代码块或前后夹带解释文字的情况。
fn extract_json(content: &str) -> String {
    let trimmed = content.trim();

    // 1) 代码块 ```json ... ``` / ``` ... ```
    if let Some(start) = trimmed.find("```") {
        let rest = &trimmed[start + 3..];
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }

    // 2) 直接是 JSON 对象/数组
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return trimmed.to_string();
    }

    // 3) 截取第一个 { 到最后一个 }
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}'))
        && start < end
    {
        return trimmed[start..=end].to_string();
    }

    trimmed.to_string()
}
