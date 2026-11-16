use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTools, CreateChatCompletionRequestArgs,
};

use crate::tools::calculator::execute::{CalculatorArgs, calculator};

pub async fn chat_completion(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    tools: Vec<ChatCompletionTools>,
) -> anyhow::Result<String> {
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

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages.clone())
        .tools(tools.clone())
        .max_tokens(2048u32)
        .build()?;

    let response = client.chat().create(request).await?;
    tracing::info!("response: {:#?}", response);

    let resp = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choice in response"))?
        .message;

    match resp.tool_calls {
        Some(tool_calls) => {
            messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()?
                    .into(),
            );

            for tool_call in tool_calls {
                match tool_call {
                    ChatCompletionMessageToolCalls::Function(function_call) => {
                        let function_name = function_call.function.name;
                        let arguments = function_call.function.arguments;
                        tracing::info!("function_name: {function_name}, arguments: {arguments}");

                        if function_name == "calculator" {
                            let args: CalculatorArgs = serde_json::from_str(&arguments)?;
                            let result =
                                calculator(&args.operator, args.first_number, args.second_number);
                            let tool_result = match result {
                                Ok(value) => value.to_string(),
                                Err(error) => format!("Error: {error}"),
                            };
                            tracing::info!("tool_result: {tool_result}");
                            messages.push(
                                ChatCompletionRequestToolMessageArgs::default()
                                    .tool_call_id(function_call.id.clone())
                                    .build()?
                                    .into(),
                            );
                        }
                    }
                    _ => {
                        tracing::error!("Unsupported tool call: {:#?}", tool_call);
                    }
                }
            }

            let request = CreateChatCompletionRequestArgs::default()
                .model(model)
                .messages(messages.clone())
                .tools(tools.clone())
                .max_tokens(2048u32)
                .build()?;

            let response = client.chat().create(request).await?;

            let content = response
                .choices
                .into_iter()
                .next()
                .and_then(|choice| choice.message.content)
                .ok_or_else(|| anyhow::anyhow!("No choice in response"))?;

            tracing::info!("tool call content: {content}");
            return Ok(content);
        }
        None => {
            let content = resp
                .content
                .ok_or_else(|| anyhow::anyhow!("No content in response"))?;
            tracing::info!("no toll call content: {content}");
            return Ok(content);
        }
    }

    // if let Some(tool_calls) = resp.tool_calls {}

    // let content = resp
    //     .content
    //     .ok_or_else(|| anyhow::anyhow!("No content in response"))?;
    // tracing::info!("final content: {content}");
    // return Ok(content); // if let Some(tool_calls) = resp.tool_calls {}

    // let content = resp
    //     .content
    //     .ok_or_else(|| anyhow::anyhow!("No content in response"))?;
    // tracing::info!("final content: {content}");
    // return Ok(content);
}
