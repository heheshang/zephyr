use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTools, CreateChatCompletionRequestArgs,
};

use crate::{agent::context::ExecutionContext, session::model::Session, tools::ToolBox};

pub async fn chat_complete(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    toolbox: &ToolBox,
    session_id: &str,
) -> anyhow::Result<String> {
    let client = async_openai::Client::new();
    let mut messages = Vec::new();
    let session = Session::new(session_id.to_string(), None);
    let context = ExecutionContext::new(session);

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

    let tool_definitions: Vec<ChatCompletionTools> = toolbox
        .values()
        .filter_map(|t| match t.definition() {
            Ok(definition) => Some(definition),
            Err(error) => {
                tracing::warn!(
                    "Skip tool {}, failed to get its definition: {error}",
                    t.name()
                );
                None
            },
        })
        .collect();

    loop {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(messages.clone())
            .tools(tool_definitions.clone())
            .max_tokens(2048u32)
            .build()?;
        let response = client.chat().create(request).await?;
        tracing::info!("response: {:#?}", response);

        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choice in response"))?
            .message;

        if let Some(tool_calls) = message.tool_calls {
            messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()?
                    .into(),
            );

            for tool_call in tool_calls {
                if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                    let function_name = &function_call.function.name;
                    let arguments = &function_call.function.arguments;
                    tracing::info!("function_name: {function_name}, arguments: {arguments}");

                    let tool_result = match toolbox.get(function_name) {
                        Some(tool) => match tool.execute(arguments, &context).await {
                            Ok(result) => {
                                tracing::info!("tool_result: {result}");
                                result
                            },
                            Err(err) => {
                                let msg = format!("tool execute error: {err}");
                                tracing::error!("{}", msg);
                                msg
                            },
                        },
                        None => {
                            let msg = format!("tool not found: {function_name}");
                            tracing::error!("{}", msg);
                            msg
                        },
                    };

                    messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(function_call.id.clone())
                            .content(tool_result)
                            .build()?
                            .into(),
                    );
                }
            }
            // 这里不应该返回，应该继续循环，直到模型返回纯文本内容,体现llM 自我调整的能力
            // let request = CreateChatCompletionRequestArgs::default()
            //     .model(model)
            //     .messages(messages.clone())
            //     .tools(tool_definitions.clone())
            //     .max_tokens(2048u32)
            //     .build()?;

            // let response = client.chat().create(request).await?;

            // let content = response
            //     .choices
            //     .into_iter()
            //     .next()
            //     .and_then(|choice| choice.message.content)
            //     .ok_or_else(|| anyhow::anyhow!("No choice in response"))?;
            // tracing::info!("has tool call content: {content}");
            // return Ok(content);
        } else {
            let content = message
                .content
                .ok_or_else(|| anyhow::anyhow!("No content in response"))?;
            tracing::info!("final  content: {content}");
            return Ok(content);
        }
    }
}
