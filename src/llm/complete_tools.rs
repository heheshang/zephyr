use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTools, CreateChatCompletionRequestArgs,
};

use crate::{agent::context::ExecutionContext, session::model::Session, tools::ToolBox};

/// 单次对话允许的最大工具调用轮次，避免模型陷入死循环。
const MAX_TOOL_ROUNDS: usize = 8;

/// 单次请求允许的最大输出 token 数。
const MAX_TOKENS: u32 = 2048;

/// 支持工具调用的对话入口。
///
/// 流程：发起请求 → 若模型返回工具调用，则执行工具并把结果回灌到消息列表 → 再次请求，
/// 直到模型返回纯文本内容或达到 [`MAX_TOOL_ROUNDS`] 上限。
pub async fn chat_complete(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    toolbox: &ToolBox,
    session_id: &str,
) -> anyhow::Result<String> {
    let session = Session::new(session_id.to_string(), None);
    let client = async_openai::Client::new();
    let mut messages = build_messages(system, prompt)?;
    let tool_definitions = collect_tool_definitions(toolbox);

    for round in 0..MAX_TOOL_ROUNDS {
        let request = build_request(model, &messages, &tool_definitions)?;
        let response = client.chat().create(request).await?;
        tracing::info!("round {round}, response: {response:#?}");

        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choice in response"))?
            .message;
        // 如果模型没有返回工具调用，则直接返回内容
        let Some(tool_calls) = message.tool_calls else {
            let content = message
                .content
                .ok_or_else(|| anyhow::anyhow!("No content in response"))?;
            tracing::info!("no tool call content: {content}");
            return Ok(content);
        };

        // 把 assistant 的工具调用请求回灌进上下文，保证下一次请求的对话连贯。
        let mut assistant = ChatCompletionRequestAssistantMessageArgs::default();
        assistant.tool_calls(tool_calls.clone());
        if let Some(content) = message.content {
            assistant.content(content);
        }
        messages.push(assistant.build()?.into());

        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                let function_name = &function_call.function.name;
                let arguments = &function_call.function.arguments;
                tracing::info!("function_name: {function_name}, arguments: {arguments}");

                let tool_result = execute_tool(toolbox, function_name, arguments, &session).await;

                messages.push(
                    ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(function_call.id.clone())
                        .content(tool_result)
                        .build()?
                        .into(),
                );
            }
        }
    }

    Err(anyhow::anyhow!(
        "Exceeded max tool rounds ({MAX_TOOL_ROUNDS}) without a final answer"
    ))
}

/// 执行单个工具调用，把成功结果或错误信息都转成可回灌给模型的文本。
async fn execute_tool(
    toolbox: &ToolBox,
    function_name: &str,
    arguments: &str,
    session: &Session,
) -> String {
    let Some(tool) = toolbox.get(function_name) else {
        let msg = format!("tool not found: {function_name}");
        tracing::error!("{msg}");
        return msg;
    };
    let context = ExecutionContext::new(session.clone());
    match tool.execute(arguments, &context).await {
        Ok(result) => {
            tracing::info!("tool_result: {result}");
            result
        },
        Err(err) => {
            let msg = format!("tool execute error: {err}");
            tracing::error!("{msg}");
            msg
        },
    }
}

/// 构造初始消息列表（可选 system + 一条 user）。
fn build_messages(
    system: Option<&str>,
    prompt: &str,
) -> anyhow::Result<Vec<async_openai::types::chat::ChatCompletionRequestMessage>> {
    let mut messages = Vec::new();

    if let Some(system) = system {
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()?
                .into(),
        );
    }

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    );

    Ok(messages)
}

/// 收集工具箱中所有可用工具的定义，失败的工具只告警不阻断。
fn collect_tool_definitions(toolbox: &ToolBox) -> Vec<ChatCompletionTools> {
    toolbox
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
        .collect()
}

/// 根据当前消息列表与工具定义构造请求。
fn build_request(
    model: &str,
    messages: &[async_openai::types::chat::ChatCompletionRequestMessage],
    tools: &[ChatCompletionTools],
) -> anyhow::Result<async_openai::types::chat::CreateChatCompletionRequest> {
    CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages.to_vec())
        .tools(tools.to_vec())
        .max_tokens(MAX_TOKENS)
        .build()
        .map_err(Into::into)
}
