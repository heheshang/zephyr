//! # Agent 运行时（Runtime）
//!
//! 本模块实现了一个最小可用的 **ReAct 风格 Agent 循环**：模型在每一步要么「回答用户」，
//! 要么「请求调用工具」，工具执行结果会被回填进上下文，循环直到模型给出最终答案。
//!
//! ## 整体流程
//!
//! ```text
//!                    ┌──────────────────────────────┐
//!                    │  run() / run_structured()    │
//!                    └──────────────┬───────────────┘
//!                                   ▼
//!                    ┌──────────────────────────────┐
//!                    │ 1. 创建 ExecutionContext     │  ← 记录事件流 + token 用量
//!                    │    写入 user 事件            │
//!                    └──────────────┬───────────────┘
//!                                   ▼
//!                    ┌──────────────────────────────┐
//!                    │ 2. 收集工具定义 (schema)     │  ← 来自 ToolBox
//!                    └──────────────┬───────────────┘
//!                                   ▼
//!                    ┌──────────────────────────────┐
//!                    │ 3. while step < max_steps    │◄────────────┐
//!                    └──────────────┬───────────────┘             │
//!                                   ▼                             │
//!                    ┌──────────────────────────────┐             │
//!                    │ 4. build_messages()           │             │
//!                    │    事件流 → OpenAI 消息列表   │             │
//!                    └──────────────┬───────────────┘             │
//!                                   ▼                             │
//!                    ┌──────────────────────────────┐             │
//!                    │ 5. 调用 LLM（指数退避重试 3 次）│            │
//!                    └──────────────┬───────────────┘             │
//!                                   ▼                             │
//!                    ┌──────────────────────────────┐             │
//!                    │ 6. 累加 token 用量            │             │
//!                    └──────────────┬───────────────┘             │
//!                                   ▼                             │
//!                    ┌──────────────────────────────┐             │
//!                    │ 7. 有 tool_calls ?            │             │
//!                    │    ├─ 是 → 记录调用 + 执行工具 │─────────────┤
//!                    │    └─ 否 → 记录 assistant 消息  │             │
//!                    │             返回最终答案 ◄─────┘(终止)        │
//!                    └────────────────────────────────┘             │
//!                                   │                             │
//!                                   └── step += 1 ─────────────────┘
//! ```
//!
//! ## 两种运行模式
//!
//! | 模式 | 入口 | 结束条件 | 输出 |
//! |------|------|----------|------|
//! | 自由对话 | [`Agent::run`] | 模型返回无 `tool_calls` 的 assistant 消息 | `String` |
//! | 结构化输出 | [`Agent::run_structured`] | 模型调用 `final_answer` 工具 | 反序列化后的 `T` |
//!
//! 结构化模式通过向模型额外注入一个名为 `final_answer` 的工具，并强制
//! `tool_choice = required`，让模型必须以「调用工具」的方式提交符合 JSON Schema 的
//! 最终答案，从而绕开不同厂商对 `response_format: json_schema` 的兼容性差异。

use crate::{
    agent::{
        context::ExecutionContext,
        event::{ContentItem, Event, ToolResultStatus},
    },
    tools::ToolBox,
};
use anyhow::Result;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObjectArgs,
    ToolChoiceOptions,
};
use backon::{ExponentialBuilder, Retryable};
use std::{sync::Arc, vec};

/// 单次运行产生的文本输出与完整执行上下文。
///
/// - `output`：模型给出的最终答案（自由对话模式）。
/// - `context`：完整的事件流与 token 用量，便于事后审计/回放。
#[derive(Debug)]
pub struct AgentResult {
    pub output: String,
    pub context: ExecutionContext,
}

/// 结构化运行的结果：把 `final_answer` 工具参数反序列化成强类型 `T`。
#[derive(Debug)]
pub struct StructuredAgentResult<T> {
    pub output: T,
    pub context: ExecutionContext,
}

/// Agent 运行时的配置容器。
///
/// * `model`：模型名（OpenAI 兼容网关上的模型标识）。
/// * `instructions`：可选的 system prompt，会被放在消息列表最前面。
/// * `toolbox`：可用工具集合，以 `Arc` 共享，便于在多任务间复用。
/// * `max_steps`：Agent 循环的最大步数，防止无限调用模型/工具。
pub struct Agent {
    model: String,
    instructions: Option<String>,
    toolbox: Arc<ToolBox>,
    max_steps: u32,
}
impl Agent {
    /// 创建一个 Agent 实例，默认 `max_steps = 10`。
    pub fn new(
        model: impl Into<String>,
        instructions: Option<impl Into<String>>,
        toolbox: Arc<ToolBox>,
    ) -> Self {
        Self {
            model: model.into(),
            instructions: instructions.map(|i| i.into()),
            toolbox,
            max_steps: 10,
        }
    }

    /// 覆盖最大步数限制（builder 风格，可链式调用）。
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 自由对话模式：模型可以多轮调用工具，直到直接给出文本答案。
    ///
    /// 流程：
    /// 1. 新建 `ExecutionContext` 并写入用户消息事件；
    /// 2. 收集工具 JSON Schema；
    /// 3. 循环：拼消息 → 请求模型 → 若返回 tool_calls 则执行工具并回填，否则视为最终答案返回；
    /// 4. 每轮结束后累加 token 用量并递增步数，超过 `max_steps` 直接报错。
    pub async fn run(&self, user_input: &str) -> Result<AgentResult> {
        // ── 步骤 1：初始化上下文，并把用户输入作为第一条 user 事件写入事件流 ──
        let mut context = ExecutionContext::new();

        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

        // ── 步骤 2：创建 OpenAI 兼容客户端（默认读取环境变量 OPENAI_API_KEY / OPENAI_BASE_URL）──
        let client = async_openai::Client::new();

        // ── 步骤 3：把工具箱中每个工具的定义转成 OpenAI 的 tool schema ──
        // 单个工具定义失败不阻断整体流程，只记录告警并跳过该工具。
        let tool_definitions = self
            .toolbox
            .values()
            .filter_map(|t| match t.definition() {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::warn!("Skip tool {}, failed to get its definition: {e}", t.name());
                    None
                },
            })
            .collect::<Vec<_>>();

        // ── 步骤 4：Agent 主循环 ──
        loop {
            // 4.1 步数守卫：超过上限说明模型陷入死循环，直接终止并报错。
            if context.current_step >= self.max_steps {
                tracing::warn!("Max steps reached, stopping execution");
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            // 4.2 把事件流转换成 OpenAI 消息数组（system + 历史 user/assistant/tool）。
            let message = self.build_messages(&context)?;

            // 4.3 组装请求：附带工具定义，让模型可以选择调用工具。
            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .tools(tool_definitions.clone())
                .max_tokens(2048_u32)
                .messages(message)
                .build()?;

            // 4.4 调用模型，失败时按指数退避最多重试 3 次。
            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;

            // 4.5 累加本轮 token 消耗，便于统计成本。
            if let Some(usage) = &response.usage {
                context.useage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            // 4.6 取第一个 choice 的消息（这里不做多分支采样）。
            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No response from the model"))?
                .message;

            // 4.7 分流：有工具调用 → 走工具循环；没有 → 就是最终答案。
            if let Some(tool_calls) = message.tool_calls {
                // 先把「assistant 决定调用哪些工具」记录进事件流，
                // 这样下一轮拼消息时才能还原成 assistant 的 tool_calls 消息。
                self.record_tool_calls(&mut context, &tool_calls);
                // 依次执行工具并把结果写入事件流。
                self.execute_tool_calls(&mut context, &tool_calls).await;
            } else {
                // 模型直接给出文本，视为最终答案。
                let content = message
                    .content
                    .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;

                // 记录 assistant 消息事件。
                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "agent",
                    vec![ContentItem::Message {
                        role: "assistant".to_string(),
                        content: content.clone(),
                    }],
                ));
                // 保存最终结果。
                context.final_result = Some(content.clone());
                return Ok(AgentResult {
                    output: content,
                    context,
                });
            }

            // 4.8 本轮结束，步数 +1，进入下一轮循环。
            context.increment_step();
        }
    }

    /// 把模型返回的工具调用记录为一条 `agent` 事件。
    ///
    /// 只处理 `Function` 类型的调用；参数解析失败时退化为 `Value::Null`，
    /// 以免因为一段非法 JSON 让整轮执行崩掉。
    fn record_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut call_items = Vec::new();
        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                let arguments: serde_json::Value =
                    serde_json::from_str(&function_call.function.arguments)
                        .unwrap_or(serde_json::Value::Null);
                call_items.push(ContentItem::ToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_call.function.name.clone(),
                    arguments,
                });
            }
        }
        context.add_event(Event::new(
            context.execution_id.clone(),
            "agent",
            call_items,
        ));
    }

    /// 顺序执行本轮的全部工具调用，并把每个结果收集成一条 `tool` 事件。
    ///
    /// 工具执行失败（不存在 / 返回 Err）不会中断整轮，而是转成
    /// `ToolResultStatus::Error` 的文本反馈给模型，让模型自行纠错重试。
    async fn execute_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut result_items = Vec::new();

        for tool_call in tool_calls {
            // 目前只支持 function 类型的调用，其他类型直接跳过。
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };
            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("Tool call: {function_name}({arguments})");

            // 查找工具并执行，产出 (状态, 内容)。
            let (status, content) = match self.toolbox.get(function_name) {
                Some(tool) => match tool.execute(arguments, context).await {
                    Ok(result) => {
                        tracing::info!("Tool result: {result}");
                        (ToolResultStatus::Success, result)
                    },
                    Err(err) => {
                        let msg = format!("Tool execution error: {err}");
                        tracing::error!("{msg}");
                        (ToolResultStatus::Error, msg)
                    },
                },
                None => {
                    let msg = format!("Tool execution error: unknown tool {function_name}");
                    tracing::error!("{msg}");
                    (ToolResultStatus::Error, msg)
                },
            };

            // 工具结果必须携带 tool_call_id，与模型之前的调用一一对应。
            result_items.push(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_name.clone(),
                status,
                content,
            });
        }

        context.add_event(Event::new(
            context.execution_id.clone(),
            "tool",
            result_items,
        ));
    }

    /// 把内部事件流翻译成 OpenAI 兼容的消息数组。
    ///
    /// 映射规则：
    /// - `instructions` → 一条 system 消息（位于最前）；
    /// - `Message{role}` → user / assistant 消息；
    /// - `ToolCall` → assistant 消息上的 `tool_calls` 字段（若上一条已是
    ///   assistant 消息则合并进去，避免产生多余的 assistant 消息）；
    /// - `ToolResult` → 一条 tool 消息（必须紧跟在其对应的 assistant tool_calls 之后）。
    fn build_messages(
        &self,
        context: &ExecutionContext,
    ) -> Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages = vec![];

        // system prompt：可选，充当 Agent 的人设/任务说明。
        if let Some(system) = &self.instructions {
            let meassge = ChatCompletionRequestSystemMessageArgs::default()
                .content(system.as_str())
                .build()?
                .into();
            messages.push(meassge);
        }

        // 顺序遍历事件流，逐条转成请求消息。
        for event in &context.events {
            for item in &event.content {
                match item {
                    ContentItem::Message { role, content } => {
                        // 只有 user 归为 user 消息，其余（assistant）归为 assistant 消息。
                        let message: ChatCompletionRequestMessage = if role == "user" {
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(content.clone())
                                .build()?
                                .into()
                        } else {
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .content(content.clone())
                                .build()?
                                .into()
                        };

                        messages.push(message);
                    },
                    ContentItem::ToolCall {
                        tool_call_id,
                        name,
                        arguments,
                    } => {
                        // 构造 OpenAI 的 tool_call 结构，参数需序列化成字符串。
                        let tool_call = ChatCompletionMessageToolCalls::Function(
                            ChatCompletionMessageToolCall {
                                id: tool_call_id.clone(),
                                function: FunctionCall {
                                    name: name.clone(),
                                    arguments: arguments.to_string(),
                                },
                            },
                        );

                        // 合并策略：若最后一条已经是 assistant 消息，直接追加 tool_calls；
                        // 否则新建一条只带 tool_calls 的 assistant 消息。
                        if let Some(ChatCompletionRequestMessage::Assistant(last)) =
                            messages.last_mut()
                        {
                            last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                        } else {
                            messages.push(
                                ChatCompletionRequestAssistantMessageArgs::default()
                                    .tool_calls(vec![tool_call])
                                    .build()?
                                    .into(),
                            );
                        }
                    },
                    ContentItem::ToolResult {
                        tool_call_id,
                        content,
                        ..
                    } => {
                        // tool 消息按 tool_call_id 与上一次的调用配对。
                        messages.push(
                            ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(tool_call_id.clone())
                                .content(content.clone())
                                .build()?
                                .into(),
                        );
                    },
                }
            }
        }

        Ok(messages)
    }

    /// 结构化输出模式：强制模型通过 `final_answer` 工具提交符合 `T` 的 JSON。
    ///
    /// 与 [`Agent::run`] 的差异：
    /// - 额外注入 `final_answer` 工具，参数 schema 由 `schemars` 从 `T` 生成；
    /// - 设置 `tool_choice = required`，保证模型一定返回 tool_calls；
    /// - 一旦检测到 `final_answer` 调用即反序列化并返回，不再继续循环。
    ///
    /// 泛型约束 `T: JsonSchema + DeserializeOwned` 保证既能生成 schema 又能反序列化。
    pub async fn run_structured<T>(
        &self,
        user_input: &str,
    ) -> anyhow::Result<StructuredAgentResult<T>>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        // 与自由模式一致：初始化上下文并写入用户消息。
        let mut context = ExecutionContext::new();

        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

        let client = async_openai::Client::new();

        // 收集常规工具定义，并追加 `final_answer` 工具。
        let mut tool_definitions: Vec<ChatCompletionTools> = self
            .toolbox
            .values()
            .filter_map(|t| match t.definition() {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::warn!("Skip tool {}, failed to get its definition: {e}", t.name());
                    None
                },
            })
            .collect();
        tool_definitions.push(final_answer_tool_definition::<T>()?);

        loop {
            // 步数守卫，与自由模式一致。
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            // 拼装历史消息。
            let messages = self.build_messages(&context)?;

            // 组装请求：这次额外要求「必须调用工具」。
            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .tool_choice(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::Required,
                ))
                .max_tokens(2048u32)
                .build()?;

            // 调用模型（指数退避重试）。
            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;

            // 累加 token 用量。
            if let Some(useage) = &response.usage {
                context.useage.add(
                    useage.prompt_tokens,
                    useage.completion_tokens,
                    useage.total_tokens,
                );
            }

            // 取第一个 choice。
            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                .message;

            // 因为 tool_choice = required，这里没有 tool_calls 属于异常情况。
            let tool_calls = message.tool_calls.ok_or_else(|| {
                anyhow::anyhow!("Model returned no tool call despite tool_choice = required")
            })?;

            // 记录本轮工具调用事件。
            self.record_tool_calls(&mut context, &tool_calls);

            // 在所有调用中查找 `final_answer`。
            let final_call = tool_calls.iter().find_map(|tool_call| match tool_call {
                ChatCompletionMessageToolCalls::Function(f)
                    if f.function.name == "final_answer" =>
                {
                    Some(f)
                },
                _ => None,
            });

            // 命中 final_answer：反序列化为 T 并返回。
            if let Some(final_call) = final_call {
                let raw_arguments = final_call.function.arguments.clone();
                // JSON 不合法会在这里直接报错返回给调用方。
                let parsed: T = serde_json::from_str(&raw_arguments)?;

                // 同样把 final_answer 的结果写进事件流，保持事件流完整。
                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "tool",
                    vec![ContentItem::ToolResult {
                        tool_call_id: final_call.id.clone(),
                        name: "final_answer".to_string(),
                        status: ToolResultStatus::Success,
                        content: raw_arguments.clone(),
                    }],
                ));
                context.final_result = Some(raw_arguments);

                return Ok(StructuredAgentResult {
                    output: parsed,
                    context,
                });
            }

            // 没有 final_answer：正常执行其它工具，然后进入下一轮。
            self.execute_tool_calls(&mut context, &tool_calls).await;
            context.increment_step();
        }
    }
}

/// 依据泛型 `T` 的 `schemars` schema，生成名为 `final_answer` 的工具定义。
///
/// 模型通过调用该工具并传入符合 schema 的 JSON 参数来提交最终答案。
fn final_answer_tool_definition<T: schemars::JsonSchema>() -> anyhow::Result<ChatCompletionTools> {
    // 由 T 推导 JSON Schema。
    let schema = schemars::schema_for!(T);
    // 转成 serde_json::Value，便于塞进 FunctionObject 的 parameters 字段。
    let schema_json = serde_json::to_value(&schema)?;

    let function = FunctionObjectArgs::default()
        .name("final_answer")
        .description("Return the final structured answer matching the required schema.")
        .parameters(schema_json)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build final_answer tool definition: {e}"))?;

    Ok(ChatCompletionTools::Function(ChatCompletionTool {
        function,
    }))
}
