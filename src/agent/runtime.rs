//! 带回调（callback）能力的 Agent 运行时。
//!
//! 与 `super::runtime` 中的基础 Agent 相比，本模块在「模型调用」与「工具调用」两个环节
//! 引入了四个扩展点：
//!
//! - [`BeforeLlmCallback`]：在大模型真正请求**之前**被调用。返回 `Some(content)` 表示
//!   短路（short-circuit）——跳过本次模型请求，直接把回调返回的 `content` 当作模型回复。
//!   典型用途：命中缓存、降级到兜底答案、限流拦截。
//! - [`AfterLlmCallback`]：在大模型返回**之后**被调用。返回 `Some(content)` 表示
//!   覆盖模型返回的文本内容。典型用途：结果脱敏、输出格式化、记录审计日志。
//! - [`BeforeToolCallback`]：在工具真正执行**之前**被调用。返回 `Some(content)` 表示
//!   短路（short-circuit）——本次工具不再真正执行，直接把回调返回的 `content` 当作工具结果。
//!   典型用途：人工审批（approval）、敏感操作拦截。
//! - [`AfterToolCallback`]：在工具执行**之后**被调用。返回 `Some((status, content))` 表示
//!   覆盖原本的执行状态与输出内容。典型用途：对工具结果做压缩、脱敏、二次加工。
//!
//! 此外还有 [`OnTaskEndCallback`]：在整个任务结束（无论成功或失败）后被调用一次，
//! 用于生成执行摘要、落库、上报监控等收尾工作。
//!
//! 整体执行流程是一个「模型 → 工具 → 模型 → 工具 → …… → 最终答案」的循环（ReAct 风格），
//! 每一步都会被记录成 [`Event`] 追加到 [`ExecutionContext`] 中，便于事后回放与审计。
//!
//! # 总体流程图
//!
//! ```mermaid
//! flowchart TD
//!     A["run(user_input) / run_structured(user_input)"] --> B["创建 ExecutionContext<br/>(execution_id / events / step=0)"]
//!     B --> C["写入 user 事件<br/>ContentItem::Message"]
//!     C --> D["从 toolbox 收集工具定义<br/>ChatCompletionTools"]
//!     D --> E{"current_step 尚未达到 max_steps ?"}
//!     E -- "否" --> E1["bail! 超过最大步数，未产出最终答案"]
//!     E -- "是" --> F["build_messages(context)<br/>事件流 → messages 数组"]
//!     F --> F1["触发 BeforeLlmCallback<br/>（返回 Some 则短路）"]
//!     F1 --> F2{"短路 ?"}
//!     F2 -- "是" --> F3["直接使用短路内容作为最终答案"]
//!     F2 -- "否" --> G["构造 ChatCompletionRequest<br/>model + messages + tools"]
//!     G --> H["调用 LLM（指数退避重试 ≤ 3 次）"]
//!     H --> I["累加 token 用量<br/>context.usage.add(...)"]
//!     I --> I1["触发 AfterLlmCallback<br/>（可覆盖返回文本）"]
//!     I1 --> J["取 choices[0].message"]
//!     J --> K{"消息里有 tool_calls ?"}
//!
//!     K -- "是（普通 run）" --> L["record_tool_calls<br/>写 assistant 事件"]
//!     L --> M["execute_tool_calls<br/>（含 before/after 回调）"]
//!     M --> N["increment_step()"]
//!     N --> E
//!
//!     K -- "是（structured）" --> L2{"含 final_answer ?"}
//!     L2 -- "否" --> M
//!     L2 -- "是" --> O["反序列化 arguments → T"]
//!     O --> P["写 tool 事件 + final_result"]
//!     P --> Q["返回 StructuredAgentResult"]
//!
//!     K -- "否（普通 run 才有）" --> R["取 message.content 作为最终答案"]
//!     R --> S["写 assistant 事件 + final_result"]
//!     S --> T["返回 AgentResult"]
//!     Q --> U["触发 OnTaskEndCallback<br/>（成功/失败都会触发）"]
//!     T --> U
//!     E1 --> U
//! ```
//!
//! # 单次模型调用内部流程图
//!
//! ```mermaid
//! flowchart TD
//!     L0["构造 ChatCompletionRequest"] --> L1["触发 BeforeLlmCallback"]
//!     L1 --> L2{"回调返回 Some(content) ?"}
//!     L2 -- "是" --> L3["ShortCircuit(content)<br/>跳过 HTTP 请求"]
//!     L2 -- "否" --> L4["发起请求（指数退避重试 ≤ 3 次）"]
//!     L4 --> L5["累加 token 用量"]
//!     L5 --> L6["触发 AfterLlmCallback"]
//!     L6 --> L7{"回调返回 Some(content) ?"}
//!     L7 -- "是" --> L8["覆盖 response 中的文本内容"]
//!     L7 -- "否" --> L9["沿用原响应"]
//!     L8 --> L10["返回 Response"]
//!     L9 --> L10
//!     L3 --> L10
//! ```
//!
//! # 单次工具调用内部流程图
//!
//! ```mermaid
//! flowchart TD
//!     S0["来自模型的一个 tool_call"] --> S1["解析 arguments → Value<br/>(失败降级为 Null)"]
//!     S1 --> S2["构造 ToolCallView<br/>(id / name / arguments)"]
//!     S2 --> S3{"遍历 before_tool_callbacks"}
//!     S3 --> S4{"回调返回 Some(content) ?"}
//!
//!     S4 -- "是 → 短路" --> S5["short_circuited = Some(content)<br/>工具不再真正执行"]
//!     S4 -- "否" --> S6["toolbox.get(name)"]
//!     S6 --> S7{"工具存在 ?"}
//!     S7 -- "否" --> S8["(Error, 'unknown tool ...')"]
//!     S7 -- "是" --> S9["tool.execute(arguments, context)"]
//!     S9 --> S10{"执行成功 ?"}
//!     S10 -- "是" --> S11["(Success, result)"]
//!     S10 -- "否" --> S12["(Error, 'Tool execution error: ...')<br/>错误回传给模型让它自我纠正"]
//!
//!     S5 --> S13["遍历 after_tool_callbacks"]
//!     S11 --> S13
//!     S8 --> S13
//!     S12 --> S13
//!     S13 --> S14{"回调返回 Some((status, content)) ?"}
//!     S14 -- "是" --> S15["覆盖 status 与 content"]
//!     S14 -- "否" --> S16["沿用原 status 与 content"]
//!     S15 --> S17["push ContentItem::ToolResult"]
//!     S16 --> S17
//!     S17 --> S18{"本轮所有工具都处理完 ?"}
//!     S18 -- "否" --> S0
//!     S18 -- "是" --> S19["合并为一条 tool 事件写回 context"]
//! ```
//!
//! # 消息构建流程图（build_messages）
//!
//! ```mermaid
//! flowchart TD
//!     M0["遍历 context.events"] --> M1{"ContentItem 变体"}
//!     M1 -- "Message(role == user)" --> M2["user 消息"]
//!     M1 -- "Message(其他 role)" --> M3["assistant 消息"]
//!     M1 -- "ToolCall" --> M4["构造 Function tool_call"]
//!     M4 --> M5{"上一条消息是 assistant ?"}
//!     M5 -- "是" --> M6["追加到它的 tool_calls<br/>（并行调用合并在同一条消息）"]
//!     M5 -- "否" --> M7["新建只含 tool_calls 的 assistant 消息"]
//!     M1 -- "ToolResult" --> M8["tool 消息<br/>用 tool_call_id 关联回调用"]
//!     M2 --> M9["push 进 messages"]
//!     M3 --> M9
//!     M6 --> M9
//!     M7 --> M9
//!     M8 --> M9
//! ```

use std::sync::Arc;

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    FunctionCall, FunctionObjectArgs, ToolChoiceOptions,
};
use backon::{ExponentialBuilder, Retryable};
use serde_json::Value;
use std::time::Instant;

use crate::{
    agent::callback::{
        AfterLlmCallback, AfterToolCallback, BeforeLlmCallback, BeforeToolCallback, LlmRequestView,
        LlmResponseView, OnTaskEndCallback, TaskOutcome, ToolCallView,
    },
    tools::ToolBox,
};

use super::{
    context::ExecutionContext,
    event::{ContentItem, Event, ToolResultStatus},
};

/// 普通（非结构化）Agent 一次执行结束后的结果。
///
/// - `output`：模型给出的最终文本答案。
/// - `context`：完整的执行上下文，包含所有事件（用户输入、工具调用、工具结果）与 token 用量。
#[derive(Debug)]
pub struct AgentResult {
    pub output: String,
    pub context: ExecutionContext,
}

/// 一次大模型调用的结果。
///
/// 区分「正常拿到响应」与「被模型调用前回调短路」两种情况，
/// 让上层主循环能分别走「解析响应」与「直接使用短路内容」两条分支。
enum LlmOutcome {
    /// 正常返回的模型响应。
    Response(async_openai::types::chat::CreateChatCompletionResponse),
    /// 被「模型调用前」回调短路，直接给出最终内容。
    ShortCircuit(String),
}

/// 结构化 Agent 一次执行结束后的结果。
///
/// 与 [`AgentResult`] 的区别仅在于 `output` 已经按 schema 反序列化成了强类型 `T`。
#[derive(Debug)]
pub struct StructuredAgentResult<T> {
    pub output: T,
    pub context: ExecutionContext,
}

/// 带回调支持的 Agent 实例。
///
/// 通过链式构造器（`with_*`）来配置：最大步数、执行前回调、执行后回调。
pub struct Agent {
    /// 使用的模型名（如 `qwen-plus`、`gpt-4o`），会原样下发给 OpenAI 兼容网关。
    model: String,
    /// 系统提示词（system prompt）。为 `None` 时请求中不带 system 消息。
    instructions: Option<String>,
    /// 工具箱，保存所有可用工具（共享所有权，便于在多个 Agent 间复用）。
    toolbox: Arc<ToolBox>,
    /// 最大循环步数。超过该步数仍未产出最终答案则报错终止，防止无限循环。
    max_steps: u32,
    /// 工具执行前回调，按注册顺序依次尝试，任一返回 `Some` 即短路。
    before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    /// 工具执行后回调，按注册顺序依次尝试，任一返回 `Some` 即覆盖结果。
    after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    /// 大模型调用前回调，按注册顺序依次尝试，任一返回 `Some` 即短路本次模型调用。
    before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
    /// 大模型调用后回调，按注册顺序依次尝试，任一返回 `Some` 即覆盖模型返回的文本。
    after_llm_callbacks: Vec<Arc<dyn AfterLlmCallback>>,
    /// 任务结束回调，任务正常或异常结束时都会触发一次。
    on_task_end_callbacks: Vec<Arc<dyn OnTaskEndCallback>>,
}

impl Agent {
    /// 创建一个 Agent。
    ///
    /// - `model`：模型名。
    /// - `instructions`：可选的系统提示词。
    /// - `toolbox`：共享的工具箱。
    pub fn new(
        model: impl Into<String>,
        instructions: Option<impl Into<String>>,
        toolbox: Arc<ToolBox>,
    ) -> Self {
        Self {
            model: model.into(),
            instructions: instructions.map(Into::into),
            toolbox,
            // 默认最多 10 步，防止模型陷入「反复调用工具」的死循环。
            max_steps: 10,
            before_tool_callbacks: Vec::new(),
            after_tool_callbacks: Vec::new(),
            before_llm_callbacks: Vec::new(),
            after_llm_callbacks: Vec::new(),
            on_task_end_callbacks: Vec::new(),
        }
    }

    /// 设置最大步数（消费 self，返回新的 Agent）。
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 注册一个「工具执行前」回调，可多次调用以注册多个。
    pub fn with_before_tool_callback(mut self, callback: Arc<dyn BeforeToolCallback>) -> Self {
        self.before_tool_callbacks.push(callback);
        self
    }

    /// 注册一个「工具执行后」回调，可多次调用以注册多个。
    pub fn with_after_tool_callback(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
        self.after_tool_callbacks.push(callback);
        self
    }

    /// 注册一个「大模型调用前」回调，可多次调用以注册多个。
    pub fn with_before_llm_callback(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
        self.before_llm_callbacks.push(callback);
        self
    }

    /// 注册一个「大模型调用后」回调，可多次调用以注册多个。
    pub fn with_after_llm_callback(mut self, callback: Arc<dyn AfterLlmCallback>) -> Self {
        self.after_llm_callbacks.push(callback);
        self
    }

    /// 注册一个「任务结束」回调，可多次调用以注册多个。
    ///
    /// 该回调在任务**正常结束或异常结束**时都会触发一次。
    pub fn with_on_task_end_callback(mut self, callback: Arc<dyn OnTaskEndCallback>) -> Self {
        self.on_task_end_callbacks.push(callback);
        self
    }

    /// 运行一次非结构化的 Agent 会话。
    ///
    /// 循环语义：
    /// 1. 用当前上下文构建消息列表，调用大模型；
    /// 2. 若模型返回 `tool_calls`，则记录并执行这些工具，把结果追加进上下文，继续下一轮；
    /// 3. 若模型返回纯文本内容，则认为这是最终答案，记录事件并返回。
    ///
    /// `max_steps` 用尽仍未拿到最终答案时返回错误。
    ///
    /// 无论成功还是失败，注册的 [`OnTaskEndCallback`] 都会被触发一次。
    pub async fn run(&self, user_input: &str) -> anyhow::Result<AgentResult> {
        // 每次运行都创建一份全新的执行上下文：包含本次执行的 UUID、事件流、步数、状态与用量。
        let mut context = ExecutionContext::new();

        // 把用户输入作为第一条事件写入上下文（author = "user"）。
        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

        // 主循环：模型 ⇄ 工具 交替执行。结果用 `Result` 承载，便于区分成功/失败并触发任务结束回调。
        let outcome = self.run_loop(&mut context).await;

        match outcome {
            Ok(output) => {
                // 任务正常结束：触发任务结束回调，回调可覆盖最终输出。
                let output = self
                    .trigger_task_end(&context, &TaskOutcome::Success)
                    .await
                    .unwrap_or(output);
                // 若回调改写了输出，同步更新上下文里的 final_result，保持一致。
                context.final_result = Some(output.clone());
                Ok(AgentResult { output, context })
            },
            Err(err) => {
                // 任务异常结束：同样触发任务结束回调，让调用方能记录失败原因。
                self.trigger_task_end(
                    &context,
                    &TaskOutcome::Failure {
                        error: err.to_string(),
                    },
                )
                .await;
                Err(err)
            },
        }
    }

    /// 非结构化模式的主循环（从 [`Agent::run`] 中抽出，便于统一处理错误）。
    ///
    /// 每轮流程：构建消息 → 触发「模型调用前」回调 → 调用模型（含重试）→
    /// 触发「模型调用后」回调 → 有 tool_calls 则执行工具，否则产出最终答案。
    async fn run_loop(&self, context: &mut ExecutionContext) -> anyhow::Result<String> {
        // 从工具箱收集所有工具的 JSON Schema 定义，随每次请求一起下发给模型。
        // 某个工具定义构造失败时只跳过该工具并告警，不影响整体运行。
        let tool_definitions: Vec<ChatCompletionTools> = self
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

        loop {
            // 步数上限检查。注意判断在进入循环前，因此实际最多执行 max_steps 次模型调用。
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            // 把上下文里的事件转换为 OpenAI 兼容的 messages 数组。
            let messages = self.build_messages(context)?;

            // 组装请求：模型名 + 消息 + 工具定义 + 最大输出 token。
            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .max_tokens(2048u32)
                .build()?;

            // 调用模型（内部会触发 before/after llm 回调，并累加 token 用量）。
            let outcome = self.call_llm(context, request).await?;

            match outcome {
                // 「模型调用前」回调短路：直接把它给的内容当作最终答案。
                LlmOutcome::ShortCircuit(content) => {
                    context.add_event(Event::new(
                        context.execution_id.clone(),
                        "agent",
                        vec![ContentItem::Message {
                            role: "assistant".to_string(),
                            content: content.clone(),
                        }],
                    ));
                    context.final_result = Some(content.clone());
                    return Ok(content);
                },
                // 正常拿到模型响应。
                LlmOutcome::Response(response) => {
                    // 取第一个 choice 的消息（贪心解码场景下只有一个候选）。
                    let message = response
                        .choices
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                        .message;

                    // 分支一：模型请求调用工具。
                    if let Some(tool_calls) = message.tool_calls {
                        // 先把「工具调用请求」记录进上下文（assistant 事件）。
                        self.record_tool_calls(context, &tool_calls);
                        // 执行工具（内部会触发 before/after 回调），并把结果写入上下文。
                        self.execute_tool_calls(context, &tool_calls).await;
                    } else {
                        // 分支二：模型直接给出文本 —— 视为最终答案。
                        let content = message
                            .content
                            .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;

                        // 记录 assistant 事件。
                        context.add_event(Event::new(
                            context.execution_id.clone(),
                            "agent",
                            vec![ContentItem::Message {
                                role: "assistant".to_string(),
                                content: content.clone(),
                            }],
                        ));
                        // 同时写入 final_result 字段，便于外部直接读取最终答案。
                        context.final_result = Some(content.clone());
                        return Ok(content);
                    }
                },
            }

            // 本轮结束，步数 +1，进入下一轮。
            context.increment_step();
        }
    }

    /// 运行一次结构化的 Agent 会话。
    ///
    /// 与 [`Agent::run`] 的关键差异：
    /// - 额外注册一个名为 `final_answer` 的工具，其参数 schema 由泛型 `T` 通过 `schemars` 生成；
    /// - 设置 `tool_choice = required`，强制模型必须以工具调用的形式回应；
    /// - 当模型调用 `final_answer` 时，把其 JSON 参数反序列化成 `T` 返回。
    ///
    /// 约束：`T` 必须实现 `schemars::JsonSchema`（用于生成 schema）与 `serde::de::DeserializeOwned`
    /// （用于反序列化结果）。
    ///
    /// 无论成功还是失败，注册的 [`OnTaskEndCallback`] 都会被触发一次。
    pub async fn run_structured<T>(
        &self,
        user_input: &str,
    ) -> anyhow::Result<StructuredAgentResult<T>>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        // 与 run 一样，每次调用都使用全新的上下文。
        let mut context = ExecutionContext::new();

        // 记录用户输入。
        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

        // 主循环，结果用 Result 承载，便于区分成功/失败并触发任务结束回调。
        let outcome = self.run_structured_loop::<T>(&mut context).await;

        match outcome {
            Ok(output) => {
                // 任务正常结束：触发任务结束回调。
                // 回调返回新内容时会尝试反序列化为 T；失败则保留原输出。
                let output = match self.trigger_task_end(&context, &TaskOutcome::Success).await {
                    Some(raw) => match serde_json::from_str::<T>(&raw) {
                        Ok(parsed) => parsed,
                        Err(e) => {
                            tracing::warn!(
                                "OnTaskEndCallback returned content that is not valid for the \
                                 expected schema, keeping the original output: {e}"
                            );
                            output
                        },
                    },
                    None => output,
                };
                Ok(StructuredAgentResult { output, context })
            },
            Err(err) => {
                // 任务异常结束：同样触发任务结束回调，让调用方能记录失败原因。
                self.trigger_task_end(
                    &context,
                    &TaskOutcome::Failure {
                        error: err.to_string(),
                    },
                )
                .await;
                Err(err)
            },
        }
    }

    /// 结构化模式的主循环（从 [`Agent::run_structured`] 中抽出）。
    ///
    /// 与 [`Agent::run_loop`] 的差异：请求强制带上 `tool_choice = required`，
    /// 并且额外注入了 `final_answer` 工具；当模型调用该工具时即视为产出最终答案。
    async fn run_structured_loop<T>(&self, context: &mut ExecutionContext) -> anyhow::Result<T>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        // 收集普通工具定义。
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
        // 追加专门用于产出结构化答案的 `final_answer` 工具。
        tool_definitions.push(final_answer_tool_definition::<T>()?);

        loop {
            // 步数上限检查。
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            let messages = self.build_messages(context)?;

            // 与 run 的差别：设置 tool_choice = Required，强制模型必须返回工具调用。
            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .tool_choice(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::Required,
                ))
                .max_tokens(2048u32)
                .build()?;

            // 调用模型（内部会触发 before/after llm 回调，并累加 token 用量）。
            let outcome = self.call_llm(context, request).await?;

            // 「模型调用前」回调短路：把内容当作 final_answer 的结果直接解析。
            let response = match outcome {
                LlmOutcome::ShortCircuit(content) => {
                    let parsed: T = serde_json::from_str(&content)?;
                    context.add_event(Event::new(
                        context.execution_id.clone(),
                        "tool",
                        vec![ContentItem::ToolResult {
                            tool_call_id: "short_circuit".to_string(),
                            name: "final_answer".to_string(),
                            status: ToolResultStatus::Success,
                            content: content.clone(),
                        }],
                    ));
                    context.final_result = Some(content);
                    return Ok(parsed);
                },
                LlmOutcome::Response(response) => response,
            };

            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                .message;

            // 因为 tool_choice = required，正常情况下必然有 tool_calls；否则视为协议异常。
            let tool_calls = message.tool_calls.ok_or_else(|| {
                anyhow::anyhow!("Model returned no tool call despite tool_choice = required")
            })?;

            // 记录工具调用事件。
            self.record_tool_calls(context, &tool_calls);

            // 在本轮所有工具调用中查找 `final_answer`。
            let final_call = tool_calls.iter().find_map(|tool_call| match tool_call {
                ChatCompletionMessageToolCalls::Function(f)
                    if f.function.name == "final_answer" =>
                {
                    Some(f)
                },
                _ => None,
            });

            // 找到 `final_answer`：其 arguments 即为最终结构化结果。
            if let Some(final_call) = final_call {
                let raw_arguments = final_call.function.arguments.clone();
                // 反序列化到强类型 T，失败则整个调用返回错误。
                let parsed: T = serde_json::from_str(&raw_arguments)?;

                // 以 tool 结果的形式记录，便于统一回放。
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

                return Ok(parsed);
            }

            // 否则视为普通的工具调用，执行后进入下一轮。
            self.execute_tool_calls(context, &tool_calls).await;
            context.increment_step();
        }
    }

    /// 调用一次大模型，并在调用前后触发对应的回调。
    ///
    /// 流程：
    /// 1. 触发「模型调用前」回调；任一回调返回 `Some(content)` 即短路，
    ///    直接把 `content` 当作模型回复，不再发起 HTTP 请求；
    /// 2. 发起请求（指数退避重试最多 3 次），并累加 token 用量；
    /// 3. 触发「模型调用后」回调；任一回调返回 `Some(content)` 即用它覆盖
    ///    模型返回的文本内容（不影响工具调用请求）。
    ///
    /// 返回值 [`LlmOutcome::ShortCircuit`] 表示被前置回调短路，
    /// [`LlmOutcome::Response`] 表示正常拿到模型响应。
    async fn call_llm(
        &self,
        context: &mut ExecutionContext,
        request: CreateChatCompletionRequest,
    ) -> anyhow::Result<LlmOutcome> {
        // 第一步：模型调用前回调。任一回调返回 Some 即短路本次调用。
        let request_view = LlmRequestView {
            model: &request.model,
            messages: &request.messages,
            step: context.current_step,
        };
        for callback in &self.before_llm_callbacks {
            if let Some(content) = callback.call(context, request_view).await {
                tracing::info!("BeforeLlmCallback short-circuited the model call");
                return Ok(LlmOutcome::ShortCircuit(content));
            }
        }

        // 第二步：真正发起请求。OpenAI 兼容客户端；base_url / api_key 从环境变量读取。
        let client = async_openai::Client::new();
        // 记录开始时间，用于在「模型调用后」回调中提供耗时。
        let started_at = Instant::now();
        // 网络请求带指数退避重试，最多重试 3 次；request 需要 clone 供多次尝试使用。
        let response = (|| async { client.chat().create(request.clone()).await })
            .retry(ExponentialBuilder::default().with_max_times(3))
            .await?;
        let elapsed = started_at.elapsed();

        // 累加 token 用量，便于统计整轮执行的成本。
        let usage = response.usage.clone();
        if let Some(usage) = &usage {
            context.usage.add(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            );
        }

        // 第三步：模型调用后回调。任一回调返回 Some 即覆盖模型返回的文本内容。
        // 回调里需要拿到「覆盖后的内容」，因此先把响应里的文本与工具调用拆出来。
        let (raw_content, tool_calls) = response
            .choices
            .first()
            .map(|choice| {
                (
                    choice.message.content.clone(),
                    choice.message.tool_calls.clone().unwrap_or_default(),
                )
            })
            .unwrap_or_else(|| (None, Vec::new()));

        let response_view = LlmResponseView {
            model: &response.model,
            content: raw_content.as_deref(),
            tool_calls: &tool_calls,
            prompt_tokens: usage.as_ref().map_or(0, |u| u.prompt_tokens),
            completion_tokens: usage.as_ref().map_or(0, |u| u.completion_tokens),
            total_tokens: usage.as_ref().map_or(0, |u| u.total_tokens),
            elapsed,
        };

        let mut overridden_content = None;
        for callback in &self.after_llm_callbacks {
            if let Some(content) = callback.call(context, response_view).await {
                overridden_content = Some(content);
                break;
            }
        }

        // 把可能被回调覆盖的文本写回响应对象，供上层分支判断。
        let mut response = response;
        if let (Some(content), Some(choice)) = (overridden_content, response.choices.first_mut()) {
            choice.message.content = Some(content);
        }

        Ok(LlmOutcome::Response(response))
    }

    /// 触发所有「任务结束」回调。
    ///
    /// 按注册顺序依次调用，任一回调返回 `Some(content)` 即以其作为覆盖内容并停止后续回调。
    /// 回调本身 panic 不会影响主流程的错误传播。
    async fn trigger_task_end(
        &self,
        context: &ExecutionContext,
        outcome: &TaskOutcome,
    ) -> Option<String> {
        let mut overridden = None;
        for callback in &self.on_task_end_callbacks {
            if let Some(content) = callback.call(context, outcome).await {
                overridden = Some(content);
                break;
            }
        }
        overridden
    }

    /// 把本轮的「工具调用请求」记录为一条 assistant 事件。
    ///
    /// 只有函数调用（`Function`）类型的 tool call 会被记录；参数解析失败时降级为 `Value::Null`，
    /// 以免因为一条坏数据让整轮执行失败。
    fn record_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut call_items = Vec::new();
        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                // 把 JSON 字符串形式的 arguments 解析为 Value，便于结构化记录与后续处理。
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

    /// 依次执行本轮的所有工具调用，并把结果记录为一条 tool 事件。
    ///
    /// 每个工具调用的处理顺序：
    /// 1. 解析参数，构造 [`ToolCallView`]；
    /// 2. 依次执行 `before_tool_callbacks`，若某个回调返回 `Some(content)` 则短路，不再真正执行工具；
    /// 3. 否则从工具箱查找并执行工具，把成功/失败统一转成 `(ToolResultStatus, String)`；
    /// 4. 依次执行 `after_tool_callbacks`，若某个回调返回 `Some((status, content))` 则覆盖状态与内容；
    /// 5. 把最终的 `(status, content)` 作为 `ContentItem::ToolResult` 收集起来。
    async fn execute_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut result_items = Vec::new();

        for tool_call in tool_calls {
            // 只处理函数调用类型，其余变体（如自定义工具）跳过。
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };
            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("Tool call: {function_name}({arguments})");

            // 把 arguments 解析成 Value，供回调读取（解析失败降级为 Null）。
            let arguments_value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
            // 回调看到的只读视图。
            let view = ToolCallView {
                tool_call_id: &function_call.id,
                name: function_name,
                arguments: &arguments_value,
            };

            // 第一步：执行前回调。任一回调返回 Some 即短路，直接使用其返回内容作为结果。
            let mut short_circuited = None;
            for callback in &self.before_tool_callbacks {
                if let Some(result) = callback.call(context, view).await {
                    short_circuited = Some(result);
                    break;
                }
            }

            // 第二步：真正执行工具（或使用短路结果），统一为 (状态, 内容) 元组。
            let (mut status, mut content) = match short_circuited {
                // 被回调短路：视为成功。
                Some(result) => (ToolResultStatus::Success, result),
                // 未短路：查找并执行工具。
                None => match self.toolbox.get(function_name) {
                    Some(tool) => match tool.execute(arguments, context).await {
                        Ok(result) => {
                            tracing::info!("Tool result: {result}");
                            (ToolResultStatus::Success, result)
                        },
                        Err(err) => {
                            // 工具执行失败不终止整个 Agent，而是以错误状态回传给模型，
                            // 让模型有机会自行纠正。
                            let msg = format!("Tool execution error: {err}");
                            tracing::error!("{msg}");
                            (ToolResultStatus::Error, msg)
                        },
                    },
                    // 工具箱中不存在该工具（模型幻觉调用了未提供的工具）。
                    None => {
                        let msg = format!("Tool execution error: unknown tool {function_name}");
                        tracing::error!("{msg}");
                        (ToolResultStatus::Error, msg)
                    },
                },
            };

            // 第三步：执行后回调。任一回调返回 Some 即覆盖（状态, 内容）并停止后续回调。
            for callback in &self.after_tool_callbacks {
                if let Some((new_status, new_content)) = callback
                    .call(context, &function_call.id, function_name, status, &content)
                    .await
                {
                    status = new_status;
                    content = new_content;
                    break;
                }
            }

            // 收集最终结果。
            result_items.push(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_name.clone(),
                status,
                content,
            });
        }

        // 所有工具结果合并为一条 tool 事件追加到上下文。
        context.add_event(Event::new(
            context.execution_id.clone(),
            "tool",
            result_items,
        ));
    }

    /// 把执行上下文中的事件序列转换为 OpenAI 兼容的 `messages` 数组。
    ///
    /// 转换规则：
    /// - `ContentItem::Message{role == "user"}` → user 消息；其余 role → assistant 消息；
    /// - `ContentItem::ToolCall` → assistant 消息的 `tool_calls` 字段。若上一条已经是 assistant
    ///   消息，则直接追加到它的 `tool_calls` 中（OpenAI 协议要求并行工具调用合并在同一条
    ///   assistant 消息里），否则新建一条 assistant 消息；
    /// - `ContentItem::ToolResult` → tool 消息，通过 `tool_call_id` 与对应的工具调用配对。
    ///
    /// 若配置了 `instructions`，则在最前面插入一条 system 消息。
    fn build_messages(
        &self,
        context: &ExecutionContext,
    ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages = Vec::new();

        // 系统提示词放在最前面。
        if let Some(system) = &self.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system.as_str())
                    .build()?
                    .into(),
            );
        }

        // 按事件顺序展开内容项。
        for event in &context.events {
            for item in &event.content {
                match item {
                    // 普通文本消息。
                    ContentItem::Message { role, content } => {
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
                    // 工具调用请求：合并进前一条 assistant 消息的 tool_calls。
                    ContentItem::ToolCall {
                        tool_call_id,
                        name,
                        arguments,
                    } => {
                        let tool_call = ChatCompletionMessageToolCalls::Function(
                            ChatCompletionMessageToolCall {
                                id: tool_call_id.clone(),
                                function: FunctionCall {
                                    name: name.clone(),
                                    arguments: arguments.to_string(),
                                },
                            },
                        );

                        // 若最后一条消息就是 assistant，则把该 tool_call 追加进去。
                        if let Some(ChatCompletionRequestMessage::Assistant(last)) =
                            messages.last_mut()
                        {
                            last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                        } else {
                            // 否则新建一条只含 tool_calls 的 assistant 消息。
                            messages.push(
                                ChatCompletionRequestAssistantMessageArgs::default()
                                    .tool_calls(vec![tool_call])
                                    .build()?
                                    .into(),
                            );
                        }
                    },
                    // 工具执行结果：作为 tool 消息，通过 tool_call_id 关联回调用请求。
                    ContentItem::ToolResult {
                        tool_call_id,
                        content,
                        ..
                    } => {
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
}

/// 为结构化输出生成 `final_answer` 工具的 JSON Schema 定义。
///
/// 通过 `schemars::schema_for!` 由泛型 `T` 生成 JSON Schema，再包成 OpenAI 的 function tool。
fn final_answer_tool_definition<T: schemars::JsonSchema>() -> anyhow::Result<ChatCompletionTools> {
    // 由类型 T 生成 JSON Schema（含 `$schema`、`$defs` 等元信息）。
    let schema = schemars::schema_for!(T);
    // 转成 serde_json::Value，便于作为 tool 的 parameters 字段下发。
    let schema_json = serde_json::to_value(&schema)?;

    // 组装 function 定义，工具名固定为 `final_answer`。
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
