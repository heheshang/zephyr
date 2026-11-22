//! Agent 运行时：把「模型调用 → 工具执行 → 结果回灌」组织成一个有界的循环。
//!
//! # 模块职责
//!
//! 本模块是整个 `zephyr` Agent 的**执行内核**，负责：
//!
//! 1. 组装送入大模型的上下文（system prompt + 历史事件）；
//! 2. 发起 Chat Completion 请求，并带指数退避重试；
//! 3. 解析模型返回：普通文本回答 / 工具调用；
//! 4. 执行工具（支持前置/后置回调短路、危险工具人工确认）；
//! 5. 把工具结果回灌成消息，进入下一轮，直到拿到最终答案或超出 `max_steps`。
//!
//! 配套的两种运行模式：
//!
//! - [`Agent::run`]：自由模式，模型可以调用工具，也可以直接输出最终文本；
//! - [`Agent::run_structured`]：结构化模式，额外注入一个 `final_answer` 工具，
//!   并设置 `tool_choice = required`，强制模型必须以「调用工具」的方式给出
//!   符合 JSON Schema 的强类型结果。
//!
//! # 核心数据结构
//!
//! - [`ExecutionContext`]：一次执行的全部上下文，持有事件流、token 用量、当前步数；
//! - [`Event`] / [`ContentItem`]：事件流中的最小单位，`ContentItem` 有三种形态
//!   （普通消息、工具调用、工具结果），最终被翻译成 OpenAI 协议的 messages 数组；
//! - [`AgentStatus`]：运行终态，`Complete` 表示已有最终答案，
//!   `PendingConfirmation` 表示存在需要用户确认的危险工具调用，流程被**挂起**。
//!
//! # 主流程（自由模式 `run`）
//!
//! ```text
//!  run(user_input, session_id, tool_confirmations)
//!        │
//!        ├─ session_manager.get_or_create(session_id)
//!        ├─ ExecutionContext::new(session)
//!        ├─ 追加 user 事件（若给了 user_input）
//!        ├─ 注入 tool_confirmations 到 context.state（若给了确认结果）
//!        │
//!        ▼
//!  ┌───────────────────────────────────────────────────────────┐
//!  │ loop {                                                    │
//!  │   1. current_step >= max_steps ? → bail!("exceeded max steps") │
//!  │   2. prepare_llm_request(&mut context)                     │
//!  │        ├─ 消费 state 里的 pending_tool_calls + confirmations │
//!  │        │    → 执行被批准的工具，产出 ToolResult 事件          │
//!  │        ├─ 把 context.events() 摊平成 contents                │
//!  │        └─ 执行 before_llm_callbacks（可追加 instructions）    │
//!  │   3. build_messages(&llm_request) → OpenAI messages 数组     │
//!  │   4. client.chat().create(request) （指数退避重试 3 次）      │
//!  │   5. 累加 usage（prompt/completion/total tokens）            │
//!  │   6. 取 choices[0].message                                  │
//!  │        ├─ 有 tool_calls：                                    │
//!  │        │     record_tool_calls() → 记录 ToolCall 事件          │
//!  │        │     execute_tool_calls() → 执行/挂起工具             │
//!  │        │     若 state 里出现 pending_tool_calls：              │
//!  │        │        保存 session → return PendingConfirmation      │
//!  │        │     否则 increment_step() → 继续循环                 │
//!  │        └─ 无 tool_calls：                                    │
//!  │              记录 assistant 消息事件                          │
//!  │              保存 session → return Complete(output)          │
//!  └───────────────────────────────────────────────────────────┘
//! ```
//!
//! # 工具调用的两条分支
//!
//! ```text
//!  execute_tool_calls()
//!        │
//!        ▼
//!  对每个 tool_call:
//!        ├─ tool.requires_confirmation() == true ?
//!        │     └─ 是 → 记入 pending_tool_calls，本轮不执行，
//!        │           挂起整个 Agent，等用户确认后由下一次 run() 续跑
//!        └─ 否 → before_tool_callbacks
//!                 ├─ 某个回调返回 Some(x) → 短路，直接用 x 作为结果（Success）
//!                 └─ 都返回 None → 真正执行 tool.execute()
//!                        ├─ 成功 → Success
//!                        ├─ 工具报错 → Error("Tool execution error: ...")
//!                        └─ 工具不存在 → Error("unknown tool ...")
//!              → after_tool_callbacks（可改写 status/content）
//! ```
//!
//! # 人工确认的续跑协议
//!
//! ```text
//!  第一次 run()  ──▶ 遇到危险工具（如 delete）
//!         │              │
//!         │              ├─ context.state["pending_tool_calls"] = [...]   （JSON）
//!         │              └─ return AgentResult { status: PendingConfirmation,
//!         │                                 pending_tool_calls: [...] }
//!         ▼
//!    用户（外部）决定 approve / reject / 修改参数
//!         │
//!         ▼
//!  第二次 run() 传入 tool_confirmations
//!         │
//!         ▼
//!  prepare_llm_request() 检测到 state 里同时存在
//!  pending_tool_calls 与 tool_confirmations
//!         │
//!         ├─ approved=true  → 合并 modified_arguments 后真正执行工具
//!         ├─ approved=false → 生成一条 Error 的 ToolResult（附带拒绝原因）
//!         └─ 无对应确认     → 生成一条 Error 的 ToolResult（未获批准）
//!         ▼
//!    继续正常循环（工具结果回灌给模型）
//! ```

use std::{collections::HashMap, sync::Arc};

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObjectArgs,
    ToolChoiceOptions,
};
use backon::{ExponentialBuilder, Retryable};
use serde_json::Value;

use super::{
    context::ExecutionContext,
    event::{ContentItem, Event, ToolResultStatus},
};
use crate::{
    agent::{
        AfterToolCallback, BeforeToolCallback, ToolCallView,
        confirmation::{PendingToolCall, ToolConfirmation},
        event::ToolCall,
        llm_request::{BeforeLlmCallback, LlmRequest},
    },
    session::manager::{InMemorySessionManager, SessionManager},
    tools::ToolBox,
};

/// Agent 一次运行的终态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    /// 已经产出最终答案（`AgentResult::output` 为 `Some`）。
    Complete,
    /// 存在待人工确认的工具调用，流程被挂起；
    /// 调用方需要带着 [`ToolConfirmation`] 再次调用 `run()` 续跑。
    PendingConfirmation,
}

/// 自由模式运行的结果。
#[derive(Debug)]
pub struct AgentResult {
    /// 最终答案文本；`PendingConfirmation` 时为 `None`。
    pub output: Option<String>,
    /// 本次执行的完整上下文（事件流、token 用量等）。
    pub context: ExecutionContext,
    /// 运行终态。
    pub status: AgentStatus,
    /// 待确认的工具调用列表（仅 `PendingConfirmation` 时非空）。
    pub pending_tool_calls: Vec<PendingToolCall>,
}

/// 结构化模式运行的结果：把最终答案直接反序列化成了强类型 `T`。
#[derive(Debug)]
pub struct StructuredAgentResult<T> {
    /// 反序列化后的最终答案。
    pub output: T,
    /// 本次执行的完整上下文。
    pub context: ExecutionContext,
    /// 运行终态。
    pub status: AgentStatus,
}

/// Agent 执行器：持有一份工具箱与若干回调，在有界循环里驱动模型与工具协作。
pub struct Agent {
    /// 使用的模型名（对应 OpenAI 兼容网关上的 model id）。
    model: String,
    /// 静态 system 提示词，会作为第一条 system 消息下发给模型。
    instructions: Option<String>,
    /// 工具箱（工具名 → 工具实现），以 `Arc` 共享，便于回调内部取用。
    toolbox: Arc<ToolBox>,
    /// 循环步数上限，防止模型无限调工具。
    max_steps: u32,
    /// 工具执行**前**的回调链，任一返回 `Some` 即短路。
    before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
    /// 工具执行**后**的回调链，可改写结果状态与内容。
    after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    /// 模型调用**前**的回调链，可追加 system 指令或裁剪上下文。
    before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
    /// 会话管理器：负责按 `session_id` 持久化/恢复会话。
    session_manager: Box<dyn SessionManager>,
}

impl Agent {
    /// 创建 Agent。默认 `max_steps = 10`，使用内存会话管理器。
    pub fn new(
        model: impl Into<String>,
        instructions: Option<impl Into<String>>,
        toolbox: Arc<ToolBox>,
    ) -> Self {
        Self {
            model: model.into(),
            instructions: instructions.map(Into::into),
            toolbox,
            max_steps: 10,
            before_tool_callbacks: Vec::new(),
            after_tool_callbacks: Vec::new(),
            before_llm_callbacks: Vec::new(),
            session_manager: Box::new(InMemorySessionManager::new()),
        }
    }

    /// 设置最大循环步数（链式调用）。
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 注册「工具执行前」回调。回调返回 `Some(result)` 可短路工具执行。
    pub fn with_before_tool_callback(mut self, callback: Arc<dyn BeforeToolCallback>) -> Self {
        self.before_tool_callbacks.push(callback);
        self
    }

    /// 注册「工具执行后」回调。回调可返回 `(status, content)` 覆盖原结果。
    pub fn with_after_tool_callback(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
        self.after_tool_callbacks.push(callback);
        self
    }

    /// 注册「模型调用前」回调，用于注入额外 system 指令或压缩上下文。
    pub fn with_before_llm_callback(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
        self.before_llm_callbacks.push(callback);
        self
    }

    /// 替换会话管理器（链式调用），用于接入持久化存储。
    pub fn with_session_manager(mut self, session_manager: impl SessionManager + 'static) -> Self {
        self.session_manager = Box::new(session_manager);
        self
    }

    /// 自由模式主入口。
    ///
    /// - `user_input`：本轮用户输入；为 `None` 时表示这是「确认后续跑」的请求，
    ///   此时应通过 `tool_confirmations` 传入上一轮挂起的确认结果。
    /// - `session_id`：会话标识，用于恢复历史上下文。
    /// - `tool_confirmations`：对上一轮挂起工具调用的确认结果。
    pub async fn run(
        &self,
        user_input: Option<&str>,
        session_id: &str,
        tool_confirmations: Option<Vec<ToolConfirmation>>,
    ) -> anyhow::Result<AgentResult> {
        // 1. 取出（不存在则创建）会话，作为本次执行的历史基础。
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        // 2. 把用户输入记录成一个 user 事件，进入事件流。
        if let Some(input) = user_input {
            context.add_event(Event::new(
                context.execution_id.clone(),
                "user",
                vec![ContentItem::Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                }],
            ));
        }

        // 3. 若外部带来了确认结果，先暂存进 context.state，
        //    等下一次 `prepare_llm_request` 时与 pending_tool_calls 配对处理。
        if let Some(confirmations) = tool_confirmations {
            let value = serde_json::to_value(&confirmations)?;
            context
                .state_mut()
                .insert("tool_confirmations".to_string(), value);
        }

        // 4. 创建 OpenAI 兼容客户端（读取环境变量里的 base_url / api_key）。
        let client = async_openai::Client::new();

        // 5. 把工具箱里的所有工具翻译成协议层的 function 定义。
        //    某个工具定义构建失败不阻断整体，只记 warning 并跳过。
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
            // 6. 步数上限检查：防止模型在工具调用上无限打转。
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            // 7. 准备本轮请求：处理人工确认 → 摊平事件 → 跑 before_llm 回调。
            let llm_request = self.prepare_llm_request(&mut context).await?;
            let messages = self.build_messages(&llm_request)?;

            // 8. 构造请求体：模型 + messages + tools。
            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .max_tokens(2048u32)
                .build()?;

            // 9. 发起请求，失败时按指数退避最多重试 3 次。
            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;

            // 10. 累加 token 用量（整轮执行的总开销）。
            if let Some(usage) = &response.usage {
                context.usage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            // 11. 取第一个 choice 的消息。
            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                .message;

            if let Some(tool_calls) = message.tool_calls {
                // 12. 模型请求调用工具：先记录调用事件（写入事件流），
                //     再执行（或挂起）这些工具。
                self.record_tool_calls(&mut context, &tool_calls);
                self.execute_tool_calls(&mut context, &tool_calls).await;

                // 13. 若存在需要人工确认的工具调用 → 挂起，返回给调用方。
                if let Some(raw_pending) = context.state_mut().get("pending_tool_calls").cloned() {
                    let pending_tool_calls: Vec<PendingToolCall> =
                        serde_json::from_value(raw_pending)?;
                    self.session_manager.save(context.session.clone()).await?;
                    return Ok(AgentResult {
                        output: None,
                        context,
                        status: AgentStatus::PendingConfirmation,
                        pending_tool_calls,
                    });
                }
            } else {
                // 14. 模型没有调用工具 → 这就是最终答案，收尾返回。
                let content = message
                    .content
                    .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;

                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "agent",
                    vec![ContentItem::Message {
                        role: "assistant".to_string(),
                        content: content.clone(),
                    }],
                ));
                context.final_result = Some(content.clone());
                self.session_manager.save(context.session.clone()).await?;
                return Ok(AgentResult {
                    output: Some(content),
                    context,
                    status: AgentStatus::Complete,
                    pending_tool_calls: Vec::new(),
                });
            }

            // 15. 本轮结束，步数 +1，继续下一轮（工具结果已在事件流中，会被回灌）。
            context.increment_step();
        }
    }

    /// 结构化模式主入口：强制模型通过 `final_answer` 工具输出符合 schema 的结果。
    ///
    /// 与 [`Agent::run`] 的差异：
    /// - 额外注入 `final_answer` 工具，其 parameters 是 `T` 的 JSON Schema；
    /// - 设置 `tool_choice = required`，模型必须发起工具调用；
    /// - 找到 `final_answer` 调用后直接 `serde_json::from_str` 成 `T` 返回。
    pub async fn run_structured<T>(
        &self,
        user_input: &str,
        session_id: &str,
    ) -> anyhow::Result<StructuredAgentResult<T>>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        // 1. 准备会话与上下文。
        let session = self.session_manager.get_or_create(session_id, None).await?;
        let mut context = ExecutionContext::new(session);

        // 2. 记录用户输入。
        context.add_event(Event::new(
            context.execution_id.clone(),
            "user",
            vec![ContentItem::Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        ));

        // 3. 创建客户端。
        let client = async_openai::Client::new();

        // 4. 组装工具定义：普通工具 + `final_answer` 结构化工具。
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
            // 5. 步数上限检查。
            if context.current_step >= self.max_steps {
                anyhow::bail!(
                    "Agent exceeded the maximum of {} steps without a final answer",
                    self.max_steps
                );
            }

            // 6. 准备请求 + 构造 messages（与自由模式一致）。
            let llm_request = self.prepare_llm_request(&mut context).await?;
            let messages = self.build_messages(&llm_request)?;

            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                // 强制要求模型发起工具调用，从而保证最终一定走 `final_answer`。
                .tool_choice(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::Required,
                ))
                .max_tokens(2048u32)
                .build()?;

            // 7. 请求 + 指数退避重试。
            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;

            // 8. 累加 token 用量。
            if let Some(usage) = &response.usage {
                context.usage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            // 9. 取消息。
            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?
                .message;

            // 10. tool_choice=required 理论上必有 tool_calls，缺失即视为协议异常。
            let tool_calls = message.tool_calls.ok_or_else(|| {
                anyhow::anyhow!("Model returned no tool call despite tool_choice = required")
            })?;

            // 11. 记录工具调用事件。
            self.record_tool_calls(&mut context, &tool_calls);

            // 12. 在所有调用中寻找 `final_answer`。
            let final_call = tool_calls.iter().find_map(|tool_call| match tool_call {
                ChatCompletionMessageToolCalls::Function(f)
                    if f.function.name == "final_answer" =>
                {
                    Some(f)
                },
                _ => None,
            });

            if let Some(final_call) = final_call {
                // 13. 命中 final_answer：把 arguments 当作 JSON 解析成 T。
                let raw_arguments = final_call.function.arguments.clone();
                let parsed: T = serde_json::from_str(&raw_arguments)?;

                // 14. 记录一条成功状态的 tool_result 事件，便于审计。
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
                self.session_manager.save(context.session.clone()).await?;
                return Ok(StructuredAgentResult {
                    output: parsed,
                    context,
                    status: AgentStatus::Complete,
                });
            }

            // 15. 没有 final_answer → 说明模型还想继续调工具，执行后进入下一轮。
            self.execute_tool_calls(&mut context, &tool_calls).await;
            context.increment_step();
        }
    }

    /// 组装本轮送给模型的请求内容。
    ///
    /// 职责有三：
    /// 1. 若 state 中同时存在 `pending_tool_calls` 与 `tool_confirmations`，
    ///    说明上一轮因危险工具挂起，这里执行被批准的工具并产出结果事件；
    /// 2. 把事件流摊平成 `contents`（模型看到的「历史」）；
    /// 3. 执行 `before_llm_callbacks`，允许回调注入额外 system 指令或裁剪内容。
    async fn prepare_llm_request(
        &self,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<LlmRequest> {
        // 1. 人工确认续跑：两个 key 都存在才继续（remove 之后即消费掉）。
        if let (Some(raw_pending), Some(raw_confirmations)) = (
            context.state_mut().remove("pending_tool_calls"),
            context.state_mut().remove("tool_confirmations"),
        ) {
            let confirmation_results = self
                .process_confirmations(context, raw_pending, raw_confirmations)
                .await?;

            // 工具结果作为 "tool" 作者的事件写回事件流，供后续回灌给模型。
            if !confirmation_results.is_empty() {
                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "tool",
                    confirmation_results,
                ));
            }
        }

        // 2. 把事件流摊平成扁平的 content 列表。
        let mut request = LlmRequest {
            instructions: Vec::new(),
            contents: context
                .events()
                .iter()
                .flat_map(|ev| ev.content.iter().cloned())
                .collect(),
        };

        // 3. 模型调用前回调：可向 instructions 追加 system 提示，也可直接改 contents。
        for cb in &self.before_llm_callbacks {
            cb.call(context, &mut request).await;
        }

        Ok(request)
    }

    /// 处理人工确认结果：对每个挂起的工具调用，
    /// 根据确认决定执行 / 拒绝，并统一产出 `ContentItem::ToolResult`。
    async fn process_confirmations(
        &self,
        context: &ExecutionContext,
        raw_pending: Value,
        raw_confirmations: Value,
    ) -> anyhow::Result<Vec<ContentItem>> {
        // 1. 反序列化挂起列表与确认列表。
        let pending_calls: Vec<PendingToolCall> = serde_json::from_value(raw_pending)?;
        let confirmations: Vec<ToolConfirmation> = serde_json::from_value(raw_confirmations)?;

        // 2. 以 tool_call_id 为键索引确认结果，便于 O(1) 查找。
        let confirmation_map: HashMap<String, ToolConfirmation> = confirmations
            .into_iter()
            .map(|c| (c.tool_call_id.clone(), c))
            .collect();

        let mut results = Vec::with_capacity(pending_calls.len());

        // 3. 逐个处理挂起的工具调用。
        for pending in pending_calls {
            let tool_call_id = pending.tool_call.tool_call_id;
            let name = pending.tool_call.name;
            let confirmation = confirmation_map.get(&tool_call_id);

            let (status, content) = match confirmation {
                // 3.1 已批准：合并用户修改过的参数后真正执行。
                Some(c) if c.approved => {
                    let mut arguments = pending
                        .tool_call
                        .arguments
                        .as_object()
                        .cloned()
                        .unwrap_or_default();
                    // 用户修改的参数覆盖模型原始参数。
                    if let Some(overrides) = &c.modified_arguments {
                        arguments.extend(overrides.clone());
                    }
                    let args_json = Value::Object(arguments).to_string();

                    match self.toolbox.get(&name) {
                        Some(tool) => match tool.execute(&args_json, context).await {
                            Ok(result) => (ToolResultStatus::Success, result),
                            Err(err) => (
                                ToolResultStatus::Error,
                                format!("Tool execution error: {err}"),
                            ),
                        },
                        None => (
                            ToolResultStatus::Error,
                            format!("Tool execution error: unknown tool {name}"),
                        ),
                    }
                },
                // 3.2 明确拒绝：返回一条错误结果，附带拒绝原因（让模型知道被拒了）。
                Some(c) => {
                    let reason = c
                        .reason
                        .clone()
                        .unwrap_or_else(|| "Tool execution was rejected by user.".to_string());
                    (ToolResultStatus::Error, reason)
                },
                // 3.3 没有对应确认（超时/漏传）：同样视为未获批准。
                None => (
                    ToolResultStatus::Error,
                    "Tool execution was not approved.".to_string(),
                ),
            };

            results.push(ContentItem::ToolResult {
                tool_call_id,
                name,
                status,
                content,
            });
        }

        Ok(results)
    }

    /// 把模型返回的 tool_calls 记录为一条 `ContentItem::ToolCall` 事件。
    ///
    /// 注意：这里只记录「调用意图」，不代表已执行；执行结果由
    /// [`Agent::execute_tool_calls`] 另外写入 `ToolResult` 事件。
    fn record_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut call_items = Vec::new();
        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                // 参数是 JSON 字符串，解析失败就退化为 Null（不影响整体流程）。
                let arguments: serde_json::Value =
                    serde_json::from_str(&function_call.function.arguments)
                        .unwrap_or(serde_json::Value::Null);
                call_items.push(ContentItem::ToolCall(ToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_call.function.name.clone(),
                    arguments,
                }));
            }
        }
        context.add_event(Event::new(
            context.execution_id.clone(),
            "agent",
            call_items,
        ));
    }

    /// 执行一批工具调用，产出 `ToolResult` 事件。
    ///
    /// 三类特殊情况：
    /// - 需要人工确认的工具 → 不执行，记入 `pending_tool_calls` 并挂起；
    /// - 前置回调短路 → 直接把回调返回值当作成功结果；
    /// - 工具不存在 / 执行报错 → 生成 `Error` 状态的结果，不中断整批执行。
    async fn execute_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut result_items = Vec::new();
        let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();

        for tool_call in tool_calls {
            // 只处理 function 类型的调用，其他类型跳过。
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };
            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("Tool call: {function_name}({arguments})");

            let arguments_value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);

            // 1. 危险工具：需要用户确认，本轮不执行。
            if let Some(tool) = self.toolbox.get(function_name)
                && tool.requires_confirmation()
            {
                tracing::info!("Tool call requires confirmation: {function_name}");
                pending_tool_calls.push(PendingToolCall {
                    tool_call: ToolCall {
                        tool_call_id: function_call.id.clone(),
                        name: function_name.clone(),
                        arguments: arguments_value.clone(),
                    },
                    confirmation_message: tool.get_confirmation_message(&arguments_value),
                });
                continue;
            }

            // 2. 构造回调视图（借用 id/name/arguments，避免克隆开销）。
            let view = ToolCallView {
                tool_call_id: &function_call.id,
                name: function_name,
                arguments: &arguments_value,
            };

            // 3. 前置回调链：任一返回 Some 即短路，不再真正执行工具。
            let mut short_circuited = None;
            for callback in &self.before_tool_callbacks {
                if let Some(result) = callback.call(context, view).await {
                    short_circuited = Some(result);
                    break;
                }
            }

            // 4. 得到初始的 (状态, 内容)。
            let (mut status, mut content) = match short_circuited {
                Some(result) => (ToolResultStatus::Success, result),
                None => match self.toolbox.get(function_name) {
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
                },
            };

            // 5. 后置回调链：可改写状态与内容（例如脱敏、补充上下文）。
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

            result_items.push(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_name.clone(),
                status,
                content,
            });
        }

        // 6. 若有挂起的调用，写入 context.state，供下一次 `run()` 读取。
        if !pending_tool_calls.is_empty() {
            let value = serde_json::to_value(&pending_tool_calls).unwrap_or(Value::Null);
            context
                .state_mut()
                .insert("pending_tool_calls".to_string(), value);
        }

        // 7. 把本批工具结果作为一个整体事件写入事件流。
        context.add_event(Event::new(
            context.execution_id.clone(),
            "tool",
            result_items,
        ));
    }

    /// 把内部抽象的 [`LlmRequest`] 翻译成 OpenAI 协议的 messages 数组。
    ///
    /// 翻译规则：
    /// - 静态 `instructions` → 第一条 system 消息；
    /// - 回调追加的 `request.instructions` → 紧随其后的 system 消息；
    /// - `ContentItem::Message` → user / assistant 消息（按 role 区分）；
    /// - `ContentItem::ToolCall` → assistant 消息的 `tool_calls` 字段；
    ///   若上一条已经是 assistant 消息则合并进去，否则新建一条 assistant 消息；
    /// - `ContentItem::ToolResult` → `role = tool` 的消息，必须带 `tool_call_id`。
    fn build_messages(
        &self,
        request: &LlmRequest,
    ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages = Vec::new();

        // 1. 静态 system 提示词。
        if let Some(system) = &self.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system.as_str())
                    .build()?
                    .into(),
            );
        }

        // 2. 回调动态追加的 system 提示词。
        for extra_instruction in &request.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(extra_instruction.as_str())
                    .build()?
                    .into(),
            );
        }

        // 3. 事件流内容逐条翻译。
        for item in &request.contents {
            match item {
                ContentItem::Message { role, content } => {
                    // role == "user" 走 user 消息，其余一律按 assistant 处理。
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
                ContentItem::ToolCall(ToolCall {
                    tool_call_id,
                    name,
                    arguments,
                }) => {
                    // 工具调用必须挂在 assistant 消息的 tool_calls 上。
                    let tool_call =
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: tool_call_id.clone(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: arguments.to_string(),
                            },
                        });

                    // 合并到上一条 assistant 消息，避免生成多条只有 tool_calls 的消息。
                    if let Some(ChatCompletionRequestMessage::Assistant(last)) = messages.last_mut()
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
                    // 工具结果消息必须携带 tool_call_id，与调用一一对应。
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

        Ok(messages)
    }
}

/// 构造 `final_answer` 工具定义：参数 schema 直接来自泛型 `T` 的 JSON Schema。
///
/// 这样做的好处是：不同厂商对 `response_format: json_schema` 的支持差异很大，
/// 而「工具调用 + 工具参数 schema」是各家都支持的通用做法。
fn final_answer_tool_definition<T: schemars::JsonSchema>() -> anyhow::Result<ChatCompletionTools> {
    let schema = schemars::schema_for!(T);
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
