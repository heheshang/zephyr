use std::time::Duration;

use async_openai::types::chat::{ChatCompletionMessageToolCalls, ChatCompletionRequestMessage};
use serde_json::Value;

use crate::agent::{context::ExecutionContext, event::ToolResultStatus};

/// 工具调用的只读视图，用于在回调中向调用方暴露「即将执行什么」。
///
/// 内部字段全部是引用，因此可以廉价地 `Copy`，方便在多个回调之间传递。
#[derive(Clone, Copy)]
pub struct ToolCallView<'a> {
    pub tool_call_id: &'a str,
    pub name: &'a str,
    pub arguments: &'a Value,
}

#[async_trait::async_trait]
pub trait BeforeToolCallback: Send + Sync {
    async fn call(&self, context: &ExecutionContext, tool_call: ToolCallView<'_>)
    -> Option<String>;
}

#[async_trait::async_trait]
pub trait AfterToolCallback: Send + Sync {
    async fn call(
        &self,
        context: &ExecutionContext,
        tool_call_id: &str,
        tool_name: &str,
        status: ToolResultStatus,
        content: &str,
    ) -> Option<(ToolResultStatus, String)>;
}

/// 大模型调用前的只读视图。
///
/// 暴露本次要发给模型的模型名、完整消息列表以及当前步数（0 表示第一轮），
/// 便于在回调里做日志、埋点、限流、缓存查询等。
///
/// 字段全部是引用，因此可以廉价地 `Copy`。
#[derive(Clone, Copy)]
pub struct LlmRequestView<'a> {
    /// 本次请求使用的模型名。
    pub model: &'a str,
    /// 本次请求下发给模型的完整消息列表（system + 历史 user/assistant/tool）。
    pub messages: &'a [ChatCompletionRequestMessage],
    /// 当前步数（从 0 开始计数），即本轮是第几次调用模型。
    pub step: u32,
}

/// 大模型调用后的只读视图。
///
/// 除了模型返回的文本与工具调用请求外，还带上了本次调用的耗时与 token 用量，
/// 便于在回调里做性能统计、成本核算或结果加工。
#[derive(Clone, Copy)]
pub struct LlmResponseView<'a> {
    /// 实际响应中回传的模型名。
    pub model: &'a str,
    /// 模型返回的文本内容；模型只请求调用工具时通常为 `None`。
    pub content: Option<&'a str>,
    /// 模型请求调用的工具列表。
    pub tool_calls: &'a [ChatCompletionMessageToolCalls],
    /// 本次调用的输入 token 数。
    pub prompt_tokens: u32,
    /// 本次调用的输出 token 数。
    pub completion_tokens: u32,
    /// 本次调用的总 token 数。
    pub total_tokens: u32,
    /// 本次调用（含重试）的墙钟耗时。
    pub elapsed: Duration,
}

/// 整个任务（一次 `run` / `run_structured`）结束后的结果状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOutcome {
    /// 任务正常结束，已产出最终答案。
    Success,
    /// 任务异常结束（模型报错、步数超限、结果反序列化失败等）。
    Failure { error: String },
}

/// 大模型调用**之前**的回调。
///
/// 在每次真正发起 HTTP 请求之前被调用（重试的每一次都会触发）。
/// 返回 `Some(content)` 表示**短路**：跳过本次模型调用，直接把 `content`
/// 当作模型的回复内容。典型用途：命中缓存直接返回、降级到兜底答案。
#[async_trait::async_trait]
pub trait BeforeLlmCallback: Send + Sync {
    async fn call(&self, context: &ExecutionContext, request: LlmRequestView<'_>)
    -> Option<String>;
}

/// 大模型调用**之后**的回调。
///
/// 在拿到模型响应之后被调用。返回 `Some(content)` 表示用 `content` 覆盖
/// 模型返回的文本内容（不影响工具调用请求）。典型用途：结果脱敏、
/// 输出格式化、记录审计日志。
#[async_trait::async_trait]
pub trait AfterLlmCallback: Send + Sync {
    async fn call(
        &self,
        context: &ExecutionContext,
        response: LlmResponseView<'_>,
    ) -> Option<String>;
}

/// 整个任务结束后的回调。
///
/// 无论任务是正常结束还是异常结束（含步数超限、模型报错），都会被触发一次。
/// 返回 `Some(content)` 表示用 `content` 覆盖最终输出（结构化模式下会尝试
/// 反序列化为 `T`，失败则保留原输出）。典型用途：生成执行摘要、落库、
/// 上报监控、打印统计信息。
#[async_trait::async_trait]
pub trait OnTaskEndCallback: Send + Sync {
    async fn call(&self, context: &ExecutionContext, outcome: &TaskOutcome) -> Option<String>;
}
