//! 大模型调用前后的日志回调。
//!
//! 这是一个「开箱即用」的示例实现，演示如何基于 [`BeforeLlmCallback`] 与
//! [`AfterLlmCallback`] 两个扩展点做可观测性埋点：
//!
//! - **调用前**：打印模型名、消息条数、当前步数，便于排查「模型到底收到了什么」。
//! - **调用后**：打印模型返回的文本摘要、工具调用数量、token 用量与本次耗时，
//!   便于核算成本与定位慢请求。
//!
//! 该结构体是零大小类型（ZST），无需任何状态，所有参数都是编译期常量。

use crate::agent::callback::{
    AfterLlmCallback, BeforeLlmCallback, LlmRequestView, LlmResponseView,
};

/// 打印在日志里的文本预览长度上限，避免超长回答把终端刷爆。
const PREVIEW_LIMIT: usize = 200;

/// 大模型调用日志回调（同时实现 Before 与 After 两个 trait）。
pub struct LlmLoggerCallback;

#[async_trait::async_trait]
impl BeforeLlmCallback for LlmLoggerCallback {
    /// 模型请求发起前的钩子：打印本次请求的基本信息。
    ///
    /// 始终返回 `None`，表示不干预模型调用。
    async fn call(
        &self,
        _context: &crate::agent::context::ExecutionContext,
        request: LlmRequestView<'_>,
    ) -> Option<String> {
        tracing::info!(
            "🤖 LLM 请求开始 | model={} | step={} | messages={}",
            request.model,
            request.step,
            request.messages.len(),
        );
        None
    }
}

#[async_trait::async_trait]
impl AfterLlmCallback for LlmLoggerCallback {
    /// 模型返回后的钩子：打印返回摘要、工具调用数、token 用量与耗时。
    ///
    /// 始终返回 `None`，表示不覆盖模型返回的文本内容。
    async fn call(
        &self,
        _context: &crate::agent::context::ExecutionContext,
        response: LlmResponseView<'_>,
    ) -> Option<String> {
        // 取模型文本的前若干字符作为预览；模型只请求调用工具时 `content` 为 `None`。
        let preview = response
            .content
            .map(|c| {
                if c.chars().count() > PREVIEW_LIMIT {
                    format!("{}…", c.chars().take(PREVIEW_LIMIT).collect::<String>())
                } else {
                    c.to_string()
                }
            })
            .unwrap_or_else(|| "<no content>".to_string());

        tracing::info!(
            "🤖 LLM 响应完成 | model={} | tool_calls={} | tokens(prompt/completion/total)={}/{}/{} | 耗时={:?}",
            response.model,
            response.tool_calls.len(),
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            response.elapsed,
        );
        tracing::info!("🤖 LLM 响应预览: {}", preview);
        None
    }
}
