use crate::agent::{ExecutionContext, event::ContentItem};

#[derive(Debug, Clone, Default)]
pub struct LlmRequest {
    pub instructions: Vec<String>,
    pub contents: Vec<ContentItem>,
}
impl LlmRequest {
    pub fn append_instructions(&mut self, instruction: impl Into<String>) {
        self.instructions.push(instruction.into());
    }
}

/// 大模型调用**之前**的回调。
///
/// 在每次真正发起 HTTP 请求之前被调用（重试的每一次都会触发）。
/// 返回 `Some(content)` 表示**短路**：跳过本次模型调用，直接把 `content`
/// 当作模型的回复内容。典型用途：命中缓存直接返回、降级到兜底答案。
#[async_trait::async_trait]
pub trait BeforeLlmCallback: Send + Sync {
    async fn call(&self, context: &mut ExecutionContext, request: &mut LlmRequest);
}
