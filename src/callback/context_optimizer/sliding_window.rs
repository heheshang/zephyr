//! # 滑动窗口优化器（Sliding Window）
//!
//! ## 策略
//!
//! 当一次请求的 token 估算值超过 `token_threshold` 时，把历史消息裁剪到只剩
//! 最后 `window_size` 条，并把裁剪掉的旧消息**直接丢弃**（不做摘要）。
//!
//! ## 与其他优化器的差异
//!
//! - [`super::compaction`]：不丢内容，只把大段工具结果替换成占位文本；
//! - [`super::summarization`]：不丢语义，把旧历史浓缩成一段摘要放进 instructions。
//!
//! 滑动窗口是三者中**最激进**的一种：它节省的 token 最多，但会丢掉旧消息里的
//! 具体信息（尤其是早期工具调用与结论）。因此它被设计为"兜底手段"——只在
//! token 真的超标时才触发，通常排在其它优化器之后执行。
//!
//! ## 安全裁剪
//!
//! 裁剪起点不是简单的 `len - window_size`，而是由 [`find_safe_start`] 修正过的值，
//! 以保证不会出现"有 tool_result 却没有配对 tool_call"的情况（协议层会直接 400）。
//! 另外起点还会被强制不小于"第一条 user 消息之后"的位置，确保用户原始诉求不被裁掉。

use crate::{
    agent::{
        ExecutionContext,
        event::ContentItem,
        llm_request::{BeforeLlmCallback, LlmRequest},
    },
    callback::context_optimizer::{count_tokens, find_safe_start},
};

/// 滑动窗口优化器。
///
/// 字段均为公开的，便于在构造时按需调整：
/// - `model`: 用于 token 估算的模型名；
/// - `token_threshold`: 触发裁剪的 token 阈值；
/// - `window_size`: 裁剪后保留的**条目数**（不是 token 数）。
pub struct SlidingWindow {
    /// 参与 token 估算的模型名。
    pub model: String,
    /// token 估算值 ≥ 该阈值时触发裁剪。
    pub token_threshold: usize,
    /// 裁剪后保留的最近条目数量。
    pub window_size: usize,
}

#[async_trait::async_trait]
impl BeforeLlmCallback for SlidingWindow {
    /// 在真正发起模型请求之前裁剪历史。
    ///
    /// 步骤：
    /// 1. 估算当前 token 数，未超阈值直接返回；
    /// 2. 历史条目本身不足窗口大小，无需裁剪，直接返回；
    /// 3. 定位第一条 user 消息，作为不可裁剪的下界；
    /// 4. 计算安全裁剪起点，裁掉 `user_idx+1 .. start` 区间；
    /// 5. 打印裁剪前后的 token 估算与剩余条目数，便于观测效果。
    ///
    /// 注意：本实现只做裁剪，不产生摘要；旧信息若仍需要保留，应在注册时把
    /// [`super::summarization::Summarization`] 放在它之前执行。
    async fn call(&self, _context: &mut ExecutionContext, request: &mut LlmRequest) {
        // 1) 估算当前 token 数。
        let before = count_tokens(&self.model, request);

        // 2a) 未超过阈值：上下文还很小，什么都不做。
        if before < self.token_threshold {
            return;
        }

        // 2b) 条目数本身已少于窗口大小：即使全部保留也达不到裁剪效果，直接返回。
        if request.contents.len() < self.window_size {
            return;
        }

        // 3) 找到第一条 user 消息的位置，作为裁剪区间的下界（其之前的系统/工具定义不动）。
        let user_idx = request
            .contents
            .iter()
            .position(|item| matches!(item, ContentItem::Message { role, .. } if role == "user"));
        // 找不到 user 消息说明上下文结构异常，保持原样以免误删。
        let Some(user_idx) = user_idx else { return };

        tracing::info!(
            "SlidingWindow: {before} tokens >= threshold {}, trimming to last {} items",
            self.token_threshold,
            self.window_size
        );

        // 4) 计算裁剪起点：
        //    - 期望起点 = 保留最近 window_size 条；
        //    - find_safe_start 保证不会切断 tool_call / tool_result 配对；
        //    - .max(user_idx + 1) 保证不会裁掉 user 消息本身。
        let start = find_safe_start(&request.contents, request.contents.len() - self.window_size);
        let start = start.max(user_idx + 1);

        // 裁剪：删除 (user 消息之后) 到 (安全起点之前) 的全部条目。
        request.contents.drain(user_idx + 1..start);

        // 5) 输出裁剪后的统计信息，便于在日志中观察压缩效果。
        let after = count_tokens(&self.model, request);
        tracing::info!(
            "SlidingWindow: after trim, {after} tokens ({} items kept)",
            request.contents.len()
        );
    }
}
