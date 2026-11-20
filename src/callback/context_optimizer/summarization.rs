//! # 摘要压缩优化器（Summarization）
//!
//! ## 策略
//!
//! 当历史已经积累到一定长度时，把"较早的那一段历史"交给一次额外的 LLM 调用，
//! 浓缩成一段结构化摘要，再把这段摘要作为**额外指令**注入 `instructions`，
//! 最后把被摘要的那段原文从 `contents` 中移除。
//!
//! 与滑动窗口相比，摘要压缩**不丢语义**：模型仍然能看到"之前发现了什么、
//! 调过哪些工具、还剩什么没做"，只是以压缩后的形式呈现。
//!
//! ## 关键设计
//!
//! ### 1. 增量摘要
//!
//! 摘要结果是**累积**的：每次只对"上次摘要之后新增的历史"做摘要，然后与
//! 上一次的摘要拼接。这样避免了每轮都重新摘要全部历史（成本高、也容易越摘要越糊）。
//! 上一次的摘要保存在 `ExecutionContext.state` 的 `"context_summary"` 键中。
//!
//! ### 2. 状态记忆
//!
//! - `context.state["context_summary"]`：累积的摘要全文；
//! - `context.state["last_summary_idx"]`：上一次摘要所覆盖到的下标边界。
//!
//! 由于 `ExecutionContext` 在整个 Agent 运行期间一直存活，这两个键可以跨轮次复用。
//!
//! ### 3. 安全边界
//!
//! 摘要区间同样受 [`find_safe_start`] 约束，不能切断 `tool_call` / `tool_result`
//! 配对；并且第一条 user 消息永远不会被摘掉。

use anyhow::Ok;
use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use serde_json::Value;

use crate::{
    agent::{ExecutionContext, event::ContentItem, llm_request::LlmRequest},
    callback::context_optimizer::find_safe_start,
};

/// 摘要用的 system prompt 模板。
///
/// `{history}` 会被替换为待摘要的历史文本。提示词要求模型输出三部分：
/// 已有关键发现、已调用的工具、待完成事项，并强调要简洁。
const SUMMARIZATION_PROMPT: &str = "You are summarizing an AI agent's work-in-progress \
history. Given the execution history below, write a short, structured summary covering: \
1) key findings so far, 2) tools that were called, 3) what remains to be done. \
Be concise — a few sentences is enough.\n\nExecution history:\n{history}";

/// 摘要压缩优化器。
pub struct Summarization {
    /// 执行摘要所使用的模型名（通常用便宜的小模型即可）。
    pub model: String,
    /// 保留最近多少条内容不参与摘要。
    pub keep_recent: usize,
}

impl Summarization {
    /// 对历史执行摘要压缩。
    ///
    /// 完整流程：
    /// 1. 条目数不足 `keep_recent` 时直接返回；
    /// 2. 定位第一条 user 消息（永不摘要）；
    /// 3. 读取上次的摘要状态，确定本次摘要的起始下标；
    /// 4. 用 [`find_safe_start`] 计算摘要区间的结束下标（保护 tool 配对）；
    /// 5. 把待摘要区间格式化为纯文本，调用 LLM 生成摘要；
    /// 6. 把累积摘要作为额外指令注入 `instructions`；
    /// 7. 从 `contents` 中移除被摘要的原文，并更新 `state` 中的摘要与边界标记。
    ///
    /// # 参数
    /// - `context`: 执行上下文，用于读取/写入跨轮次的摘要状态。
    /// - `request`: 待优化的请求对象，会被就地修改。
    pub async fn apply(
        &self,
        context: &mut ExecutionContext,
        request: &mut LlmRequest,
    ) -> anyhow::Result<()> {
        // 1) 历史还很短，无需摘要。
        if request.contents.len() <= self.keep_recent {
            return Ok(());
        }

        // 2) 定位第一条 user 消息。它是任务的锚点，永不进入摘要区间。
        let user_idx = request
            .contents
            .iter()
            .position(|item| matches!(item,  ContentItem::Message { role, .. } if role == "user"));
        // 找不到 user 消息说明上下文结构异常，保持原样。
        let Some(user_idx) = user_idx else {
            return Ok(());
        };

        // 3) 读取上次的摘要边界；没有记录时从 user 消息之后开始（即摘要全部中间历史）。
        let last_summary_idx = context
            .session
            .state
            .get("last_summary_idx")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(user_idx);

        // 本次摘要区间：[summary_start, summary_end)
        let summary_start = last_summary_idx + 1;

        // 4) 结束下标：期望保留最近 keep_recent 条，再由 find_safe_start 修正，
        //    确保不会把 tool_result 留在它的 tool_call 之前被裁掉。
        let summary_end =
            find_safe_start(&request.contents, request.contents.len() - self.keep_recent);
        // 区间为空（说明中间可摘要的内容已被上一轮清空），无需处理。
        if summary_end <= summary_start {
            return Ok(());
        }

        // 5) 格式化待摘要历史，并调用 LLM 生成新的摘要片段。
        let to_summarize = &request.contents[summary_start..summary_end];
        let history_text = format_history(to_summarize);
        let new_summary_chunk = self.generate_summary(&history_text).await?;

        // 6) 与上一轮摘要拼接，形成累积摘要。
        let existing_summary = context
            .session
            .state
            .get("context_summary")
            .and_then(Value::as_str)
            .map(str::to_string);
        let full_summary = match existing_summary {
            Some(prev) => format!("{prev}\n\n{new_summary_chunk}"),
            None => new_summary_chunk,
        };

        // 把摘要作为额外 system 指令注入，让模型在看到原始历史之前先读到"之前的进展"。
        request.append_instructions(format!("[Summary of earlier progress]\n{full_summary}"));

        // 7) 移除被摘要的原文区间，并更新状态标记。
        request.contents.drain((user_idx + 1)..summary_end);
        context.session.state.insert(
            "last_summary_idx".to_string(),
            Value::from((summary_end - 1) as u64),
        );
        context.session.state.insert(
            "last_summary_idx".to_string(),
            Value::from((summary_end - 1) as u64),
        );
        context
            .session
            .state
            .insert("context_summary".to_string(), Value::String(full_summary));

        Ok(())
    }

    /// 调用一次 LLM 把历史文本压缩成一段摘要。
    ///
    /// 使用 system prompt 承载历史与指令、user 消息仅为触发语，
    /// 并限制 `max_tokens = 512`，避免摘要本身过长反而挤占上下文。
    ///
    /// # 参数
    /// - `history`: 已格式化的历史文本。
    ///
    /// # 返回值
    /// 生成的摘要文本；若响应中没有内容则返回空字符串。
    async fn generate_summary(&self, history: &str) -> anyhow::Result<String> {
        // 使用环境变量中的 OPENAI_API_KEY / OPENAI_BASE_URL 构建客户端。
        let client = async_openai::Client::new();
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![
                // system：注入提示词模板与待摘要的历史。
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(SUMMARIZATION_PROMPT.replace("{history}", history))
                    .build()?
                    .into(),
                // user：一句触发语，确保模型开始生成。
                ChatCompletionRequestUserMessageArgs::default()
                    .content("Please summarize.")
                    .build()?
                    .into(),
            ])
            .max_tokens(512u32)
            .build()?;

        let response = client.chat().create(request).await?;

        // 取第一个 choice 的正文；没有正文时返回空串。
        Ok(response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default())
    }
}

/// 把 `ContentItem` 序列格式化为供摘要模型阅读的纯文本。
///
/// 为了控制送入摘要模型的 token 量，各类型内容都会做截断：
/// - 普通消息正文最多 500 字符；
/// - 工具结果正文最多 200 字符；
/// - 工具调用保留完整的名称与参数（参数本身较小，且是重要线索）。
///
/// # 参数
/// - `items`: 待格式化的历史条目。
///
/// # 返回值
/// 以换行连接的可读文本。
fn format_history(items: &[ContentItem]) -> String {
    items
        .iter()
        .map(|item| match item {
            // 普通消息：截断到 500 字符。
            ContentItem::Message { role, content } => {
                let preview: String = content.chars().take(500).collect();
                format!("[{role}]: {preview}")
            },
            // 工具调用：保留完整名称与参数。
            ContentItem::ToolCall {
                name, arguments, ..
            } => {
                format!("[tool call]: {name}({arguments})")
            },
            // 工具结果：截断到 200 字符。
            ContentItem::ToolResult { name, content, .. } => {
                let preview: String = content.chars().take(200).collect();
                format!("[tool result]: {name} -> {preview}")
            },
        })
        .collect::<Vec<_>>()
        .join("\n")
}
