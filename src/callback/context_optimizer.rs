//! # 上下文优化器（Context Optimizer）
//!
//! 本模块负责在**每次调用大模型之前**，对送入模型的上下文（prompt）进行裁剪与压缩，
//! 以控制 token 消耗、避免超出上下文窗口，同时尽量不丢失关键的推理线索。
//!
//! ## 整体定位
//!
//! 运行时（`src/agent/runtime.rs`）在每次请求模型前，会把 `ExecutionContext.events`
//! 中累积的所有 `ContentItem` 平铺成一个 `LlmRequest`，然后依次执行注册的
//! `BeforeLlmCallback`。本模块提供的优化器就实现为 `BeforeLlmCallback`（或可被手动
//! 调用的普通结构体），因此它们的执行顺序直接决定了裁剪策略的叠加效果。
//!
//! ## 三种优化策略
//!
//! | 子模块 | 策略 | 触发时机 | 手段 |
//! |--------|------|----------|------|
//! | [`sliding_window`] | 滑动窗口 | token 数超过阈值时 | 直接丢弃较早的历史消息 |
//! | [`compaction`] | 内容压实 | 总是执行 | 把旧的工具结果替换成"已处理过"的占位文本 |
//! | [`summarization`] | 摘要压缩 | 历史足够长时 | 用一次 LLM 调用把旧历史浓缩成一段摘要 |
//!
//! ## 上下文数据流
//!
//! ```text
//!  ExecutionContext.events
//!          │
//!          │ flat_map(ContentItem)
//!          ▼
//!  LlmRequest { instructions: Vec<String>, contents: Vec<ContentItem> }
//!          │
//!          │ BeforeLlmCallback::call  (按注册顺序逐个执行)
//!          ├──► SlidingWindow  : token 超阈值 → drain 掉过老的历史
//!          ├──► Compaction     : 把窗口外工具结果替换为占位文本
//!          └──► Summarization : 旧历史 → LLM 摘要 → 写入 instructions
//!          │
//!          ▼
//!  runtime 构建 ChatCompletionRequest → 真正发起 HTTP 请求
//! ```
//!
//! ## 两个必须遵守的约束
//!
//! 1. **不能切断 tool_call / tool_result 的配对关系。**
//!    OpenAI 兼容协议要求：一条 `tool_result` 必须能对应到前面某条同 `tool_call_id`
//!    的 `tool_call`。若裁剪把 `tool_call` 丢掉而保留了 `tool_result`，服务端会直接报错。
//!    [`find_safe_start`] 就是用来计算"安全裁剪起点"的公共函数。
//!
//! 2. **必须保留 user 消息。**
//!    用户的原始诉求是整个任务的锚点，一旦丢失模型就失去了"我在让你干什么"的上下文，
//!    因此所有优化器都以"第一条 user 消息之后"为可裁剪区间的下界。

pub mod compaction;
pub mod sliding_window;
pub mod summarization;

use std::collections::HashSet;

use crate::{
    agent::{
        ExecutionContext,
        event::ContentItem,
        llm_request::{BeforeLlmCallback, LlmRequest},
    },
    callback::context_optimizer::{compaction::Compaction, summarization::Summarization},
};

pub struct ContextOptimizer {
    pub model: String,
    pub token_threshold: usize,
    pub enable_compaction: bool,
    pub compaction_keep_recent: usize,
    pub enable_summarization: bool,
    pub keep_recent: usize,
}

impl ContextOptimizer {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            token_threshold: 20_000,
            enable_compaction: true,
            compaction_keep_recent: 4,
            enable_summarization: true,
            keep_recent: 5,
        }
    }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for ContextOptimizer {
    async fn call(&self, context: &mut ExecutionContext, request: &mut LlmRequest) {
        let before = count_tokens(&self.model, request);
        if before < self.token_threshold {
            return;
        }
        tracing::info!(
            "ContextOptimizer: {before} tokens >= threshold {}, trimming to last {} items",
            self.token_threshold,
            self.keep_recent
        );

        if self.enable_compaction {
            Compaction {
                keep_recent: self.compaction_keep_recent,
            }
            .apply(request);

            let after = count_tokens(&self.model, request);
            tracing::info!(
                "ContextOptimizer: after compaction, {after} tokens ({} items kept)",
                request.contents.len()
            );
            if after < self.token_threshold {
                return;
            }
        }
        if self.enable_summarization {
            let summary = Summarization {
                model: self.model.clone(),
                keep_recent: self.keep_recent,
            };
            match summary.apply(context, request).await {
                Ok(_) => {
                    let after = count_tokens(&self.model, request);
                    tracing::info!(
                        "ContextOptimizer: after summarization, {after} tokens ({} items kept)",
                        request.contents.len()
                    );
                },
                Err(err) => {
                    tracing::warn!("ContextOptimizer: summarization failed: {err}");
                },
            }
        }
    }
}

/// 估算一次 LLM 请求所消耗的 token 数量。
///
/// 该函数使用 `tiktoken` 的 `cl100k_base` 编码对文本做分词计数。注意这是一次
/// **近似估算**：真实网关（Qwen / DeepSeek 等）使用的分词器与 `cl100k_base` 并不完全
/// 一致，因此结果只适合用于"是否超过阈值"这种相对判断，不适合做精确计费。
///
/// 计数规则：
/// - 每条 `instructions` 额外加 4 个 token，作为消息分隔符的固定开销估算；
/// - 每条 `contents` 同样额外加 4 个 token 的固定开销；
/// - `Message` 计入完整正文的 token 数；
/// - `ToolCall` 计入工具名 + 参数序列化后的 token 数（参数是 JSON 文本）；
/// - `ToolResult` 只计入工具名，**不计入结果正文**——因为压实阶段会把大段结果替换成
///   占位文本，实际送出去的内容远小于原始结果，计全文会高估。
///
/// # 参数
/// - `model`: 模型名。当前实现未按模型切换分词器，保留该参数是为了将来接入
///   不同厂商 tokenizer 时不必修改调用方签名。
/// - `request`: 待统计的请求对象。
///
/// # 返回值
/// 估算出的 token 总数。
pub fn count_tokens(_model: &str, request: &LlmRequest) -> usize {
    // 载入 cl100k_base 分词器；该编码是 tiktoken 内置的静态表，不会失败。
    let bpe =
        tiktoken_rs::cl100k_base().expect("tiktoken should resolve to the cl100k_base encoding");

    let mut total = 0usize;

    // 统计系统指令（system prompt 以及优化器注入的额外指令）。
    for instruction in &request.instructions {
        // 4 位经验值：每条消息的固定结构开销（角色标记、分隔符等）。
        total += 4 + bpe.encode_ordinary(instruction).len();
    }

    // 统计对话历史。
    for item in &request.contents {
        // 每条 content 同样计入 4 个 token 的固定开销。
        total += 4;
        total += match item {
            // 普通消息：计入完整正文。
            crate::agent::event::ContentItem::Message { content, .. } => {
                bpe.encode_ordinary(content).len()
            },
            // 工具调用：工具名 + 参数（JSON 文本）。
            crate::agent::event::ContentItem::ToolCall {
                name, arguments, ..
            } => {
                bpe.encode_ordinary(name).len()
                    + bpe.encode_ordinary(arguments.to_string().as_str()).len()
            },
            // 工具结果：只计工具名，结果正文在压实后会被占位文本替代，故不计。
            crate::agent::event::ContentItem::ToolResult { name, .. } => {
                bpe.encode_ordinary(name).len()
            },
        }
    }

    total
}

/// 计算一个"安全的裁剪起点"（safe start）。
///
/// ## 背景
///
/// 上下文里存在这种结构（`tool_call_id` 相同）：
///
/// ```text
/// ToolCall   { tool_call_id: "call_1", name: "read_file", ... }
/// ToolResult { tool_call_id: "call_1", name: "read_file", ... }
/// ```
///
/// 如果裁剪起点落在 `ToolCall` 与 `ToolResult` 之间，模型发出的请求就会带着一条
/// "没有对应 tool_call 的 tool_result"，OpenAI 兼容网关会返回 400 错误
/// （`messages with role 'tool' must be a response to a preceding tool call`）。
///
/// ## 算法
///
/// 从传入的 `start` 出发反复向前回退：
/// 1. 收集 `contents[start..]` 中出现过的所有 `tool_call_id`；
/// 2. 若 `contents[start..]` 中存在某个 `tool_result`，其 `tool_call_id` 不在上述集合中，
///    说明它的配对 `tool_call` 被裁掉了，必须回退；
/// 3. 在 `contents[..start]` 中找到该 `tool_call` 的位置，把 `start` 回退到它的位置；
/// 4. 重复 1~3，直到没有"孤立的 tool_result"为止。
///
/// ## 终止性
///
/// 每次迭代 `start` 严格减小（回退到 `call_idx < start`），因此必然收敛。
/// 若在 `contents[..start]` 中找不到对应的 `tool_call`（说明该 result 本就是
/// 历史遗留的孤立项），则直接停止并返回当前 `start`。
///
/// # 参数
/// - `contents`: 完整的历史条目列表。
/// - `start`: 期望的裁剪起点（通常为 `contents.len() - keep_recent`）。
///
/// # 返回值
/// 修正后的、保证不会出现孤立 `tool_result` 的裁剪起点。
fn find_safe_start(contents: &[ContentItem], mut start: usize) -> usize {
    loop {
        // 步骤 1：收集 start 之后出现过的所有 tool_call_id。
        let call_ids: HashSet<&str> = contents[start..]
            .iter()
            .filter_map(|item| match item {
                ContentItem::ToolCall { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect();

        // 步骤 2：找出 start 之后"找不到配对 tool_call"的孤立 tool_result。
        let missing_call = contents[start..].iter().find_map(|item| match item {
            ContentItem::ToolResult { tool_call_id, .. }
                if !call_ids.contains(&tool_call_id.as_str()) =>
            {
                Some(tool_call_id)
            },
            _ => None,
        });

        // 没有孤立 result，说明当前 start 就是安全起点。
        let Some(missing_call) = missing_call else {
            break;
        };

        // 步骤 3：在被裁掉的前半段中定位该 tool_call 的位置。
        let Some(call_idx) = contents[..start].iter().position(|item| {
            matches!(item,
            ContentItem::ToolCall { tool_call_id, .. } if tool_call_id == missing_call )
        }) else {
            // 找不到配对（本身就是孤立项），无法继续回退，直接结束。
            break;
        };

        // 步骤 4：回退到 tool_call 所在位置，然后重新检查。
        start = call_idx;
    }

    start
}
