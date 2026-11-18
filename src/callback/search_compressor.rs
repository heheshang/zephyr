use serde_json::Value;

use crate::{
    agent::{
        callback::AfterToolCallback,
        context::ExecutionContext,
        event::{ContentItem, ToolResultStatus},
    },
    knowledge_base::{chunk::fixed_length_chunking, search::vector_search},
};

/// 搜索结果压缩回调（AfterToolCall 阶段）。
///
/// **背景问题**：`web_search` 工具返回的原始内容常常非常长（几千到上万字符），
/// 这些内容会被整段塞回 LLM 上下文，既浪费 token，又淹没真正相关的信息。
///
/// **解决思路**：
/// 1. 工具执行成功后，若原始内容超过阈值（`COMPRESS_THRESHOLD`），才触发压缩；
/// 2. 用原始的搜索关键词作为 query，把长文本按固定长度切块；
/// 3. 对切块做向量检索（余弦相似度 Top-K），只保留最相关的 `TOP_K` 个块；
/// 4. 用压缩后的内容替换原工具结果返回给 Agent。
///
/// 该结构体是零大小类型（ZST），无需任何状态，所有参数都是编译期常量。
pub struct SearchCompressorCallback;

/// 只有当原始内容超过该字符数时才值得做一次向量检索压缩。
const COMPRESS_THRESHOLD: usize = 2000;

/// 切块长度（按字符计）。块太大会降低检索精度，太小则会丢失上下文语义。
const CHUNK_SIZE: usize = 500;

/// 相邻块的重叠字符数，避免语义在块边界被硬切断开。
const CHUNK_OVERLAP: usize = 50;

/// 压缩后保留的最相关块数量。
const TOP_K: usize = 3;

#[async_trait::async_trait]
impl AfterToolCallback for SearchCompressorCallback {
    /// 工具执行后的钩子：尝试压缩 `web_search` 的返回内容。
    ///
    /// # 参数
    /// - `context`: 当前执行上下文，用于回溯本次工具调用的原始查询参数。
    /// - `tool_call_id`: 本次工具调用的唯一 ID，用于在事件流中定位对应的 `ToolCall`。
    /// - `tool_name`: 被调用工具的名称。
    /// - `status`: 工具执行结果状态。
    /// - `content`: 工具返回的原始文本内容。
    ///
    /// # 返回
    /// - `None`：表示不做任何干预，沿用原始结果（由 Agent 框架继续处理）。
    /// - `Some((status, content))`：用新状态与压缩后的内容替换原工具结果。
    async fn call(
        &self,
        context: &ExecutionContext,
        tool_call_id: &str,
        tool_name: &str,
        status: ToolResultStatus,
        content: &str,
    ) -> Option<(ToolResultStatus, String)> {
        // 只处理 web_search 且执行成功的场景：
        // 其他工具不压缩，失败结果也不需要压缩（保留错误信息更利于排查）。
        if tool_name != "web_search" || status != ToolResultStatus::Success {
            return None;
        }

        // 内容不够长，压缩收益（省 token）抵不上额外一次向量检索的开销，直接放行。
        if content.len() < COMPRESS_THRESHOLD {
            return None;
        }

        // 从事件流中找回这条工具调用当初使用的 query，作为向量检索的检索词。
        // 找不到（例如事件被裁剪）则放弃压缩，保留原始内容。
        let query = extract_query(context, tool_call_id)?;
        // 按固定长度 + 重叠把长文本切成多个候选块。
        let chunks = fixed_length_chunking(content, CHUNK_SIZE, CHUNK_OVERLAP);

        // 切不出任何块（例如内容全是空白），同样放弃压缩。
        if chunks.is_empty() {
            return None;
        }

        tracing::info!(
            "🔍 Compressing web_search result: {} chars → chunking into {} pieces...",
            content.len(),
            chunks.len(),
        );

        // 对所有块做向量检索，取语义最相关的 TOP_K 块。
        match vector_search(&query, &chunks, TOP_K).await {
            Ok(hits) => {
                // 把命中的块按检索得分顺序拼接为新的压缩结果。
                let compressed = hits
                    .into_iter()
                    .map(|hit| hit.text)
                    .collect::<Vec<String>>()
                    .join("\n");
                tracing::info!(
                    "✅ Compression complete: {} chars → {} chars (top {} of {} chunks)",
                    content.len(),
                    compressed.len(),
                    TOP_K,
                    chunks.len(),
                );
                // 返回新内容，替换原工具结果交给 Agent。
                Some((ToolResultStatus::Success, compressed))
            },
            Err(e) => {
                // 检索失败（网络/模型异常）时不做降级处理，直接返回 None 保留原始内容，
                // 避免因为压缩失败反而丢失了本可用的信息。
                tracing::error!("🔍 Vector search failed: {}", e);
                None
            },
        }
    }
}

/// 从执行上下文的事件流中，根据 `tool_call_id` 找回对应的 `web_search` 查询参数。
///
/// 事件流中每个 `Event` 持有若干 `ContentItem`，其中 `ContentItem::ToolCall` 变体
/// 记录了调用工具时的 `tool_call_id`、工具名与参数（JSON 对象）。
///
/// # 返回
/// - `Some(query)`：成功取到 `arguments["query"]` 字符串。
/// - `None`：没有匹配的 `ToolCall`，或该调用没有 `query` 字段 / 该字段不是字符串。
fn extract_query(context: &ExecutionContext, tool_call_id: &str) -> Option<String> {
    context
        .events
        .iter()
        // 扁平化每个事件里的多个 content 项，得到一条 `ContentItem` 迭代流。
        .flat_map(|event| &event.content)
        // `find_map` 会在遇到第一个返回 `Some` 的元素时停止，
        // 因此这里的模式匹配同时起到「过滤 + 提取」两个作用。
        .find_map(|item| match item {
            ContentItem::ToolCall {
                tool_call_id: id,
                name,
                arguments,
            } if id == tool_call_id && name == "web_search" => arguments
                // `arguments` 是 `serde_json::Value`，用 `get` 安全取键（不存在返回 None）。
                .get("query")
                // `as_str` 仅在值是 JSON 字符串时返回 `Some`，否则为 `None`。
                .and_then(Value::as_str)
                // 克隆成独立拥有的 `String`，避免生命周期与事件借用冲突。
                .map(str::to_owned),
            // 不匹配的项（其他工具、其他调用、或非 ToolCall 变体）一律跳过。
            _ => None,
        })
}
