//! # 内容压实优化器（Compaction）
//!
//! ## 策略
//!
//! **不丢弃任何条目**，只把"很久以前、且大概率不会再被使用"的工具结果正文
//! 替换成一句极短的占位说明，从而在不破坏对话结构的前提下省下大量 token。
//!
//! ## 为什么这样做
//!
//! Agent 循环中工具结果往往是最占地方的部分：一次 `read_file` 可能返回上千行源码，
//! 一次 `web_search` 可能返回一大段网页正文。而这些内容在几十轮之后几乎不会再被
//! 引用——模型真正需要记住的是"我已经读过那个文件 / 搜过那个关键词"这个**事实**，
//! 而不是原文。把正文替换成占位文本后，语义几乎不损失，token 却降一个量级。
//!
//! ## 处理规则
//!
//! - 只处理**窗口外**（即最近 `keep_recent` 条之外）的 `ToolResult`；
// - 只对已知的高噪音工具做替换：`read_file` 与 `web_search`；
// - 替换文本会带上关键参数（文件路径 / 搜索关键词），便于模型需要时重新调用；
// - `ToolCall` 条目本身不动（参数很小，且保留了"调过什么"的线索）。
//!
//! ## 依赖关系
//!
//! 本优化器**依赖** `ToolCall` 出现在对应的 `ToolResult` 之前，才能通过
//! `tool_call_id` 找到参数。因此它应在 [`super::sliding_window`] 之后执行，
//! 或者要求历史本身保持调用在前、结果在后的顺序。

use std::collections::HashMap;

use serde_json::Value;

use crate::agent::{
    event::{ContentItem, ToolCall},
    llm_request::LlmRequest,
};

/// 内容压实优化器。
pub struct Compaction {
    /// 保留最近多少条内容不动。位于该窗口之外的工具结果才会被压实。
    pub keep_recent: usize,
}

impl Compaction {
    /// 对请求中的历史内容执行压实。
    ///
    /// 实现方式是**原地修改**（`&mut LlmRequest`），且遍历是单趟的：
    /// - 第一趟记录每个 `tool_call_id` 对应的调用参数；
    /// - 在同一次遍历中处理 `ToolResult`，用已记录的参数生成占位文本。
    ///
    /// 由于历史中"调用在前、结果在后"，当遇到某个 `ToolResult` 时，
    /// 它的 `ToolCall` 一定已经被记录进 `call_args`，因此单趟即可完成。
    ///
    /// # 参数
    /// - `request`: 待压实的请求对象，会被就地修改。
    pub fn apply(&self, request: &mut LlmRequest) {
        // 保护边界：下标 >= protect_from 的条目属于"最近窗口"，保持原样。
        let protect_from = request.contents.len().saturating_sub(self.keep_recent);

        // tool_call_id -> 调用参数（JSON），用于在生成占位文本时回填文件路径/搜索词。
        let mut call_args: HashMap<String, Value> = HashMap::new();

        for (idx, item) in request.contents.iter_mut().enumerate() {
            match item {
                // 记录工具调用及其参数。
                ContentItem::ToolCall(ToolCall {
                    tool_call_id,
                    name: _,
                    arguments,
                }) => {
                    call_args.insert(tool_call_id.clone(), arguments.clone());
                },
                // 处理工具结果：窗口外的、已知高噪音工具的结果正文被替换。
                ContentItem::ToolResult {
                    tool_call_id,
                    name,
                    content,
                    ..
                } => {
                    // 最近窗口内的结果保留全文。
                    if idx >= protect_from {
                        continue;
                    }

                    // 取出该结果对应调用的参数（可能因历史被裁剪而缺失）。
                    let args = call_args.get(tool_call_id);

                    // 根据工具名决定占位文本；未列入白名单的工具返回 None（保持原样）。
                    let replacement = match name.as_str() {
                        // 文件读取：保留路径，暗示"已读过，可重新读取"。
                        "read_file" => {
                            let path = args
                                .and_then(|a| a.get("file_path"))
                                .and_then(Value::as_str)
                                .unwrap_or("unknown");

                            Some(format!(
                                "File '{path}' was already read. Call read_file again if you need it."
                            ))
                        },
                        // 联网搜索：保留搜索词，暗示"已搜过，可重新搜索"。
                        "web_search" => {
                            let query = args
                                .and_then(|a| a.get("query"))
                                .and_then(Value::as_str)
                                .unwrap_or("unknown");
                            Some(format!(
                                "Search results for '{query}' were already processed. Call web_search again if you need them."
                            ))
                        },
                        // 其它工具不做压实。
                        _ => None,
                    };

                    // 有占位文本则替换正文。
                    if let Some(replacement) = replacement {
                        *content = replacement;
                    }
                },
                // 普通消息不参与压实。
                _ => {},
            }
        }
    }
}
