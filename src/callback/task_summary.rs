//! 任务结束时的执行摘要回调。
//!
//! 这个回调在整个任务结束（无论成功还是失败）后被触发一次，把本次执行的
//! 关键信息汇总成一段人类可读的文本：
//!
//! - 执行 ID、总步数、事件条数；
//! - token 用量（prompt / completion / total）；
//! - 最终结果摘要（截断到固定长度）；
//! - 失败时的错误原因。
//!
//! 典型用途：打印到控制台、写进日志、落库做审计、或作为兜底答案返回。

use crate::agent::{
    callback::{OnTaskEndCallback, TaskOutcome},
    context::ExecutionContext,
};

/// 摘要中最终结果的最大字符数，避免超长回答把摘要撑爆。
const RESULT_PREVIEW_LIMIT: usize = 500;

/// 任务结束摘要回调。
///
/// 该结构体是零大小类型（ZST），无需任何状态，所有参数都是编译期常量。
pub struct TaskSummaryCallback;

#[async_trait::async_trait]
impl OnTaskEndCallback for TaskSummaryCallback {
    /// 任务结束后生成一段执行摘要。
    ///
    /// # 返回
    /// - `Some(summary)`：返回生成的摘要文本。若任务失败，则**不**返回摘要
    ///   （避免用摘要覆盖掉真实的错误信息）。
    /// - `None`：不覆盖原始输出。
    async fn call(&self, context: &ExecutionContext, outcome: &TaskOutcome) -> Option<String> {
        // 任务失败时保留原始错误，不覆盖输出。
        if let TaskOutcome::Failure { error } = outcome {
            tracing::error!(
                "❌ 任务失败 | execution_id={} | steps={} | error={}",
                context.execution_id,
                context.current_step,
                error,
            );
            return None;
        }

        // 取最终结果的预览。
        let preview = context
            .final_result
            .as_deref()
            .map(|result| {
                if result.chars().count() > RESULT_PREVIEW_LIMIT {
                    format!(
                        "{}…",
                        result
                            .chars()
                            .take(RESULT_PREVIEW_LIMIT)
                            .collect::<String>()
                    )
                } else {
                    result.to_string()
                }
            })
            .unwrap_or_else(|| "<无最终结果>".to_string());

        let summary = format!(
            "✅ 任务完成 | execution_id={} | steps={} | events={} | tokens(prompt/completion/total)={}/{}/{} | 结果={}",
            context.execution_id,
            context.current_step,
            context.events.len(),
            context.usage.prompt_tokens,
            context.usage.completion_tokens,
            context.usage.total_tokens,
            preview,
        );

        tracing::info!("{}", summary);
        Some(summary)
    }
}
