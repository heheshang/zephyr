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
