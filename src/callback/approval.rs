use std::{collections::HashSet, io::Write};

use crate::agent::{
    callback::{BeforeToolCallback, ToolCallView},
    context::ExecutionContext,
};

/// 高危工具的人工审批回调。
///
/// 职责：在真正执行某个「危险工具」之前，把工具名与参数打印到终端，
/// 并同步等待用户在 stdin 上输入 `y`/`n` 决定是否放行。
///
/// 约定（与 `BeforeToolCallback` 语义一致）：
/// - 返回 `None`：不干预，继续执行该工具（用户已批准）。
/// - 返回 `Some(msg)`：短路，跳过该工具的执行，并把 `msg` 作为工具结果
///   回灌给模型（用户已拒绝）。
///
/// 注意：本实现是**交互式**的，会阻塞等待终端输入，因此只适用于
/// 命令行本地运行；服务端/批处理场景请换成非交互的实现（例如
/// 通过 HTTP 审批接口或预配置白名单）。
pub struct ApprovalCallback {
    /// 需要人工审批的工具名集合（精确匹配，大小写敏感）。
    dangerous_tools: HashSet<String>,
}

impl ApprovalCallback {
    /// 构造审批回调。
    ///
    /// `dangerous_tools` 可以是任意可迭代的、元素可转成 `String` 的集合，
    /// 例如 `&["delete", "shell"]`、`vec!["delete".to_string()]`。
    pub fn new(dangerous_tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            dangerous_tools: dangerous_tools.into_iter().map(|t| t.into()).collect(),
        }
    }
}

#[async_trait::async_trait]
impl BeforeToolCallback for ApprovalCallback {
    /// 工具执行前的拦截点。
    ///
    /// - `_context`：本次任务的执行上下文（步数、消息历史等），此处未使用，
    ///   故以下划线开头抑制未使用告警。
    /// - `tool_call`：本次即将执行的工具调用（名称 + 参数的只读视图）。
    async fn call(
        &self,
        _context: &ExecutionContext,
        tool_call: ToolCallView<'_>,
    ) -> Option<String> {
        // 非高危工具：直接放行，不打断模型的多轮工具循环。
        if !self.dangerous_tools.contains(tool_call.name) {
            return None;
        }

        // 命中高危工具：打印工具名与参数，让用户知道将要发生什么。
        println!("\n⚠️  即将执行高危操作");
        println!("工具: {}", tool_call.name);
        println!("参数: {}", tool_call.arguments);

        // stdin 读取是阻塞操作，不能直接放在 async 上下文里，
        // 否则会卡住整个 tokio 运行时线程；因此放到阻塞线程池中执行。
        let approved = tokio::task::spawn_blocking(|| {
            print!("是否执行？(y/n): ");
            // flush 保证提示语在等待输入前就显示出来（stdout 默认行缓冲）。
            std::io::stdout().flush().ok();
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            // 只认 "y"/"Y"，其余（n、空行、EOF、读失败）一律视为拒绝。
            input.trim().eq_ignore_ascii_case("y")
        })
        .await
        // 线程池不可用或 join 失败时按「拒绝」处理（安全优先）。
        .unwrap_or(false);

        if approved {
            // 已批准：返回 None 表示不干预，工具照常执行。
            println!("✅ 已批准，继续执行...\n");
            None
        } else {
            // 已拒绝：返回 Some 短路工具执行，并把拒绝原因回灌给模型，
            // 让模型知道该操作未执行并据此调整后续计划。
            println!("❌ 已拒绝，跳过执行\n");
            Some(format!("User denied execution of {}", tool_call.name))
        }
    }
}
