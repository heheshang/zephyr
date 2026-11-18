//! 针对 Agent 新增回调扩展点的离线测试。
//!
//! 这些测试不依赖网络：全部通过「模型调用前回调短路」来跳过真实的 HTTP 请求，
//! 从而验证回调的触发顺序、参数传递与短路/覆盖语义。

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use zephyr::{
    agent::{
        callback::{
            AfterLlmCallback, BeforeLlmCallback, LlmRequestView, LlmResponseView,
            OnTaskEndCallback, TaskOutcome,
        },
        context::ExecutionContext,
        runtime::Agent,
    },
    tools::ToolBox,
};

/// 一个会在「模型调用前」返回固定内容，从而短路模型调用的回调。
struct ShortCircuitCallback {
    content: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl BeforeLlmCallback for ShortCircuitCallback {
    async fn call(
        &self,
        _context: &ExecutionContext,
        _request: LlmRequestView<'_>,
    ) -> Option<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Some(self.content.clone())
    }
}

/// 一个记录「模型调用后」是否被触发的回调（正常路径下不会触发，因为被短路了）。
struct AfterLlmSpy {
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl AfterLlmCallback for AfterLlmSpy {
    async fn call(
        &self,
        _context: &ExecutionContext,
        _response: LlmResponseView<'_>,
    ) -> Option<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        None
    }
}

/// 一个记录任务结束结果的回调。
struct TaskEndSpy {
    calls: Arc<AtomicUsize>,
    outcomes: Arc<std::sync::Mutex<Vec<TaskOutcome>>>,
}

#[async_trait::async_trait]
impl OnTaskEndCallback for TaskEndSpy {
    async fn call(&self, _context: &ExecutionContext, outcome: &TaskOutcome) -> Option<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.outcomes.lock().unwrap().push(outcome.clone());
        None
    }
}

/// 一个会覆盖最终输出的任务结束回调。
struct TaskEndOverride;

#[async_trait::async_trait]
impl OnTaskEndCallback for TaskEndOverride {
    async fn call(&self, _context: &ExecutionContext, _outcome: &TaskOutcome) -> Option<String> {
        Some("被任务结束回调覆盖的输出".to_string())
    }
}

/// 构造一个不带任何工具的 Agent（空工具箱即可满足类型要求）。
fn build_agent() -> Agent {
    Agent::new(
        "test-model",
        Some("you are a test agent"),
        Arc::new(ToolBox::new()),
    )
}

/// 「模型调用前」回调短路时，模型不会被真正调用，且短路内容成为最终答案。
#[tokio::test]
async fn before_llm_callback_short_circuits_the_model_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let after_calls = Arc::new(AtomicUsize::new(0));

    let agent = build_agent()
        .with_before_llm_callback(Arc::new(ShortCircuitCallback {
            content: "短路答案".to_string(),
            calls: calls.clone(),
        }))
        .with_after_llm_callback(Arc::new(AfterLlmSpy {
            calls: after_calls.clone(),
        }));

    let result = agent.run("随便问点什么").await.unwrap();

    // 短路内容成为最终答案。
    assert_eq!(result.output, "短路答案");
    // 前置回调被调用了 1 次。
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // 因为短路跳过了模型请求，后置回调不应被触发。
    assert_eq!(after_calls.load(Ordering::SeqCst), 0);
    // 最终结果也同步写入了上下文。
    assert_eq!(result.context.final_result.as_deref(), Some("短路答案"));
}

/// 任务正常结束时，任务结束回调被触发一次，且 outcome 为 Success。
#[tokio::test]
async fn on_task_end_callback_fires_on_success() {
    let calls = Arc::new(AtomicUsize::new(0));
    let outcomes = Arc::new(std::sync::Mutex::new(Vec::new()));

    let agent = build_agent()
        .with_before_llm_callback(Arc::new(ShortCircuitCallback {
            content: "答案".to_string(),
            calls: Arc::new(AtomicUsize::new(0)),
        }))
        .with_on_task_end_callback(Arc::new(TaskEndSpy {
            calls: calls.clone(),
            outcomes: outcomes.clone(),
        }));

    let _ = agent.run("问题").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let outcomes = outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0], TaskOutcome::Success);
}

/// 任务失败时（步数超限），任务结束回调同样被触发一次，且 outcome 为 Failure。
#[tokio::test]
async fn on_task_end_callback_fires_on_failure() {
    let calls = Arc::new(AtomicUsize::new(0));
    let outcomes = Arc::new(std::sync::Mutex::new(Vec::new()));

    let agent = build_agent()
        // max_steps = 0 意味着第一次进入循环就触发步数守卫，主循环直接失败。
        .with_max_steps(0)
        .with_on_task_end_callback(Arc::new(TaskEndSpy {
            calls: calls.clone(),
            outcomes: outcomes.clone(),
        }));

    let err = agent.run("问题").await.unwrap_err();
    assert!(err.to_string().contains("maximum of 0 steps"));

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let outcomes = outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    match &outcomes[0] {
        TaskOutcome::Failure { error } => assert!(error.contains("maximum of 0 steps")),
        TaskOutcome::Success => panic!("失败的任务不应返回 Success outcome"),
    }
}

/// 任务结束回调返回内容时，会覆盖最终输出。
#[tokio::test]
async fn on_task_end_callback_can_override_the_output() {
    let agent = build_agent()
        .with_before_llm_callback(Arc::new(ShortCircuitCallback {
            content: "原始答案".to_string(),
            calls: Arc::new(AtomicUsize::new(0)),
        }))
        .with_on_task_end_callback(Arc::new(TaskEndOverride));

    let result = agent.run("问题").await.unwrap();
    assert_eq!(result.output, "被任务结束回调覆盖的输出");
}
