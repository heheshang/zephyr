use std::sync::Arc;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use crate::{
    agent::context::ExecutionContext,
    knowledge::knowledge_base::KnowledgeBaseManager,
    tool::Tool,
};

/// 默认返回的片段数。
const DEFAULT_TOP_K: usize = 5;
/// 允许 LLM 指定的 `top_k` 上限，防止一次拉回过多内容撑爆上下文。
const MAX_TOP_K: usize = 20;

/// 检索文档知识库，把相关片段返回给 LLM。
///
/// 与 `recall_memory` 的区别：
/// - `recall_memory` 检索**解题经验**（LLM 抽取的结构化字段）
/// - `search_knowledge` 检索**原始文档**（用户导入的资料）
///
/// 本工具是**被动检索**：LLM 看到工具描述后自己决定何时调用，
/// 每次调用按 `query` 做一次向量检索。
pub struct SearchKnowledgeTool {
    manager: Arc<KnowledgeBaseManager>,
}

impl SearchKnowledgeTool {
    pub fn new(manager: Arc<KnowledgeBaseManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize, Debug, JsonSchema)]
pub struct SearchKnowledgeArgs {
    /// 要检索的问题或关键词，用自然语言描述。
    pub query: String,

    /// 返回的片段数量，默认 5，最大 20。
    #[schemars(range(min = 1, max = 20))]
    #[serde(default)]
    pub top_k: Option<usize>,
}

#[async_trait::async_trait]
impl Tool for SearchKnowledgeTool {
    fn name(&self) -> &str {
        "search_knowledge"
    }

    fn description(&self) -> &str {
        "在导入的文档知识库中做混合检索(向量语义 + 关键词 BM25,RRF 融合排序),\
         返回与问题最相关的原文片段(带标题与来源)。\
         回答涉及资料具体内容的问题前先用它查找依据。"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(SearchKnowledgeArgs))
            .expect("Failed to serialize SearchKnowledgeArgs schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: SearchKnowledgeArgs = serde_json::from_str(args_json).map_err(|e| {
            anyhow::anyhow!(
                "invalid arguments, expected {{\"query\": string, \"top_k\"?: number}}: {e}"
            )
        })?;

        let query = args.query.trim();
        if query.is_empty() {
            return Err(anyhow::anyhow!("argument `query` must not be empty"));
        }

        // LLM 可能给出离谱的 top_k，这里夹到允许区间内。
        let top_k = args
            .top_k
            .unwrap_or(DEFAULT_TOP_K)
            .clamp(1, MAX_TOP_K);

        let result = self.manager.search(query, top_k).await?;
        if result.hits.is_empty() {
            return Ok("知识库中没有找到相关的内容。".to_string());
        }

        Ok(result.format())
    }
}

/// 把文档写入知识库。
///
/// 没有入库通道的话知识库永远是空的，检索工具无从发挥，
/// 因此配套提供这个工具，让 Agent（而非仅限开发者脚本）也能维护知识库。
pub struct AddKnowledgeTool {
    manager: Arc<KnowledgeBaseManager>,
}

impl AddKnowledgeTool {
    pub fn new(manager: Arc<KnowledgeBaseManager>) -> Self {
        Self { manager }
    }
}

#[derive(Deserialize, Debug, JsonSchema)]
pub struct AddKnowledgeArgs {
    /// 文档标题。标题会参与向量计算，帮助区分语义相近的不同文档。
    pub title: String,

    /// 文档来源标识：文件路径、URL 或自由描述。
    #[serde(default)]
    pub source: Option<String>,

    /// 文档正文。
    pub content: String,
}

#[async_trait::async_trait]
impl Tool for AddKnowledgeTool {
    fn name(&self) -> &str {
        "add_knowledge"
    }

    fn description(&self) -> &str {
        "把一份文档写入知识库,之后可以用 search_knowledge 检索到它。\
         文档会被自动切片并向量化。"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(AddKnowledgeArgs))
            .expect("Failed to serialize AddKnowledgeArgs schema")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: AddKnowledgeArgs = serde_json::from_str(args_json).map_err(|e| {
            anyhow::anyhow!(
                "invalid arguments, expected {{\"title\": string, \"content\": string}}: {e}"
            )
        })?;

        let title = args.title.trim();
        if title.is_empty() {
            return Err(anyhow::anyhow!("argument `title` must not be empty"));
        }

        let content = args.content.trim();
        if content.is_empty() {
            return Err(anyhow::anyhow!("argument `content` must not be empty"));
        }

        let chunks = self
            .manager
            .add_document(
                title,
                args.source.unwrap_or_default(),
                content.to_owned(),
            )
            .await?;

        Ok(format!("已把《{title}》写入知识库,共 {chunks} 个片段。"))
    }
}

#[cfg(test)]
mod knowledge_tool_tests {
    use super::*;
    use crate::knowledge::embedder::FakeEmbedder;

    fn test_manager() -> Arc<KnowledgeBaseManager> {
        Arc::new(
            KnowledgeBaseManager::with_embedder(":memory:", Box::new(FakeEmbedder::new(32)))
                .unwrap()
                .with_chunking(50, 10)
                .with_min_similarity(0.0),
        )
    }

    #[tokio::test]
    async fn search_returns_formatted_hits() -> anyhow::Result<()> {
        let manager = test_manager();
        manager
            .add_document("圆的面积", "test://math", "圆的面积等于 π 乘以半径的平方。")
            .await?;

        let tool = SearchKnowledgeTool::new(manager);
        let output = tool
            .execute(
                r#"{"query": "圆的面积"}"#,
                &ExecutionContext::default(),
            )
            .await?;

        assert!(output.contains("圆的面积"));
        assert!(output.contains("片段"));
        Ok(())
    }

    #[tokio::test]
    async fn search_with_no_match_reports_empty() -> anyhow::Result<()> {
        let manager = test_manager();
        let tool = SearchKnowledgeTool::new(manager);

        let output = tool
            .execute(r#"{"query": "不存在的内容"}"#, &ExecutionContext::default())
            .await?;

        assert!(output.contains("没有找到"));
        Ok(())
    }

    #[tokio::test]
    async fn search_rejects_empty_query() {
        let tool = SearchKnowledgeTool::new(test_manager());

        let err = tool
            .execute(r#"{"query": "  "}"#, &ExecutionContext::default())
            .await
            .unwrap_err();

        assert!(err.to_string().contains("must not be empty"));
    }

    #[tokio::test]
    async fn search_rejects_malformed_arguments() {
        let tool = SearchKnowledgeTool::new(test_manager());

        let err = tool
            .execute("{}", &ExecutionContext::default())
            .await
            .unwrap_err();

        assert!(err.to_string().contains("invalid arguments"));
    }

    #[tokio::test]
    async fn top_k_is_clamped_to_the_allowed_range() -> anyhow::Result<()> {
        let manager = test_manager();
        manager
            .add_document("A", "test://a", "一些内容用于测试 top_k 夹取逻辑。")
            .await?;

        let tool = SearchKnowledgeTool::new(manager);

        // top_k=0 会被夹到 1，不应 panic
        let output = tool
            .execute(
                r#"{"query": "内容", "top_k": 0}"#,
                &ExecutionContext::default(),
            )
            .await?;
        assert!(!output.is_empty());

        // 超过上限也会被夹住
        let output = tool
            .execute(
                r#"{"query": "内容", "top_k": 9999}"#,
                &ExecutionContext::default(),
            )
            .await?;
        assert!(!output.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn add_tool_writes_a_document() -> anyhow::Result<()> {
        let manager = test_manager();
        let tool = AddKnowledgeTool::new(manager.clone());

        let output = tool
            .execute(
                r#"{"title": "TCP", "source": "test://net", "content": "TCP 是面向连接的可靠传输协议。"}"#,
                &ExecutionContext::default(),
            )
            .await?;

        assert!(output.contains("TCP"));
        assert!(output.contains("片段"));
        assert_eq!(manager.total_chunks()?, 1);

        Ok(())
    }

    #[tokio::test]
    async fn add_tool_rejects_empty_title_and_content() {
        let tool = AddKnowledgeTool::new(test_manager());

        let err = tool
            .execute(
                r#"{"title": "  ", "content": "x"}"#,
                &ExecutionContext::default(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("title"));

        let err = tool
            .execute(
                r#"{"title": "x", "content": "  "}"#,
                &ExecutionContext::default(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("content"));
    }

    #[test]
    fn parameters_schema_exposes_query_and_optional_top_k() {
        let tool = SearchKnowledgeTool::new(test_manager());
        let schema = tool.parameters();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["properties"]["top_k"].is_object());
        // top_k 是可选的，不应出现在 required 里
        let required = schema["required"].as_array().expect("required should be an array");
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "query");
    }
}
