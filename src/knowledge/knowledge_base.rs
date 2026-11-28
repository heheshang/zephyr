use uuid::Uuid;

use crate::{
    knowledge::{
        embedder::Embedder,
        model::{Chunk, Document, SearchHit, SearchResult, chunk_embedding_text},
        store::KnowledgeStore,
    },
    search::chunking::chunking_with_offsets,
};

/// 默认切片长度（字符数）。
pub const DEFAULT_CHUNK_SIZE: usize = 500;
/// 默认切片重叠长度（字符数）。
pub const DEFAULT_CHUNK_OVERLAP: usize = 50;
/// 默认检索返回的切片数。
pub const DEFAULT_TOP_K: usize = 5;
/// 相似度低于此值的切片不返回，避免无关内容污染上下文。
pub const MIN_SIMILARITY: f32 = 0.3;

/// 文档知识库管理器：负责"文档入库"与"语义检索"两条链路。
///
/// # 入库
///
/// ```text
/// 文档全文
///   → chunking_with_offsets（切片，带原文偏移）
///   → Embedder::embed（分批调用 embedding 接口）
///   → SQLite 写入 documents + chunks
/// ```
///
/// # 检索
///
/// ```text
/// 查询文本
///   → Embedder::embed（查询向量）
///   → 全表扫描算余弦相似度并排序
///   → 取 top_k，过滤掉低相似度的
/// ```
///
/// # 与 `TaskMemoryManager` 的区别
///
/// - `TaskMemoryManager`：存"解题经验"，内容由 LLM 抽取成结构化字段，
///   检索目标是**找到相似问题**。
/// - `KnowledgeBaseManager`：存"原始文档"，内容原样保留，
///   检索目标是**找到相关段落**。
///
/// 两者都基于 SQLite + embedding，但数据模型与语义完全不同，因此独立成模块。
pub struct KnowledgeBaseManager {
    store: KnowledgeStore,
    embedder: Box<dyn Embedder>,
    chunk_size: usize,
    chunk_overlap: usize,
    top_k: usize,
    min_similarity: f32,
}

impl KnowledgeBaseManager {
    /// 打开知识库，使用 OpenAI 兼容接口计算向量。
    pub fn new(db_path: &str, embedding_model: impl Into<String>) -> anyhow::Result<Self> {
        Self::with_embedder(
            db_path,
            Box::new(crate::knowledge::embedder::ApiEmbedder::new(embedding_model)),
        )
    }

    /// 用自定义的向量化器打开知识库（测试或替换后端时使用）。
    ///
    /// `embedding_dim` 必须与 `embedder` 实际输出的维度一致——
    /// vec0 虚表的维度在建表时就固定了，写入维度不符的向量会被拒绝。
    pub fn with_embedder(
        db_path: &str,
        embedder: Box<dyn Embedder>,
    ) -> anyhow::Result<Self> {
        let embedding_dim = embedder.embedding_dim();
        Ok(Self {
            store: KnowledgeStore::open(db_path, embedding_dim)?,
            embedder,
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            top_k: DEFAULT_TOP_K,
            min_similarity: MIN_SIMILARITY,
        })
    }

    /// 设置切片长度与重叠。
    pub fn with_chunking(mut self, chunk_size: usize, chunk_overlap: usize) -> Self {
        self.chunk_size = chunk_size;
        self.chunk_overlap = chunk_overlap;
        self
    }

    /// 设置默认返回的切片数。
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// 设置最低相似度阈值。
    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = min_similarity;
        self
    }

    /// 把一篇文档写入知识库，返回写入的切片数。
    ///
    /// 切片向量由 `Embedder` 计算，因此本方法需要网络；
    /// embedding 失败时整体失败，**不写入半份数据**
    /// （所有切片向量先算好，再一次性落库）。
    pub async fn add_document(
        &self,
        title: impl Into<String>,
        source: impl Into<String>,
        content: impl Into<String>,
    ) -> anyhow::Result<usize> {
        let document = Document {
            id: Uuid::new_v4().to_string(),
            title: title.into(),
            source: source.into(),
            content: content.into(),
            created_at: chrono::Utc::now().timestamp(),
        };

        self.add_document_with_id(document).await
    }

    /// 写入一篇指定 id 的文档。
    ///
    /// **幂等**：同 id 重复入库会先删除该文档的旧切片，再写入新切片，
    /// 不会残留过期内容。触发器保证向量索引与 FTS 索引同步清理。
    pub async fn add_document_with_id(&self, document: Document) -> anyhow::Result<usize> {
        let spans = chunking_with_offsets(&document.content, self.chunk_size, self.chunk_overlap);

        if spans.is_empty() {
            // 空文档没有切片，但仍记录元信息，方便后续发现"文档没内容"。
            // 同样要清掉旧切片，否则旧内容会继续被检索到。
            self.store.replace_chunks(&document.id, &[])?;
            self.store.add_document(&document)?;
            return Ok(0);
        }

        // 标题参与 embedding：同一段文字在不同文档里语义可能不同。
        let texts: Vec<String> = spans
            .iter()
            .map(|span| chunk_embedding_text(&document.title, &span.text))
            .collect();

        let embeddings = self.embedder.embed(&texts).await?;

        if embeddings.len() != spans.len() {
            anyhow::bail!(
                "embedder returned {} vectors for {} chunks",
                embeddings.len(),
                spans.len()
            );
        }

        let chunks: Vec<Chunk> = spans
            .into_iter()
            .zip(embeddings)
            .map(|(span, embedding)| Chunk {
                id: Uuid::new_v4().to_string(),
                document_id: document.id.clone(),
                start_offset: span.start,
                text: span.text,
                embedding,
            })
            .collect();

        self.store.add_document(&document)?;
        // 幂等写入：先删旧切片再写新的
        self.store.replace_chunks(&document.id, &chunks)?;

        Ok(chunks.len())
    }

    /// 混合检索：向量召回 + 关键词召回，RRF 融合排序。
    ///
    /// 这是默认的检索路径：
    /// - 向量路用 sqlite-vec 的近似最近邻索引（O(log N)）
    /// - 关键词路用 FTS5 的 BM25 倒排索引
    /// - 两路结果按 RRF 融合，被双方同时命中的片段排最前
    ///
    /// # 为什么不过滤低相似度
    ///
    /// 混合模式下**不套用 `min_similarity` 阈值**。
    ///
    /// 阈值是针对余弦相似度设计的，但关键词路的命中是**字面精确匹配**——
    /// 查 "std::Vec" 这类专有名词时，字面命中的片段余弦相似度可能只有 0.2，
    /// 套用阈值就会把最精准的命中过滤掉，关键词路等于白做。
    ///
    /// 两路命中本身就是"相关"的证据（向量路语义近、关键词路字面匹配），
    /// 质量由 RRF 排名和 `top_k` 截断保证。
    pub async fn search(&self, query: &str, top_k: usize) -> anyhow::Result<SearchResult> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(SearchResult::default());
        }

        let mut embeddings = self.embedder.embed(std::slice::from_ref(&query.to_string())).await?;
        let embedding = embeddings
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedder returned no vector for the query"))?;

        let hits = self.store.hybrid_search(&embedding, query, top_k)?;

        Ok(SearchResult {
            candidates: hits.len(),
            hits,
        })
    }

    /// 只走向量检索（不融合关键词路）。
    ///
    /// 语义相近但用词完全不同的情况，向量路往往更好；
    /// 精确术语（函数名、专有名词）则关键词路更准。
    ///
    /// 纯向量路径**会**套用 `min_similarity` 阈值——这里 similarity
    /// 就是唯一的相关性信号，不设阈值会召回一堆不相关内容。
    pub async fn search_vector_only(
        &self,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<SearchResult> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(SearchResult::default());
        }

        let mut embeddings = self.embedder.embed(std::slice::from_ref(&query.to_string())).await?;
        let embedding = embeddings
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedder returned no vector for the query"))?;

        let hits = self.store.query(&embedding, top_k)?;

        let candidates = hits.len();
        let hits: Vec<SearchHit> = hits
            .into_iter()
            .filter(|hit| hit.similarity >= self.min_similarity)
            .collect();

        Ok(SearchResult { hits, candidates })
    }

    /// 只走关键词检索（BM25）。
    ///
    /// 适合查专有名词、函数名、错误码这类"必须字面匹配"的场景。
    ///
    /// 与混合检索同理，**不套用 `min_similarity`**——BM25 的相关性
    /// 已经由排序体现，且字面命中的余弦相似度可能天然偏低。
    pub async fn search_keyword_only(
        &self,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<SearchResult> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(SearchResult::default());
        }

        let hits = self.store.keyword_search(query, top_k)?;

        Ok(SearchResult {
            candidates: hits.len(),
            hits,
        })
    }

    /// 用默认的 `top_k` 检索。
    pub async fn search_default(&self, query: &str) -> anyhow::Result<SearchResult> {
        self.search(query, self.top_k).await
    }

    /// 检索并直接返回可放进 prompt 的文本。
    pub async fn search_and_format(&self, query: &str, top_k: usize) -> anyhow::Result<String> {
        Ok(self.search(query, top_k).await?.format())
    }

    /// 列出全部文档。
    pub fn list_documents(&self) -> anyhow::Result<Vec<Document>> {
        self.store.list_documents()
    }

    /// 按 id 删除文档及其切片，返回被删除的切片数。
    pub fn delete_document(&self, id: &str) -> anyhow::Result<usize> {
        self.store.delete_document(id)
    }

    /// 知识库里的切片总数。
    pub fn total_chunks(&self) -> anyhow::Result<usize> {
        self.store.total_chunks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::embedder::FakeEmbedder;

    /// 用假的 embedding 构造管理器，测试不依赖网络。
    fn test_manager() -> KnowledgeBaseManager {
        KnowledgeBaseManager::with_embedder(":memory:", Box::new(FakeEmbedder::new(32))).unwrap()
    }

    fn document(id: &str, title: &str, content: &str) -> Document {
        Document {
            id: id.to_string(),
            title: title.to_string(),
            source: "test://".to_string(),
            content: content.to_string(),
            created_at: 1,
        }
    }

    #[tokio::test]
    async fn empty_document_with_explicit_id_is_stored() -> anyhow::Result<()> {
        let manager = test_manager();

        let count = manager
            .add_document_with_id(document("doc-empty", "空文档", ""))
            .await?;

        assert_eq!(count, 0);
        assert_eq!(manager.total_chunks()?, 0);
        assert_eq!(manager.list_documents()?.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn empty_query_returns_no_hits() -> anyhow::Result<()> {
        let manager = test_manager();
        let result = manager.search("   ", 3).await?;
        assert!(result.hits.is_empty());
        assert_eq!(result.candidates, 0);
        Ok(())
    }

    #[tokio::test]
    async fn adding_a_document_creates_chunks() -> anyhow::Result<()> {
        let manager = test_manager().with_chunking(20, 5);

        // 36 个字符，chunk_size=20, overlap=5 → step=15 → 起点 0/15/30 → 3 片
        let count = manager
            .add_document(
                "测试文档",
                "test://doc",
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            )
            .await?;

        assert_eq!(count, 3);
        assert_eq!(manager.total_chunks()?, 3);
        assert_eq!(manager.list_documents()?.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn empty_document_records_metadata_but_no_chunks() -> anyhow::Result<()> {
        let manager = test_manager();

        let count = manager.add_document("空文档", "test://empty", "").await?;

        assert_eq!(count, 0);
        assert_eq!(manager.total_chunks()?, 0);
        assert_eq!(manager.list_documents()?.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn reimporting_same_document_id_replaces_old_chunks() -> anyhow::Result<()> {
        let manager = test_manager().with_chunking(20, 5);

        let doc = Document {
            id: "doc-fixed".to_string(),
            title: "初版".to_string(),
            source: "test://doc".to_string(),
            content: "旧内容，旧内容，旧内容，旧内容".to_string(),
            created_at: 1,
        };
        manager.add_document_with_id(doc.clone()).await?;
        let first_count = manager.total_chunks()?;
        assert!(first_count > 0);

        // 同 id 重新入库，内容完全不同
        let updated = Document {
            title: "终版".to_string(),
            content: "全新内容".to_string(),
            ..doc
        };
        manager.add_document_with_id(updated).await?;

        // 旧切片不应残留
        assert_eq!(
            manager.total_chunks()?,
            1,
            "重复入库后旧切片必须被清理，不能残留"
        );

        // 检索到的应该是新内容
        let result = manager.search("全新内容", 3).await?;
        assert!(result.hits.iter().all(|hit| hit.text.contains("全新内容")));

        Ok(())
    }

    #[tokio::test]
    async fn reimporting_with_empty_content_clears_old_chunks() -> anyhow::Result<()> {
        let manager = test_manager().with_chunking(20, 5);

        let doc = Document {
            id: "doc-empty-later".to_string(),
            title: "标题".to_string(),
            source: "test://doc".to_string(),
            content: "有内容的文档".to_string(),
            created_at: 1,
        };
        manager.add_document_with_id(doc.clone()).await?;
        assert!(manager.total_chunks()? > 0);

        // 改成空内容，旧切片也要清掉
        let empty = Document {
            content: String::new(),
            ..doc
        };
        let count = manager.add_document_with_id(empty).await?;

        assert_eq!(count, 0);
        assert_eq!(manager.total_chunks()?, 0);

        Ok(())
    }

    #[tokio::test]
    async fn search_returns_hits_with_document_metadata() -> anyhow::Result<()> {
        let manager = test_manager().with_chunking(50, 10).with_min_similarity(0.0);

        manager
            .add_document("圆的面积", "test://math", "圆的面积等于 π 乘以半径的平方。")
            .await?;
        manager
            .add_document("宇宙速度", "test://physics", "光在真空中的速度是每秒三十万公里。")
            .await?;

        let result = manager.search("圆的面积怎么算", 3).await?;

        assert!(!result.hits.is_empty());
        // 命中的切片带上了文档标题与来源
        assert!(result.hits.iter().any(|hit| !hit.title.is_empty()));
        assert!(result.hits.iter().all(|hit| !hit.source.is_empty()));

        Ok(())
    }

    #[tokio::test]
    async fn min_similarity_filters_out_weak_matches_on_vector_path() -> anyhow::Result<()> {
        // 阈值设到 1.0（余弦相似度上限），正常文本不可能达到
        let manager = test_manager().with_min_similarity(1.0);

        manager
            .add_document("任意", "test://x", "一些内容用于测试过滤逻辑。")
            .await?;

        // 纯向量路径套用阈值
        let result = manager.search_vector_only("毫不相关的问题", 3).await?;
        assert!(result.hits.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn hybrid_path_does_not_apply_min_similarity() -> anyhow::Result<()> {
        // 混合路径不套用阈值——字面精确匹配的片段余弦相似度可能天然偏低，
        // 套用阈值会把最精准的命中误杀
        let manager = test_manager().with_min_similarity(1.0);

        manager
            .add_document("专有名词", "test://x", "std::Vec 的用法与所有权规则")
            .await?;

        // 查询里含专有名词，关键词路能字面命中
        let result = manager.search("std::Vec", 3).await?;
        assert!(
            !result.hits.is_empty(),
            "混合路径不应套用 min_similarity 阈值，否则关键词命中会被误杀"
        );

        Ok(())
    }

    #[tokio::test]
    async fn keyword_only_search_finds_exact_terms() -> anyhow::Result<()> {
        let manager = test_manager();

        manager
            .add_document("网络", "test://net", "TCP 是面向连接的可靠传输协议")
            .await?;
        manager
            .add_document("所有权", "test://rust", "Rust 的所有权机制与借用规则")
            .await?;

        // 关键词路能查到专有名词
        let result = manager.search_keyword_only("TCP", 3).await?;
        assert!(!result.hits.is_empty());
        assert!(result.hits[0].text.contains("TCP"));

        // 向量路也能查到（FakeEmbedder 按字符散列，"TCP" 与"面向连接"无重叠，
        // 所以这里只验证不报错且能返回结果）
        let result = manager.search_vector_only("TCP", 3).await?;
        assert!(result.candidates <= 3);

        Ok(())
    }

    #[tokio::test]
    async fn deleting_a_document_removes_its_chunks() -> anyhow::Result<()> {
        let manager = test_manager();

        manager
            .add_document("A", "test://a", "文档 A 的内容")
            .await?;
        manager
            .add_document("B", "test://b", "文档 B 的内容")
            .await?;

        let docs = manager.list_documents()?;
        assert_eq!(docs.len(), 2);

        let removed = manager.delete_document(&docs[0].id)?;
        assert_eq!(removed, 1);
        assert_eq!(manager.total_chunks()?, 1);

        Ok(())
    }
}
