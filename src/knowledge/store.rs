use std::collections::HashMap;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, params};

use crate::{
    knowledge::{
        model::{Chunk, Document, SearchHit},
        tokenizer::tokenize,
    },
    search::similarity::cosine_similarity,
};

/// 文档知识库的 SQLite 存储。
///
/// # 为什么用 `Mutex<Connection>`
///
/// `rusqlite::Connection` 内部用 `RefCell` 做单线程内部可变性，天生不满足
/// `Sync`；而本项目的类型通常要装进 `Box<dyn Tool>` 放进多线程 runtime。
/// `Mutex<T>` 只要 `T: Send` 就自动是 `Sync`，包一层即可满足约束。
/// 这与 `src/memory/store.rs` 的处理方式一致。
///
/// # 表结构
///
/// | 表                | 作用                                             |
/// | ----------------- | ------------------------------------------------ |
/// | `documents`       | 原始文档（id / title / source / content / created_at） |
/// | `chunks`          | 切片（id / document_id / start_offset / text / embedding） |
/// | `chunk_embeddings` | vec0 虚表，向量索引（chunk_id / embedding）        |
/// | `chunk_fts`       | FTS5 虚表，关键词索引（chunk_id / tokens）         |
///
/// # 检索为什么不用全表扫描
///
/// 早期实现把向量以 JSON 存进 `chunks.embedding`，检索时把整表读出来在
/// Rust 侧算余弦相似度——复杂度 O(N) 且每次都要反序列化 N 个数组。
///
/// 现在向量交给 **sqlite-vec** 的 `vec0` 虚表：它自己维护索引，
/// 查询是 O(log N) 的近似最近邻，并且距离由 C 扩展算，省掉反序列化。
///
/// 关键词召回交给 **FTS5**：SQLite 内置的倒排索引 + BM25 排序。
///
/// 两者结果用 **RRF（Reciprocal Rank Fusion）** 融合，理由见
/// [`crate::knowledge::hybrid`]。
pub struct KnowledgeStore {
    conn: Mutex<Connection>,
}

impl KnowledgeStore {
    /// 打开（或创建）指定路径的知识库。
    ///
    /// `embedding_dim` 是向量维度，必须与 embedding 模型输出一致
    /// （`text-embedding-v3` 是 1024 维）。维度写进 vec0 虚表的建表语句，
    /// 所以**不同维度的库不能混用**。
    pub fn open(path: &str, embedding_dim: usize) -> anyhow::Result<Self> {
        // **必须先注册扩展再建连接**：
        // `sqlite3_auto_extension` 是进程级钩子，只对**之后新建**的连接生效。
        // 顺序反了的话，这个连接就看不到 vec0 虚表，建表会报
        // "no such module: vec0"。
        crate::knowledge::index::register_extension();

        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        crate::knowledge::index::KnowledgeIndex::register_fts_tokenizer(&conn)?;
        crate::knowledge::index::KnowledgeIndex::init_schema(&conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// 建表。
    ///
    /// `documents` 与 `chunks` 之间是外键关系，删文档会级联删切片。
    ///
    /// 向量表和 FTS 表都由 [`crate::knowledge::index::KnowledgeIndex`]
    /// 用触发器与 `chunks` 保持同步，这样删除文档时索引会自动清理，
    /// 不会留下孤儿数据。
    pub fn init_schema(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;

             CREATE TABLE IF NOT EXISTS documents (
                 id TEXT PRIMARY KEY,
                 title TEXT NOT NULL,
                 source TEXT NOT NULL DEFAULT '',
                 content TEXT NOT NULL,
                 created_at INTEGER NOT NULL
             );

             CREATE TABLE IF NOT EXISTS chunks (
                 id TEXT PRIMARY KEY,
                 document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                 start_offset INTEGER NOT NULL DEFAULT 0,
                 text TEXT NOT NULL,
                 embedding BLOB NOT NULL
             );

             CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);",
        )?;
        Ok(())
    }

    /// 写入一篇文档。返回文档 id。
    pub fn add_document(&self, document: &Document) -> anyhow::Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO documents (id, title, source, content, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                source = excluded.source,
                content = excluded.content,
                created_at = excluded.created_at",
            params![
                document.id,
                document.title,
                document.source,
                document.content,
                document.created_at,
            ],
        )?;
        Ok(())
    }

    /// 批量写入切片。切片应已带向量。
    ///
    /// 向量以小端 f32 blob 存进 `chunks.embedding`，
    /// 由触发器同步到 vec0 虚表。
    pub fn add_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let mut conn = self.lock()?;
        let tx = conn.transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks (id, document_id, start_offset, text, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(id) DO UPDATE SET
                    document_id = excluded.document_id,
                    start_offset = excluded.start_offset,
                    text = excluded.text,
                    embedding = excluded.embedding",
            )?;

            for chunk in chunks {
                stmt.execute(params![
                    chunk.id,
                    chunk.document_id,
                    chunk.start_offset as i64,
                    chunk.text,
                    f32_to_bytes(&chunk.embedding),
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// 替换文档的全部切片：先删旧切片，再写新切片。
    ///
    /// 这是**幂等入库**的关键——同 id 重复入库时如果不先删旧切片，
    /// 旧切片会残留下来，检索时会召回已经不属于该文档的过期内容。
    ///
    /// 返回 `(删除的旧切片数, 写入的新切片数)`。
    pub fn replace_chunks(
        &self,
        document_id: &str,
        chunks: &[Chunk],
    ) -> anyhow::Result<(usize, usize)> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;

        let removed = tx.execute("DELETE FROM chunks WHERE document_id = ?1", [document_id])?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks (id, document_id, start_offset, text, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;

            for chunk in chunks {
                stmt.execute(params![
                    chunk.id,
                    chunk.document_id,
                    chunk.start_offset as i64,
                    chunk.text,
                    f32_to_bytes(&chunk.embedding),
                ])?;
            }
        }

        tx.commit()?;
        Ok((removed, chunks.len()))
    }

    /// 按 id 取文档。
    pub fn get_document(&self, id: &str) -> anyhow::Result<Option<Document>> {
        let conn = self.lock()?;
        let doc = conn
            .query_row(
                "SELECT id, title, source, content, created_at FROM documents WHERE id = ?1",
                [id],
                |row| {
                    Ok(Document {
                        id: row.get(0)?,
                        title: row.get(1)?,
                        source: row.get(2)?,
                        content: row.get(3)?,
                        created_at: row.get(4)?,
                    })
                },
            )
            .optional()?;
        Ok(doc)
    }

    /// 列出全部文档（按创建时间倒序）。
    pub fn list_documents(&self) -> anyhow::Result<Vec<Document>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, title, source, content, created_at
             FROM documents ORDER BY created_at DESC",
        )?;

        let docs = stmt
            .query_map([], |row| {
                Ok(Document {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    source: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(docs)
    }

    /// 删除文档及其全部切片，返回被删除的切片数。
    ///
    /// 向量索引与 FTS 索引由触发器自动同步清理。
    pub fn delete_document(&self, id: &str) -> anyhow::Result<usize> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;

        let removed_chunks = tx.execute("DELETE FROM chunks WHERE document_id = ?1", [id])?;
        tx.execute("DELETE FROM documents WHERE id = ?1", [id])?;

        tx.commit()?;
        Ok(removed_chunks)
    }

    /// 文档的切片数量。
    pub fn chunk_count(&self, document_id: &str) -> anyhow::Result<usize> {
        let conn = self.lock()?;
        let count = conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?1",
            [document_id],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(count as usize)
    }

    /// 知识库里的切片总数。
    pub fn total_chunks(&self) -> anyhow::Result<usize> {
        let conn = self.lock()?;
        let count = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| {
            row.get::<_, i64>(0)
        })?;
        Ok(count as usize)
    }

    /// 全表扫描算余弦相似度，返回相似度最高的 `top_k` 个切片。
    ///
    /// **降级路径**：仅在向量索引不可用时使用。正常情况走
    /// [`Self::hybrid_search`]，那才是带索引的混合检索。
    pub fn query(&self, query_embedding: &[f32], top_k: usize) -> anyhow::Result<Vec<SearchHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.document_id, c.start_offset, c.text, c.embedding,
                    d.title, d.source
             FROM chunks c
             LEFT JOIN documents d ON d.id = c.document_id",
        )?;

        let mut hits: Vec<SearchHit> = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Vec<u8>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })?;

        for row in rows {
            let (chunk_id, document_id, start_offset, text, embedding, title, source) = row?;

            let similarity = cosine_similarity(query_embedding, &bytes_to_f32(&embedding));
            hits.push(SearchHit {
                chunk_id,
                document_id,
                title: title.unwrap_or_default(),
                source: source.unwrap_or_default(),
                text,
                start_offset: start_offset.max(0) as usize,
                similarity,
            });
        }

        hits.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        hits.truncate(top_k);
        Ok(hits)
    }

    /// 混合检索：向量召回 + 关键词召回，RRF 融合。
    ///
    /// - 向量路：`vec0` 虚表的近似最近邻，O(log N)
    /// - 关键词路：FTS5 的 BM25 排序，倒排索引
    ///
    /// 两路各自取 `top_k * 2` 个候选（多取一些给融合留出空间），
    /// 融合后取前 `top_k`。
    pub fn hybrid_search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<SearchHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let conn = self.lock()?;

        // 多取一些候选，让融合有足够空间
        let candidate_k = top_k * 2;

        let vector_results = crate::knowledge::index::KnowledgeIndex::vector_search(
            &conn,
            query_embedding,
            candidate_k,
        )?;

        let keyword_results = crate::knowledge::index::KnowledgeIndex::keyword_search(
            &conn,
            query_text,
            candidate_k,
        )?;

        // 向量路：距离越小越相似 → 升序排名
        let vector_ranked = crate::knowledge::hybrid::to_ranked_list(&vector_results, true);
        // 关键词路：bm25 越小越相关 → 升序排名
        let keyword_ranked = crate::knowledge::hybrid::to_ranked_list(&keyword_results, true);

        let fused = crate::knowledge::hybrid::fuse(&[vector_ranked, keyword_ranked]);
        if fused.is_empty() {
            return Ok(Vec::new());
        }

        // 收集需要加载的 chunk id
        let ids: Vec<&str> = fused
            .iter()
            .take(top_k)
            .map(|(id, _)| id.as_str())
            .collect();

        // 一次性把元数据与向量取出来，避免 N 次查询
        let mut details: HashMap<String, (String, String, String, usize, Vec<f32>)> =
            HashMap::new();
        {
            let sql = format!(
                "SELECT c.id, c.document_id, c.text, c.start_offset, d.title, c.embedding
                 FROM chunks c LEFT JOIN documents d ON d.id = c.document_id
                 WHERE c.id IN ({})",
                ids.iter().map(|_| "?").collect::<Vec<_>>().join(",")
            );

            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

            let rows = stmt.query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Vec<u8>>(5)?,
                ))
            })?;

            for row in rows {
                let (id, document_id, text, start_offset, title, embedding) = row?;
                details.insert(
                    id,
                    (
                        document_id,
                        text,
                        title.unwrap_or_default(),
                        start_offset.max(0) as usize,
                        bytes_to_f32(&embedding),
                    ),
                );
            }
        }

        // 填充每个命中的余弦相似度。
        //
        // 直接用查出来的向量算，而不是只用向量路的结果——
        // 因为**关键词路独有的命中**（字面精确匹配但语义不近）
        // 也需要一个合理的 similarity 值，否则会被
        // `min_similarity` 阈值过滤掉，等于关键词路白做。
        let mut hits = Vec::with_capacity(ids.len());
        for (id, _) in fused.iter().take(top_k) {
            let Some((document_id, text, title, start_offset, embedding)) = details.get(id) else {
                continue;
            };

            let similarity = cosine_similarity(query_embedding, embedding);

            hits.push(SearchHit {
                chunk_id: id.clone(),
                document_id: document_id.clone(),
                title: title.clone(),
                source: String::new(),
                text: text.clone(),
                start_offset: *start_offset,
                similarity,
            });
        }

        // 补齐来源
        {
            let source_ids: Vec<&str> = hits.iter().map(|hit| hit.chunk_id.as_str()).collect();
            if !source_ids.is_empty() {
                let sql = format!(
                    "SELECT c.id, d.source FROM chunks c
                     LEFT JOIN documents d ON d.id = c.document_id
                     WHERE c.id IN ({})",
                    source_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",")
                );
                let mut stmt = conn.prepare(&sql)?;
                let params: Vec<&dyn rusqlite::ToSql> = source_ids
                    .iter()
                    .map(|id| id as &dyn rusqlite::ToSql)
                    .collect();

                let rows = stmt.query_map(params.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })?;

                for row in rows {
                    let (id, source) = row?;
                    if let Some(hit) = hits.iter_mut().find(|hit| hit.chunk_id == id) {
                        hit.source = source.unwrap_or_default();
                    }
                }
            }
        }

        Ok(hits)
    }

    /// 只走关键词检索（BM25）。
    pub fn keyword_search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<SearchHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let conn = self.lock()?;
        let results = crate::knowledge::index::KnowledgeIndex::keyword_search(&conn, query, top_k)?;

        if results.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        let sql = format!(
            "SELECT c.id, c.document_id, c.text, c.start_offset, d.title, d.source
             FROM chunks c LEFT JOIN documents d ON d.id = c.document_id
             WHERE c.id IN ({})",
            ids.iter().map(|_| "?").collect::<Vec<_>>().join(",")
        );

        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

        let mut details: HashMap<String, SearchHit> = HashMap::new();
        let rows = stmt.query_map(params.as_slice(), |row| {
            Ok(SearchHit {
                chunk_id: row.get::<_, String>(0)?,
                document_id: row.get::<_, String>(1)?,
                text: row.get::<_, String>(2)?,
                start_offset: row.get::<_, i64>(3)?.max(0) as usize,
                title: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                source: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                // 关键词路没有向量相似度，用 bm25 排名折算成中性分值
                similarity: 0.0,
            })
        })?;

        for row in rows {
            let hit = row?;
            details.insert(hit.chunk_id.clone(), hit);
        }

        // 按 BM25 的相关顺序输出
        //
        // similarity 用**排名归一化**而不是直接取 bm25 分数的相反数：
        // bm25 的绝对值没有统一量纲（常见词的分数可能接近 0，
        // 罕见词可能是 -8），直接当相似度没有意义。
        // 排名归一化后：第一名是 1.0，最后一名趋近 0，含义明确。
        let total = results.len();
        let mut hits = Vec::with_capacity(results.len());
        for (rank, (id, _)) in results.iter().enumerate() {
            if let Some(mut hit) = details.remove(id) {
                hit.similarity = (1.0 - (rank as f64 / total as f64)) as f32;
                hits.push(hit);
            }
        }

        Ok(hits)
    }

    fn lock(&self) -> anyhow::Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| anyhow::anyhow!("knowledge store mutex poisoned"))
    }
}

/// 把 sqlite-vec 存的小端 f32 blob 转回 Vec<f32>。
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// 把 f32 数组转成 sqlite-vec 需要的小端 blob。
pub fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// 为切片生成 FTS5 索引用的 token 串。
pub fn fts_tokens(text: &str) -> String {
    tokenize(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(id: &str, title: &str) -> Document {
        Document {
            id: id.to_string(),
            title: title.to_string(),
            source: "test://".to_string(),
            content: "内容".to_string(),
            created_at: 1,
        }
    }

    fn chunk(id: &str, document_id: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            document_id: document_id.to_string(),
            start_offset: 0,
            text: "切片内容".to_string(),
            // 维度必须与 test_store() 建的 vec0 虚表一致（4 维）
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        }
    }

    /// 建一个测试用存储（4 维向量，与测试里的 embedding 对应）。
    fn test_store() -> anyhow::Result<KnowledgeStore> {
        KnowledgeStore::open(":memory:", 4)
    }

    #[test]
    fn document_and_chunks_roundtrip() -> anyhow::Result<()> {
        let store = test_store()?;

        store.add_document(&document("doc-1", "Rust 入门"))?;
        store.add_chunks(&[chunk("c-1", "doc-1")])?;

        let doc = store.get_document("doc-1")?.expect("document should exist");
        assert_eq!(doc.title, "Rust 入门");
        assert_eq!(store.chunk_count("doc-1")?, 1);
        assert_eq!(store.total_chunks()?, 1);

        Ok(())
    }

    #[test]
    fn deleting_a_document_removes_its_chunks() -> anyhow::Result<()> {
        let store = test_store()?;

        store.add_document(&document("doc-1", "A"))?;
        store.add_document(&document("doc-2", "B"))?;
        store.add_chunks(&[chunk("c-1", "doc-1"), chunk("c-2", "doc-2")])?;

        let removed = store.delete_document("doc-1")?;
        assert_eq!(removed, 1);
        assert_eq!(store.total_chunks()?, 1);
        assert!(store.get_document("doc-1")?.is_none());

        Ok(())
    }

    #[test]
    fn query_with_zero_top_k_returns_nothing() -> anyhow::Result<()> {
        let store = test_store()?;
        store.add_document(&document("doc-1", "A"))?;
        store.add_chunks(&[chunk("c-1", "doc-1")])?;

        assert!(store.query(&[1.0, 0.0, 0.0, 0.0], 0)?.is_empty());
        Ok(())
    }

    #[test]
    fn vector_index_is_populated_on_insert() -> anyhow::Result<()> {
        let store = test_store()?;
        store.add_document(&document("doc-1", "A"))?;
        store.add_chunks(&[chunk("c-1", "doc-1")])?;

        let conn = store.lock()?;
        assert_eq!(
            crate::knowledge::index::KnowledgeIndex::vector_count(&conn)?,
            1
        );
        assert_eq!(
            crate::knowledge::index::KnowledgeIndex::fts_count(&conn)?,
            1
        );
        Ok(())
    }

    #[test]
    fn deleting_a_document_cleans_the_indexes() -> anyhow::Result<()> {
        let store = test_store()?;
        store.add_document(&document("doc-1", "A"))?;
        store.add_chunks(&[chunk("c-1", "doc-1")])?;

        store.delete_document("doc-1")?;

        let conn = store.lock()?;
        assert_eq!(
            crate::knowledge::index::KnowledgeIndex::vector_count(&conn)?,
            0
        );
        assert_eq!(
            crate::knowledge::index::KnowledgeIndex::fts_count(&conn)?,
            0
        );
        Ok(())
    }

    #[test]
    fn replace_chunks_removes_old_ones() -> anyhow::Result<()> {
        let store = test_store()?;
        store.add_document(&document("doc-1", "A"))?;
        store.add_chunks(&[chunk("c-1", "doc-1"), chunk("c-2", "doc-1")])?;
        assert_eq!(store.chunk_count("doc-1")?, 2);

        let (removed, added) = store.replace_chunks("doc-1", &[chunk("c-3", "doc-1")])?;

        assert_eq!(removed, 2);
        assert_eq!(added, 1);
        assert_eq!(store.chunk_count("doc-1")?, 1);
        Ok(())
    }

    #[test]
    fn hybrid_search_finds_by_vector_and_keyword() -> anyhow::Result<()> {
        let store = test_store()?;
        store.add_document(&document("doc-1", "面积公式"))?;
        store.add_document(&document("doc-2", "网络协议"))?;

        let mut area_chunk = chunk("c-area", "doc-1");
        area_chunk.text = "圆的面积公式是什么".to_string();
        area_chunk.embedding = vec![1.0, 0.0, 0.0, 0.0];

        let mut net_chunk = chunk("c-net", "doc-2");
        net_chunk.text = "TCP 协议与 UDP 的区别".to_string();
        net_chunk.embedding = vec![0.0, 1.0, 0.0, 0.0];

        store.add_chunks(&[area_chunk, net_chunk])?;

        // 向量路命中
        let hits = store.hybrid_search(&[1.0, 0.0, 0.0, 0.0], "面积", 5)?;
        assert!(hits.iter().any(|hit| hit.chunk_id == "c-area"));

        // 关键词路命中（"TCP" 与向量完全不相关）
        let hits = store.hybrid_search(&[1.0, 0.0, 0.0, 0.0], "TCP", 5)?;
        assert!(hits.iter().any(|hit| hit.chunk_id == "c-net"));

        Ok(())
    }

    #[test]
    fn keyword_search_matches_two_character_chinese() -> anyhow::Result<()> {
        let store = test_store()?;
        store.add_document(&document("doc-1", "面积"))?;

        let mut c = chunk("c-1", "doc-1");
        c.text = "圆的面积公式是什么".to_string();
        store.add_chunks(&[c])?;

        let hits = store.keyword_search("面积", 5)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "c-1");

        Ok(())
    }

    #[test]
    fn float_bytes_roundtrip() {
        let values = [1.0f32, -2.5, 0.0, 3.75];
        let bytes = f32_to_bytes(&values);
        assert_eq!(bytes.len(), 16);
        assert_eq!(bytes_to_f32(&bytes), values.to_vec());
    }

    #[test]
    fn fts_tokens_are_quoted() {
        let tokens = fts_tokens("面积");
        assert!(tokens.contains('"'));
    }
}
