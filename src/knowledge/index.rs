use std::sync::Once;

use rusqlite::{Connection, params};

use crate::knowledge::{store::f32_to_bytes, tokenizer::tokenize};

/// 常用 embedding 模型的向量维度（`text-embedding-v3`）。
pub const DEFAULT_EMBEDDING_DIM: usize = 1024;

/// RRF 融合常数。取 60 是 Reciprocal Rank Fusion 论文的常用值。
pub const RRF_K: f64 = 60.0;

/// sqlite-vec 扩展的全局注册。
///
/// `sqlite3_auto_extension` 是**进程级**的全局钩子：一旦注册，
/// 之后所有新建的 SQLite 连接都会自动带上 vec0 虚表。
///
/// 用 `Once` 保证只注册一次——重复注册虽然无害，但会白费时间。
static EXTENSION_INIT: Once = Once::new();

/// 注册 sqlite-vec 扩展。必须在使用 vec0 虚表前调用。
pub fn register_extension() {
    EXTENSION_INIT.call_once(|| {
        // SAFETY: sqlite3_auto_extension 只是把一个初始化函数指针登记到
        // SQLite 的全局扩展链表，不涉及内存安全；sqlite-vec 的初始化函数
        // 本身是 C ABI，无 Rust 侧状态。
        //
        // 签名不一致是 sqlite-vec crate 的已知问题——它声明的是
        // `sqlite3_vec_init()`，而 SQLite 期望的是带三个参数的入口函数。
        // 这层 transmute 是官方文档给出的用法。
        unsafe {
            let init: unsafe extern "C" fn(
                *mut libsqlite3_sys::sqlite3,
                *mut *mut i8,
                *const libsqlite3_sys::sqlite3_api_routines,
            ) -> i32 = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            rusqlite::ffi::sqlite3_auto_extension(Some(init));
        }
    });
}

/// 向量索引 + 关键词索引的管理。
///
/// # 表结构
///
/// ```sql
/// -- 向量索引：sqlite-vec 的 vec0 虚表
/// CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
///     chunk_id TEXT PRIMARY KEY,
///     embedding float[1024] distance_metric=cosine
/// );
///
/// -- 关键词索引：FTS5 倒排索引
/// CREATE VIRTUAL TABLE chunk_fts USING fts5(
///     chunk_id UNINDEXED,
///     tokens,
///     tokenize='unicode61'
/// );
/// ```
///
/// 两张表都靠**触发器**与 `chunks` 表同步，因此：
/// - 插入切片时索引自动建立
/// - 删除文档时索引自动清理，不会留下孤儿向量
///
/// # 为什么 `tokenize='unicode61'` 而不是内置的中文分词
///
/// `unicode61` 不切中文（整段中文当一个 token），
/// 但我们在写入前**自己**用 bigram 切好了（见 [`crate::knowledge::tokenizer`]），
/// 所以 FTS5 只需按空格切词即可，`unicode61` 正好合适。
pub struct KnowledgeIndex {
    embedding_dim: usize,
}

impl KnowledgeIndex {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// 建索引表与触发器。
    ///
    /// `vec0` 虚表的向量维度在建表时固定，所以这里必须知道维度。
    pub fn init_schema(conn: &Connection, embedding_dim: usize) -> anyhow::Result<()> {
        register_extension();

        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                 chunk_id TEXT PRIMARY KEY,
                 embedding float[{embedding_dim}] distance_metric=cosine
             );

             CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                 chunk_id UNINDEXED,
                 tokens,
                 tokenize='unicode61'
             );

             -- 切片入库时同步建立两个索引
             CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                 INSERT INTO chunk_embeddings(chunk_id, embedding)
                     VALUES (new.id, new.embedding);
                 INSERT INTO chunk_fts(chunk_id, tokens)
                     VALUES (new.id, fts_tokenize(new.text));
             END;

             -- 切片更新时重建索引
             CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                 DELETE FROM chunk_embeddings WHERE chunk_id = old.id;
                 DELETE FROM chunk_fts WHERE chunk_id = old.id;
                 INSERT INTO chunk_embeddings(chunk_id, embedding)
                     VALUES (new.id, new.embedding);
                 INSERT INTO chunk_fts(chunk_id, tokens)
                     VALUES (new.id, fts_tokenize(new.text));
             END;

             -- 切片删除时清理索引
             CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                 DELETE FROM chunk_embeddings WHERE chunk_id = old.id;
                 DELETE FROM chunk_fts WHERE chunk_id = old.id;
             END;",
        ))?;

        Ok(())
    }

    /// 纯 Rust 的 FTS5 分词函数。
    ///
    /// 注册成 SQLite 自定义函数，让触发器能直接调用——
    /// 这样插入切片时索引自动建立，调用方不必操心同步问题。
    pub fn register_fts_tokenizer(conn: &Connection) -> anyhow::Result<()> {
        conn.create_scalar_function(
            "fts_tokenize",
            1,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8,
            |ctx| {
                let text: String = ctx.get(0)?;
                Ok(tokenize(&text))
            },
        )?;
        Ok(())
    }

    /// 向量检索：取 top-k 最近邻。
    ///
    /// `vec0` 内部维护近似最近邻索引，查询复杂度约 O(log N)，
    /// 远好于全表扫描 + 逐个算余弦。
    ///
    /// 返回 `(chunk_id, cosine 距离)`。距离越小越相似。
    pub fn vector_search(
        conn: &Connection,
        query_embedding: &[f32],
        top_k: usize,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        // vec0 的 MATCH 参数要求向量以 JSON 数组形式传入
        let query_json = embedding_to_json(query_embedding);

        let mut stmt = conn.prepare(
            "SELECT chunk_id, distance FROM chunk_embeddings
             WHERE embedding MATCH ?1 AND k = ?2
             ORDER BY distance",
        )?;

        let rows = stmt.query_map(params![query_json, top_k as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;

        let results = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    /// 关键词检索：BM25 排序。
    ///
    /// 查询串用 **OR** 连接各 token，而不是默认的 AND。
    ///
    /// 原因：bigram 切分后，一个长查询会拆出多个 bigram，
    /// 而这些 bigram 未必都出现在同一篇文档里。用 AND 的话，
    /// 只要缺一个 bigram 就**完全召回不到**任何结果（实测确认）。
    /// 用 OR 放宽召回，再交给 BM25 按词频与 IDF 重排。
    ///
    /// 返回 `(chunk_id, bm25 分数)`。**分数越小越相关**（SQLite 的 bm25()
    /// 返回负值，越负越相关）。
    pub fn keyword_search(
        conn: &Connection,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<Vec<(String, f64)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let match_expr = tokenize_or(query);
        if match_expr.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = conn.prepare(
            "SELECT chunk_id, bm25(chunk_fts) FROM chunk_fts
             WHERE chunk_fts MATCH ?1
             ORDER BY bm25(chunk_fts)
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![match_expr, top_k as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;

        let results = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    /// 索引里的向量数量。
    pub fn vector_count(conn: &Connection) -> anyhow::Result<usize> {
        let count = conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| {
            row.get::<_, i64>(0)
        })?;
        Ok(count as usize)
    }

    /// 索引里的关键词条目数量。
    pub fn fts_count(conn: &Connection) -> anyhow::Result<usize> {
        let count = conn.query_row("SELECT COUNT(*) FROM chunk_fts", [], |row| {
            row.get::<_, i64>(0)
        })?;
        Ok(count as usize)
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// 把向量转成 sqlite-vec 的 JSON 数组形式：`[0.1,0.2,...]`。
pub fn embedding_to_json(embedding: &[f32]) -> String {
    let parts: Vec<String> = embedding.iter().map(|v| v.to_string()).collect();
    format!("[{}]", parts.join(","))
}

/// 生成 FTS5 的 OR 查询表达式。
///
/// 每个 token 用双引号包裹（避免 `*`、`"` 等字符被当作语法），
/// token 之间用 `OR` 连接（放宽召回，见 [`KnowledgeIndex::keyword_search`]）。
pub fn tokenize_or(query: &str) -> String {
    let tokens = crate::knowledge::tokenizer::tokenize_raw(query);
    tokens
        .into_iter()
        .map(|token| format!("\"{token}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// 用查询向量与库里向量算余弦相似度需要的字节表示。
pub fn embedding_bytes(embedding: &[f32]) -> Vec<u8> {
    f32_to_bytes(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_conn(dim: usize) -> anyhow::Result<Connection> {
        register_extension();
        let conn = Connection::open_in_memory()?;
        KnowledgeIndex::register_fts_tokenizer(&conn)?;
        conn.execute_batch(
            "CREATE TABLE chunks (
                 id TEXT PRIMARY KEY,
                 document_id TEXT NOT NULL,
                 start_offset INTEGER NOT NULL DEFAULT 0,
                 text TEXT NOT NULL,
                 embedding BLOB NOT NULL
             );",
        )?;
        KnowledgeIndex::init_schema(&conn, dim)?;
        Ok(conn)
    }

    fn insert_chunk(
        conn: &Connection,
        id: &str,
        text: &str,
        embedding: &[f32],
    ) -> anyhow::Result<()> {
        conn.execute(
            "INSERT INTO chunks (id, document_id, start_offset, text, embedding)
             VALUES (?1, 'doc-1', 0, ?2, ?3)",
            params![id, text, f32_to_bytes(embedding)],
        )?;
        Ok(())
    }

    #[test]
    fn embedding_json_format() {
        assert_eq!(embedding_to_json(&[1.0, 0.5]), "[1,0.5]");
        assert_eq!(embedding_to_json(&[]), "[]");
    }

    #[test]
    fn tokenize_or_joins_with_or() {
        assert_eq!(tokenize_or("TCP 协议"), "\"TCP\" OR \"协议\"");
    }

    #[test]
    fn tokenize_or_is_empty_for_punctuation_only() {
        // 纯符号不产生 token，因此没有 FTS 查询可发
        assert_eq!(tokenize_or("!!!"), "");
        assert_eq!(tokenize_or(""), "");
        assert_eq!(tokenize_or("   "), "");
    }

    #[test]
    fn triggers_populate_both_indexes() -> anyhow::Result<()> {
        let conn = test_conn(4)?;

        insert_chunk(&conn, "c1", "圆的面积公式", &[1.0, 0.0, 0.0, 0.0])?;

        assert_eq!(KnowledgeIndex::vector_count(&conn)?, 1);
        assert_eq!(KnowledgeIndex::fts_count(&conn)?, 1);
        Ok(())
    }

    #[test]
    fn deleting_a_chunk_cleans_both_indexes() -> anyhow::Result<()> {
        let conn = test_conn(4)?;

        insert_chunk(&conn, "c1", "圆的面积", &[1.0, 0.0, 0.0, 0.0])?;
        conn.execute("DELETE FROM chunks WHERE id = 'c1'", [])?;

        assert_eq!(KnowledgeIndex::vector_count(&conn)?, 0);
        assert_eq!(KnowledgeIndex::fts_count(&conn)?, 0);
        Ok(())
    }

    #[test]
    fn vector_search_finds_nearest() -> anyhow::Result<()> {
        let conn = test_conn(4)?;

        insert_chunk(&conn, "near", "A", &[1.0, 0.0, 0.0, 0.0])?;
        insert_chunk(&conn, "far", "B", &[0.0, 1.0, 0.0, 0.0])?;

        let results = KnowledgeIndex::vector_search(&conn, &[0.9, 0.1, 0.0, 0.0], 2)?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "near");
        // cosine 距离：越接近 0 越相似
        assert!(results[0].1 < results[1].1);
        Ok(())
    }

    #[test]
    fn keyword_search_matches_two_character_chinese() -> anyhow::Result<()> {
        let conn = test_conn(4)?;

        insert_chunk(&conn, "c1", "圆的面积公式是什么", &[1.0, 0.0, 0.0, 0.0])?;
        insert_chunk(&conn, "c2", "TCP 协议", &[0.0, 1.0, 0.0, 0.0])?;

        // 2 字中文查询——trigram 分词做不到，bigram 可以
        let results = KnowledgeIndex::keyword_search(&conn, "面积", 5)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "c1");

        Ok(())
    }

    #[test]
    fn keyword_search_matches_english_acronym() -> anyhow::Result<()> {
        let conn = test_conn(4)?;

        insert_chunk(&conn, "c1", "TCP 是面向连接的协议", &[1.0, 0.0, 0.0, 0.0])?;

        let results = KnowledgeIndex::keyword_search(&conn, "TCP", 5)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "c1");

        Ok(())
    }

    #[test]
    fn keyword_search_uses_or_semantics() -> anyhow::Result<()> {
        let conn = test_conn(4)?;

        insert_chunk(&conn, "c1", "Rust 借用规则", &[1.0, 0.0, 0.0, 0.0])?;

        // 查询 "Rust 的借用规则" 切出多个 bigram，
        // 用 AND 会一个都匹配不上，OR 才能召回
        let results = KnowledgeIndex::keyword_search(&conn, "Rust 的借用规则", 5)?;
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn keyword_search_on_empty_query_returns_nothing() -> anyhow::Result<()> {
        let conn = test_conn(4)?;
        insert_chunk(&conn, "c1", "内容", &[1.0, 0.0, 0.0, 0.0])?;

        assert!(KnowledgeIndex::keyword_search(&conn, "", 5)?.is_empty());
        Ok(())
    }

    #[test]
    fn vector_search_on_empty_index_returns_nothing() -> anyhow::Result<()> {
        let conn = test_conn(4)?;
        assert!(KnowledgeIndex::vector_search(&conn, &[1.0, 0.0, 0.0, 0.0], 5)?.is_empty());
        Ok(())
    }
}
