use serde::{Deserialize, Serialize};

/// 一篇被索引的原始文档。
///
/// 文档本身**不存向量**，向量只存在于它的切片上。保留原始全文是为了
/// 检索命中后能把上下文补全，而不只是返回一句碎片。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// uuid v4
    pub id: String,
    /// 文档标题。检索命中时优先用它做展示。
    pub title: String,
    /// 文档来源标识：文件路径、URL、或自由描述。
    pub source: String,
    /// 文档全文。
    pub content: String,
    /// 创建时间（Unix 秒）。
    pub created_at: i64,
}

/// 文档的一个切片，是**检索和返回的最小单位**。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// uuid v4
    pub id: String,
    /// 所属文档 id。
    pub document_id: String,
    /// 切片在原文中的起始字符下标，用于还原上下文。
    pub start_offset: usize,
    /// 切片文本。
    pub text: String,
    /// 该切片的向量（JSON 序列化的 f32 数组）。
    pub embedding: Vec<f32>,
}

/// 拼接切片参与 embedding 的文本。
///
/// 拼上标题是刻意的：同一段文字在不同文档里语义可能完全不同，
/// 带上标题能让"讲 Rust 的文档"和"讲 Python 的文档"区分开。
pub fn chunk_embedding_text(title: &str, text: &str) -> String {
    if title.is_empty() {
        text.to_string()
    } else {
        format!("{title}\n{text}")
    }
}

/// 一次检索的命中结果。
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// 命中的切片 id。
    pub chunk_id: String,
    /// 所属文档 id。
    pub document_id: String,
    /// 文档标题。
    pub title: String,
    /// 文档来源。
    pub source: String,
    /// 切片文本。
    pub text: String,
    /// 切片在原文中的起始下标。
    pub start_offset: usize,
    /// 与查询的余弦相似度。
    pub similarity: f32,
}

impl SearchHit {
    /// 把命中格式化成给 LLM 阅读的片段。
    ///
    /// 带上来源和相似度，让模型知道这段话的可信度与出处，
    /// 也方便它判断是否需要进一步追问。
    pub fn format(&self, index: usize) -> String {
        format!(
            "[片段 {}] (相似度 {:.3}, 来源: {})\n{}",
            index + 1,
            self.similarity,
            if self.source.is_empty() {
                &self.title
            } else {
                &self.source
            },
            self.text
        )
    }
}

/// 一次检索的完整结果。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchResult {
    /// 命中的切片，按相似度从高到低。
    pub hits: Vec<SearchHit>,
    /// 参与本次检索的候选切片总数（被 top_k 截断前的数量）。
    pub candidates: usize,
}

impl SearchResult {
    /// 把全部命中格式化成一段可直接放进 prompt 的文本。
    pub fn format(&self) -> String {
        self.hits
            .iter()
            .enumerate()
            .map(|(i, hit)| hit.format(i))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_prefixes_the_title() {
        assert_eq!(
            chunk_embedding_text("Rust 所有权", "每个值都有一个所有者"),
            "Rust 所有权\n每个值都有一个所有者"
        );
    }

    #[test]
    fn embedding_text_without_title_is_the_raw_text() {
        assert_eq!(chunk_embedding_text("", "正文"), "正文");
    }

    #[test]
    fn search_result_formats_numbered_hits() {
        let result = SearchResult {
            hits: vec![
                SearchHit {
                    chunk_id: "c-1".to_string(),
                    document_id: "d-1".to_string(),
                    title: "标题".to_string(),
                    source: "test://a".to_string(),
                    text: "内容一".to_string(),
                    start_offset: 0,
                    similarity: 0.9,
                },
                SearchHit {
                    chunk_id: "c-2".to_string(),
                    document_id: "d-1".to_string(),
                    title: "标题".to_string(),
                    source: "test://a".to_string(),
                    text: "内容二".to_string(),
                    start_offset: 10,
                    similarity: 0.5,
                },
            ],
            candidates: 2,
        };

        let text = result.format();
        assert!(text.contains("[片段 1]"));
        assert!(text.contains("[片段 2]"));
        assert!(text.contains("内容一"));
        assert!(text.contains("内容二"));
        assert!(text.contains("test://a"));
    }

    #[test]
    fn empty_search_result_formats_to_empty_string() {
        assert_eq!(SearchResult::default().format(), "");
    }
}
