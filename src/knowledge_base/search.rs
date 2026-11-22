use std::{cmp::Ordering, collections::BinaryHeap};

use anyhow::Result;

use crate::{
    constant::QWEN_EMBEDDING_MODEL,
    knowledge_base::embed::{embed_text, embed_texts},
};

/// 单条检索命中结果。
///
/// 由 `vector_search` 在对所有分块计算余弦相似度后产生，
/// `similarity` 越大表示与查询语义越接近（取值范围为 `[-1.0, 1.0]`）。
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// 命中的原始文本分块内容。
    pub text: String,
    /// 该分块与查询向量的余弦相似度得分。
    pub similarity: f32,
}

/// 计算两个向量之间的余弦相似度。
///
/// 公式为 `dot(a, b) / (|a| * |b|)`，取值范围 `[-1.0, 1.0]`，
/// 数值越大表示两个向量方向越一致（语义越相似）。
///
/// # 参数
/// - `a`: 第一个向量，通常是查询向量。
/// - `b`: 第二个向量，通常是分块的向量。
///
/// # 注意
/// - 使用 `zip` 配对，超出较短向量长度的部分会被静默忽略。
/// - 任意一侧为零向量时返回 `0.0`，避免除零产生 `NaN`。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // 点积：对应位置相乘后求和，衡量两个向量的方向一致程度。
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    // 各自模长（欧几里得范数）。
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // 零向量没有方向可言，相似度定义为 0。
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// 堆内部使用的中间结构：携带相似度分数和对应的分块下标。
///
/// 因为 `BinaryHeap` 只存储元素本身、不允许外部携带元数据，
/// 所以这里把「分数 + 下标」打包在一起，才能在堆排序后还原出原始分块。
struct ScoredIndex {
    /// 与查询向量的余弦相似度。
    similarity: f32,
    /// 该分数对应在 `chunks` 切片中的下标。
    index: usize,
}

/// 相等性：仅以相似度判定是否相等。
impl PartialEq for ScoredIndex {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

/// 与 `PartialEq` 一致，`f32` 不满足 `Eq`（因为 `NaN`），此处显式实现。
impl Eq for ScoredIndex {}

/// 排序规则：让 `BinaryHeap` 变成「小顶堆」。
///
/// `BinaryHeap` 总是把 `Ord` 判定为「大于」的元素放在堆顶，
/// 这里反转比较方向（`other` 与 `self` 比较），
/// 于是堆顶就是当前最小分数，弹出时自然淘汰掉最差的那个候选。
///
/// 使用 `total_cmp` 而非 `partial_cmp`：它对 `NaN` 也有全序定义，
/// 不会因为出现 `NaN` 而让比较结果为 `None` 导致堆行为不确定。
impl Ord for ScoredIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        other.similarity.total_cmp(&self.similarity)
    }
}

/// 与 `Ord` 保持一致，直接委托给 `cmp`。
impl PartialOrd for ScoredIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 基于向量余弦相似度做 Top-K 检索。
///
/// 流程：
/// 1. 用嵌入模型把查询文本转成查询向量；
/// 2. 批量把所有分块文本转成向量；
/// 3. 逐个计算余弦相似度，用一个容量为 `top_k` 的小顶堆保留最优的 K 条；
/// 4. 堆取出后按相似度降序排序，映射回原始分块文本。
///
/// # 参数
/// - `query`: 查询语句文本。
/// - `chunks`: 已经切分好的知识库文本块。
/// - `top_k`: 返回的结果条数上限。
///
/// # 返回
/// 按相似度从高到低排列的 `SearchHit` 列表；分块为空时返回空列表。
pub async fn vector_search(query: &str, chunks: &[String], top_k: usize) -> Result<Vec<SearchHit>> {
    // 没有任何候选分块时无需调用嵌入 API，直接返回空结果。
    if chunks.is_empty() {
        return Ok(vec![]);
    }
    // 单条查询向量。
    let query_embedding = embed_text(query, QWEN_EMBEDDING_MODEL).await?;
    // 批量把所有分块转成向量（一次请求，减少网络往返）。
    let chunk_embeddings = embed_texts(chunks, QWEN_EMBEDDING_MODEL).await?;

    // 小顶堆，容量预留 top_k + 1，便于「先 push 再超限 pop」。
    let mut heap = BinaryHeap::with_capacity(top_k + 1);

    for (i, chunk_embedding) in chunk_embeddings.iter().enumerate() {
        // 计算该分块与查询的相似度。
        let similarity = cosine_similarity(&query_embedding, chunk_embedding);
        heap.push(ScoredIndex {
            similarity,
            index: i,
        });
        // 堆内元素超过 top_k 时弹出堆顶（即当前最小的分数），保持只留最优 K 条。
        if heap.len() > top_k {
            heap.pop();
        }
    }

    // `into_vec` 把堆元素取出为普通数组，此时顺序为堆序（不是全局有序）。
    let mut ranked = heap.into_vec();
    // 显式按相似度降序排序，得到最终排名。
    ranked.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));

    // 把下标映射回原始分块文本，输出最终结果。
    Ok(ranked
        .into_iter()
        .map(|scored| SearchHit {
            text: chunks[scored.index].clone(),
            similarity: scored.similarity,
        })
        .collect::<Vec<_>>())
}
