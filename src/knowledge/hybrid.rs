use std::collections::HashMap;

use crate::knowledge::index::RRF_K;

/// 一路召回的结果：切片 id + 该路的排名（0 表示最好）。
pub type RankedList = Vec<(String, usize)>;

/// 用 Reciprocal Rank Fusion 融合多路召回结果。
///
/// # 为什么要 RRF 而不是加权分数相加
///
/// 向量检索返回的是**余弦相似度**（0~1，越大越像），
/// 关键词检索返回的是 **BM25 分数**（负值，越小越相关）。
/// 两者量纲完全不同，没有可直接比较的基准——
///
/// - 归一化后加权相加：归一化方式（min-max？z-score？）会随查询变化，
///   同一个词在不同查询下的权重会漂移；
/// - 直接相加：更糟，BM25 的负值会把向量分数抵消掉。
///
/// RRF 只用**排名**不用分数，天然规避了量纲问题：
///
/// ```text
/// score(d) = Σ  1 / (k + rank_i(d))
///            i
/// ```
///
/// 出现在多路结果里的文档会累积得分，这正是混合检索想要的行为——
/// 被向量和关键词同时命中的文档排在最前。
///
/// `k = 60` 是 Reciprocal Rank Fusion 论文的推荐值，
/// 它的作用是压平头部的差距：rank 1 和 rank 2 的分差很小，
/// 避免"某一路的第一名"直接锁定最终结果。
pub fn reciprocal_rank_fusion(lists: &[RankedList], k: f64) -> Vec<(String, f64)> {
    let mut scores: HashMap<&str, f64> = HashMap::new();

    for list in lists {
        for (id, rank) in list {
            *scores.entry(id.as_str()).or_insert(0.0) += 1.0 / (k + (*rank as f64 + 1.0));
        }
    }

    let mut fused: Vec<(String, f64)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect();

    // 分数降序；同分时按 id 排序，保证结果稳定可复现
    fused.sort_by(|a, b| {
        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
    });

    fused
}

/// 用默认的 `k = 60` 融合。
pub fn fuse(lists: &[RankedList]) -> Vec<(String, f64)> {
    reciprocal_rank_fusion(lists, RRF_K)
}

/// 把 sqlite-vec 的距离结果转成排名列表。
///
/// 距离越小越相似，所以要先按距离升序排，再编排名。
pub fn to_ranked_list(results: &[(String, f64)], ascending: bool) -> RankedList {
    let mut sorted = results.to_vec();
    if ascending {
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    } else {
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    }

    sorted
        .into_iter()
        .enumerate()
        .map(|(rank, (id, _))| (id, rank))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty() {
        assert!(fuse(&[]).is_empty());
        assert!(fuse(&[Vec::new()]).is_empty());
    }

    #[test]
    fn single_list_preserves_order() {
        let list: RankedList = vec![("a".to_string(), 0), ("b".to_string(), 1)];
        let fused = fuse(&[list]);

        assert_eq!(fused[0].0, "a");
        assert_eq!(fused[1].0, "b");
    }

    #[test]
    fn documents_in_both_lists_rank_higher() {
        let vec_list: RankedList = vec![("a".to_string(), 0), ("b".to_string(), 1)];
        let fts_list: RankedList = vec![("b".to_string(), 0), ("a".to_string(), 1)];

        let fused = fuse(&[vec_list, fts_list]);

        // a 和 b 各出现两次，得分相同，但 a 在向量路排前面
        assert_eq!(fused.len(), 2);
        assert!((fused[0].1 - fused[1].1).abs() < f64::EPSILON);
        assert_eq!(fused[0].0, "a");
    }

    #[test]
    fn appearing_in_both_lists_beats_single_list() {
        // "shared" 出现在两路，"solo" 只出现在向量路
        let vec_list: RankedList = vec![
            ("shared".to_string(), 0),
            ("solo".to_string(), 1),
        ];
        let fts_list: RankedList = vec![("shared".to_string(), 0)];

        let fused = fuse(&[vec_list, fts_list]);

        // shared 累积了两路的得分，应该排在只出现一路的 solo 前面
        assert_eq!(fused[0].0, "shared");
        assert!(fused[0].1 > fused[1].1);
        assert_eq!(fused[1].0, "solo");
    }

    #[test]
    fn fusion_is_deterministic_for_tied_scores() {
        let list_a: RankedList = vec![("z".to_string(), 0), ("y".to_string(), 1)];
        let list_b: RankedList = vec![("y".to_string(), 0), ("z".to_string(), 1)];

        // 同分时按 id 字典序，保证可复现
        let fused = fuse(&[list_a, list_b]);
        assert_eq!(fused[0].0, "y");
        assert_eq!(fused[1].0, "z");
    }

    #[test]
    fn to_ranked_list_sorts_by_value() {
        // 向量距离：越小越相似
        let results = vec![
            ("far".to_string(), 0.9),
            ("near".to_string(), 0.1),
        ];
        let ranked = to_ranked_list(&results, true);
        assert_eq!(ranked[0].0, "near");
        assert_eq!(ranked[0].1, 0);

        // BM25 分数：越小越相关
        let bm25 = vec![
            ("weak".to_string(), -0.5),
            ("strong".to_string(), -8.0),
        ];
        let ranked = to_ranked_list(&bm25, true);
        assert_eq!(ranked[0].0, "strong");
    }
}
