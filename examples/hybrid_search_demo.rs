//! 混合检索（向量 + BM25 + RRF）演示。
//!
//! 对比三种检索方式的效果差异：
//! - 纯向量：语义相近但用词不同也能找到
//! - 纯关键词：专有名词、精确术语更准
//! - 混合（RRF）：两者兼顾
//!
//! 运行：`cargo run --example hybrid_search_demo`

use zephyr::{
    constant::TEXT_EMBEDDING_3_SMALL_MODEL,
    knowledge::knowledge_base::KnowledgeBaseManager,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(false)
        .with_max_level(tracing::Level::INFO)
        .init();

    let kb = KnowledgeBaseManager::new(":memory:", TEXT_EMBEDDING_3_SMALL_MODEL)?
        .with_chunking(150, 25)
        .with_top_k(3);

    // 预置资料：故意让其中一条与查询"用词完全不同但语义相近"，
    // 另一条"用词完全一致但语义很远"——用来展示两路召回的互补性
    kb.add_document(
        "网络协议",
        "https://example.com/network",
        "TCP 是面向连接的可靠传输协议，提供重传、排序、流量控制和拥塞控制。\
         UDP 是无连接的，一次发送一个数据报，不保证到达和顺序，但延迟更低、开销更小。\
         QUIC 在 UDP 之上实现了可靠性，相当于加密版的 TCP。",
    )
    .await?;

    kb.add_document(
        "可靠传输",
        "https://example.com/reliable",
        "面向连接的传输协议会确认每个分片并在丢失时重发，因此能保证按序到达。\
         这类机制让数据完整性由协议层负责，应用层无需自己处理丢包。",
    )
    .await?;

    kb.add_document(
        "HTTP 与 QUIC",
        "https://example.com/http",
        "HTTP/3 基于 QUIC，把可靠传输直接建在 UDP 之上，避免了 TCP 队头阻塞。\
         TLS 1.3 的会话恢复让握手更快。",
    )
    .await?;

    println!("知识库共 {} 个切片\n", kb.total_chunks()?);

    for query in [
        "UDP 为什么比 TCP 延迟低",
        "HTTP/3 用的是什么协议",
        "TCP",
    ] {
        println!("════════════════════════════════════════════════════");
        println!("查询：{query}");
        println!("════════════════════════════════════════════════════");

        // ── 纯向量 ──
        let vec_result = kb.search_vector_only(query, 3).await?;
        println!("\n【纯向量检索】命中 {} 条", vec_result.hits.len());
        for hit in &vec_result.hits {
            println!("  [{:.3}] {}", hit.similarity, first_line(&hit.text));
        }

        // ── 纯关键词 ──
        let kw_result = kb.search_keyword_only(query, 3).await?;
        println!("\n【纯关键词检索】命中 {} 条", kw_result.hits.len());
        for hit in &kw_result.hits {
            println!("  [{:.3}] {}", hit.similarity, first_line(&hit.text));
        }

        // ── 混合 ──
        let hybrid_result = kb.search(query, 3).await?;
        println!("\n【混合检索 RRF】命中 {} 条", hybrid_result.hits.len());
        for hit in &hybrid_result.hits {
            println!("  [{:.3}] {}", hit.similarity, first_line(&hit.text));
        }

        println!();
    }

    Ok(())
}

/// 取文本的第一行，用于紧凑展示。
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or(text);
    let chars: Vec<char> = line.chars().collect();
    let snippet: String = chars.iter().take(60).collect();
    if chars.len() > 60 {
        format!("{snippet}…")
    } else {
        snippet
    }
}
