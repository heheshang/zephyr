//! 文档知识库（RAG）演示。
//!
//! 演示完整链路：
//!
//! ```text
//! 文档全文 → 切片 → 批量 embedding → SQLite 落库
//!                              ↓
//! 查询 → embedding → 余弦相似度检索 → top-k 片段
//! ```
//!
//! 运行：`cargo run --example knowledge_demo`

use zephyr::{
    constant::TEXT_EMBEDDING_3_SMALL_MODEL,
    knowledge::knowledge_base::KnowledgeBaseManager,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    // 用文件库演示持久化；换成 ":memory:" 就是纯内存库
    let db_path = "knowledge_demo.db";
    let manager = KnowledgeBaseManager::new(db_path, TEXT_EMBEDDING_3_SMALL_MODEL)?
        // 演示用小切片，让短文档也能切出多片
        .with_chunking(120, 20)
        .with_top_k(3);

    // ── 1. 入库 ────────────────────────────────────────────────
    let documents = [
        (
            "Rust 所有权",
            "https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html",
            "所有权是 Rust 特有的内存管理机制。每个值都有一个所有者(owner)。\
             当所有者离开作用域时，值会被丢弃。Rust 通过编译期检查确保没有内存泄漏、\
             没有悬垂指针、没有数据竞争。借用规则规定同一时刻要么有任意多个不可变借用，\
             要么有一个可变借用。",
        ),
        (
            "TCP 与 UDP",
            "https://example.com/network",
            "TCP 是面向连接的可靠传输协议，提供重传、排序、流量控制和拥塞控制。\
             UDP 是无连接的，一次发送一个数据报，不保证到达和顺序，但延迟更低、开销更小。\
             QUIC 在 UDP 之上实现了可靠性，相当于加密版的 TCP。",
        ),
        (
            "向量检索原理",
            "https://example.com/vector-search",
            "向量检索把文本通过 embedding 模型映射到高维空间，语义相近的文本在空间中距离更近。\
             常用相似度度量是余弦相似度，它衡量两个向量夹角的余弦值，取值范围在 -1 到 1 之间。\
             暴力检索需要计算查询向量与全部文档向量的相似度，数据量大时应改用近似最近邻索引。",
        ),
    ];

    for (title, source, content) in documents {
        let chunks = manager.add_document(title, source, content).await?;
        println!("已入库《{title}》→ {chunks} 个切片");
    }

    println!("\n知识库共 {} 个切片\n", manager.total_chunks()?);

    // ── 2. 检索 ────────────────────────────────────────────────
    for query in ["Rust 的借用规则是什么", "UDP 为什么比 TCP 快", "怎么衡量两个向量的相似度"] {
        println!("查询：{query}");
        println!("{}", "-".repeat(70));

        let result = manager.search_default(query).await?;

        if result.hits.is_empty() {
            println!("  （没有找到相关内容）\n");
            continue;
        }

        for (i, hit) in result.hits.iter().enumerate() {
            println!(
                "[片段 {}] 相似度 {:.3} | 文档《{}》",
                i + 1,
                hit.similarity,
                hit.title
            );
            println!("来源：{}", hit.source);
            println!("内容：{}", hit.text);
            println!("{}", "-".repeat(70));
        }
        println!();
    }

    // ── 3. 检索结果可直接拼进 prompt ──────────────────────────
    let context = manager.search_and_format("所有权机制解决什么问题", 2).await?;
    println!("可直接注入 prompt 的上下文：\n{context}");

    Ok(())
}
