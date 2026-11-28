/// 文本向量化器。
///
/// 抽象出来是为了两件事：
///
/// 1. **可测试**：单元测试注入假实现，不需要网络。
/// 2. **可替换**：将来换本地模型（Ollama / sentence-transformers）时，
///    只需实现这个 trait，检索链路其余部分不用动。
///
/// 默认实现 [`crate::search::embedding::embed_texts`] 走 OpenAI 兼容接口。
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// 对一批文本求向量，返回的向量顺序必须与输入顺序一致。
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// 输出向量的维度。
    ///
    /// 这个值会被写进 vec0 虚表的建表语句，**建库后不可更改**，
    /// 所以必须与 [`Self::embed`] 的实际输出严格一致。
    fn embedding_dim(&self) -> usize;
}

/// 基于 OpenAI 兼容 embedding 接口的实现。
pub struct ApiEmbedder {
    model: String,
    dim: usize,
}

impl ApiEmbedder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            dim: crate::knowledge::index::DEFAULT_EMBEDDING_DIM,
        }
    }

    /// 指定维度。默认是 `text-embedding-v3` 的 1024 维。
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

#[async_trait::async_trait]
impl Embedder for ApiEmbedder {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        crate::search::embedding::embed_texts(texts, &self.model).await
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// 假实现：按文本长度生成确定性向量。仅用于测试。
///
/// 同样的文本永远得到同样的向量，因此可以稳定地断言排序结果。
pub struct FakeEmbedder {
    dim: usize,
}

impl FakeEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait::async_trait]
impl Embedder for FakeEmbedder {
    async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                // 用字符码做确定性散列，相似文本得到相近向量
                let mut vector = vec![0.0f32; self.dim];
                for (i, ch) in text.chars().enumerate() {
                    vector[i % self.dim] += (ch as u32 % 17) as f32 / 17.0;
                }
                vector
            })
            .collect())
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_embedder_is_deterministic() -> anyhow::Result<()> {
        let embedder = FakeEmbedder::new(16);
        let texts = vec!["圆的面积".to_string(), "圆的面积".to_string()];

        let a = embedder.embed(&texts).await?;
        let b = embedder.embed(&texts).await?;

        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        Ok(())
    }

    #[test]
    fn embedding_dim_is_reported() {
        assert_eq!(FakeEmbedder::new(16).embedding_dim(), 16);
        assert_eq!(ApiEmbedder::new("m").embedding_dim(), 1024);
        assert_eq!(ApiEmbedder::new("m").with_dim(512).embedding_dim(), 512);
    }
}
