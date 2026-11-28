use anyhow::Ok;
use async_openai::types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput};

/// embedding 接口单批条目数上限。
///
/// 百炼 DashScope 的 `text-embedding-v3` 对单次请求的条目数做了硬限制，
/// 超过会直接返回 `batch size is invalid`。实测上限为 **10**。
///
/// 历史记录：早期文档写成 20，是未实测的估计值，实际调用会 400。
pub const EMBEDDING_BATCH_SIZE: usize = 10;

/// 对文本批量求 embedding。
///
/// 输入会被切成若干批（每批最多 [`EMBEDDING_BATCH_SIZE`] 条）分别请求，
/// 再按批次顺序拼接返回。因此调用方**不需要**关心接口的单批限制，
/// 可以传入任意长度的切片列表。
///
/// 注意：接口返回的 `index` 是**批内**序号，不是全局序号，
/// 所以每批必须各自排序后再拼接，否则跨批顺序会错乱。
pub async fn embed_texts(texts: &[String], model: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let client = async_openai::Client::new();
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    for batch in texts.chunks(EMBEDDING_BATCH_SIZE) {
        let request = CreateEmbeddingRequestArgs::default()
            .model(model)
            .input(EmbeddingInput::StringArray(batch.to_vec()))
            .build()?;

        let response = client.embeddings().create(request).await?;
        let mut data = response.data;
        // index 是批内序号，必须先排序再取，否则与输入顺序对不上。
        data.sort_by_key(|embedding| embedding.index);

        let batch_vectors: Vec<Vec<f32>> = data
            .into_iter()
            .map(|embedding| embedding.embedding)
            .collect();

        if batch_vectors.len() != batch.len() {
            anyhow::bail!(
                "embedding API returned {} vectors for a batch of {}",
                batch_vectors.len(),
                batch.len()
            );
        }

        vectors.extend(batch_vectors);
    }

    Ok(vectors)
}

/// 对单条文本求 embedding。
pub async fn embed_text(text: &str, model: &str) -> anyhow::Result<Vec<f32>> {
    let owned = [text.to_string()];
    let mut vectors = embed_texts(&owned, model).await?;
    vectors
        .pop()
        .ok_or_else(|| anyhow::anyhow!("embedding API returned no vectors"))
}

#[cfg(test)]
mod tests {
    use crate::{
        constant::TEXT_EMBEDDING_3_SMALL_MODEL,
        search::{embedding::embed_texts, similarity::cosine_similarity},
    };

    #[tokio::test]
    async fn test_embeddings() -> anyhow::Result<()> {
        dotenv::dotenv().ok();
        let sentences: Vec<String> = vec![
            "The cat is sleeping on the couch",
            "A kitten is playing with a toy",
            "The dog is running in the park",
        ]
        .into_iter()
        .map(|s| s.to_owned())
        .collect();
        let embeddings = embed_texts(&sentences, TEXT_EMBEDDING_3_SMALL_MODEL).await?;

        let cat_kitten = cosine_similarity(&embeddings[0], &embeddings[1]);
        let cat_dog = cosine_similarity(&embeddings[0], &embeddings[2]);

        println!("Cat vs Kitten: {:.3}", cat_kitten);
        println!("Cat vs Dog: {:.3}", cat_dog);

        assert!(cat_kitten > cat_dog);

        Ok(())
    }

    #[tokio::test]
    async fn more_than_one_batch_keeps_input_order() -> anyhow::Result<()> {
        dotenv::dotenv().ok();

        // 造 23 条文本，跨过单批上限 10，验证分批后顺序不乱。
        let sentences: Vec<String> = (0..23)
            .map(|i| format!("第 {i} 句话，用来验证分批 embedding 的顺序保持。"))
            .collect();
        let embeddings = embed_texts(&sentences, TEXT_EMBEDDING_3_SMALL_MODEL).await?;

        assert_eq!(embeddings.len(), sentences.len());

        // 相邻两句的相似度应显著高于首尾两句，说明顺序没有被重排。
        let adjacent = cosine_similarity(&embeddings[0], &embeddings[1]);
        let distant = cosine_similarity(&embeddings[0], &embeddings[22]);
        assert!(
            adjacent > distant,
            "adjacent={adjacent} should be > distant={distant}; order was scrambled"
        );

        Ok(())
    }
}
