use anyhow::Result;
use async_openai::types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput};

/// 单次 embedding 请求允许的最大输入条数。
///
/// 部分提供方（如 DashScope / Qwen 兼容网关）限制单批输入不得超过 20 条，
/// 超过会返回 `invalid_parameter_error: batch size is invalid`。
const MAX_BATCH_SIZE: usize = 20;

/// 批量把文本转换为向量。
///
/// 内部按 `MAX_BATCH_SIZE` 自动分批请求，避免触发服务端的批量大小限制；
/// 各批结果按输入顺序拼接，返回向量的数量与顺序和入参一一对应。
pub async fn embed_texts(texts: &[String], model: &str) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let client = async_openai::Client::new();
    let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    for batch in texts.chunks(MAX_BATCH_SIZE) {
        let request = CreateEmbeddingRequestArgs::default()
            .model(model)
            .input(EmbeddingInput::StringArray(batch.to_vec()))
            .build()?;

        let response = client.embeddings().create(request).await?;

        // 单批结果按 index 排序，保证与该批输入顺序一致。
        let mut data = response.data;
        data.sort_by_key(|d| d.index);

        all_vectors.extend(data.into_iter().map(|d| d.embedding));
    }

    Ok(all_vectors)
}

/// 把单条文本转换为向量。
pub async fn embed_text(text: &str, model: &str) -> Result<Vec<f32>> {
    let texts = vec![text.to_string()];
    let mut vectors = embed_texts(&texts, model).await?;
    if vectors.is_empty() {
        return Ok(vec![]);
    }
    vectors
        .pop()
        .ok_or_else(|| anyhow::anyhow!("embedding API returned no vectors"))
}
