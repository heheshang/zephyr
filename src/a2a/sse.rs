//! SSE 流的解析。
//!
//! # SSE 与 A2A 的两层信封
//!
//! ```text
//!   原始字节 ──► "data: {jsonrpc 信封}\n\n" ──► JSON-RPC 信封
//!                                                        │
//!                                                        └─ result ─► StreamResponse
//! ```
//!
//! 外层是 JSON-RPC 2.0 信封（规范 9.3 节），内层是 `StreamResponse`
//! （规范 3.2.3 节）。SSE 以空行分隔事件，每个事件的数据行以 `data: ` 开头。
//!
//! 解析独立成模块，服务端与客户端都能引用同一套规则——
//! 否则"服务端怎么发"和"客户端怎么读"很容易各写一份然后跑偏。

use futures::StreamExt;

use crate::a2a::{
    jsonrpc::JsonRpcResponse,
    types::StreamResponse,
};

/// 把 SSE 响应流解析成一串 [`StreamResponse`]。
pub fn parse_sse_stream(
    response: reqwest::Response,
) -> impl futures::Stream<Item = anyhow::Result<StreamResponse>> {
    async_stream::stream! {
        let mut bytes = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = bytes.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(anyhow::anyhow!("A2A stream error: {error}"));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // SSE 以空行分隔事件。
            while let Some(position) = buffer.find("\n\n") {
                let event: String = buffer.drain(..position + 2).collect();

                for line in event.lines() {
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim();
                    if payload.is_empty() {
                        continue;
                    }

                    // 外层是 JSON-RPC 信封，内层是 StreamResponse。
                    let envelope: JsonRpcResponse = match serde_json::from_str(payload) {
                        Ok(envelope) => envelope,
                        Err(error) => {
                            yield Err(anyhow::anyhow!("malformed A2A stream event: {error}"));
                            continue;
                        }
                    };

                    match envelope {
                        JsonRpcResponse::Success(success) => {
                            match serde_json::from_value(success.result) {
                                Ok(response) => yield Ok(response),
                                Err(error) => {
                                    yield Err(anyhow::anyhow!(
                                        "malformed A2A stream response: {error}"
                                    ));
                                }
                            }
                        }
                        JsonRpcResponse::Failure(failure) => {
                            yield Err(anyhow::anyhow!("{}", failure.error.message));
                            return;
                        }
                    }
                }
            }
        }
    }
}
