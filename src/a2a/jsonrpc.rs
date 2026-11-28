//! JSON-RPC 2.0 协议绑定。
//!
//! # 规范依据
//!
//! 规范 9 节定义了 JSON-RPC 绑定：
//!
//! - 请求：`{"jsonrpc":"2.0","id":"...","method":"SendMessage","params":{...}}`
//! - 成功响应：`{"jsonrpc":"2.0","id":"...","result":{...}}`
//! - 错误响应：`{"jsonrpc":"2.0","id":"...","error":{"code":-32001,"message":"..."}}`
//! - 方法名用 **PascalCase**（与服务端方法名一致，不是 REST 风格的动词+连字符）。
//!
//! # 模块职责
//!
//! ```text
//!   请求 ──► 解析 JSON ──► 校验 jsonrpc 版本 ──► 按 method 分派
//!                                                  │
//!                            ┌─────────────────────┼──────────────────────┐
//!                            ▼                     ▼                      ▼
//!                     SendMessage        SendStreamingMessage       GetTask ...
//!                            │                     │                      │
//!                            └──────────► 业务实现 ◄────────────────────┘
//!                                          │
//!                            ┌─────────────┴─────────────┐
//!                            ▼                           ▼
//!                      成功：result                  失败：error
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 协议版本号。
pub const JSONRPC_VERSION: &str = "2.0";

/// JSON-RPC 方法名。
///
/// 规范 9.4 节列出了全部核心方法。这里用常量避免字符串漂移。
pub mod method {
    /// 发送消息（阻塞或立即返回）。
    pub const SEND_MESSAGE: &str = "SendMessage";
    /// 发送消息并订阅流式更新。
    pub const SEND_STREAMING_MESSAGE: &str = "SendStreamingMessage";
    /// 查询任务状态。
    pub const GET_TASK: &str = "GetTask";
    /// 列出任务。
    pub const LIST_TASKS: &str = "ListTasks";
    /// 取消任务。
    pub const CANCEL_TASK: &str = "CancelTask";
    /// 订阅任务的流式更新。
    pub const SUBSCRIBE_TO_TASK: &str = "SubscribeToTask";
    /// 获取扩展版 Agent Card。
    pub const GET_EXTENDED_AGENT_CARD: &str = "GetExtendedAgentCard";
}

/// 一个 JSON-RPC 请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// 协议版本，必须是 `"2.0"`。
    pub jsonrpc: String,
    /// 请求标识。响应必须带回同一个值。
    #[serde(default)]
    pub id: Option<Value>,
    /// 方法名。
    pub method: String,
    /// 方法参数。
    #[serde(default)]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// 构造一个请求。
    pub fn new(id: impl Into<String>, method: &str, params: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id: Some(Value::String(id.into())),
            method: method.to_owned(),
            params: Some(params),
        }
    }

    /// 校验协议版本。
    ///
    /// 规范 9.3 节要求 JSON-RPC 版本必须是 `"2.0"`。
    pub fn validate_version(&self) -> Result<(), crate::a2a::error::A2AError> {
        if self.jsonrpc == JSONRPC_VERSION {
            Ok(())
        } else {
            Err(crate::a2a::error::A2AError::InvalidRequest(format!(
                "unsupported jsonrpc version: {}",
                self.jsonrpc
            )))
        }
    }
}

/// 一个 JSON-RPC 成功响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcSuccess {
    /// 协议版本。
    pub jsonrpc: String,
    /// 与请求一致���标识。
    pub id: Option<Value>,
    /// 结果。
    pub result: Value,
}

impl JsonRpcSuccess {
    /// 构造一个成功响应。
    pub fn new(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            result,
        }
    }
}

/// 一个 JSON-RPC 错误响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcFailure {
    /// 协议版本。
    pub jsonrpc: String,
    /// 与请求一致的标识。
    pub id: Option<Value>,
    /// 错误对象。
    pub error: crate::a2a::error::JsonRpcError,
}

impl JsonRpcFailure {
    /// 构造一个错误响应。
    pub fn new(id: Option<Value>, error: crate::a2a::error::A2AError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            id,
            error: crate::a2a::error::JsonRpcError::from(&error),
        }
    }
}

/// JSON-RPC 响应，可能是成功也可能是失败。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcResponse {
    /// 成功。
    Success(JsonRpcSuccess),
    /// 失败。
    Failure(JsonRpcFailure),
}

impl JsonRpcResponse {
    /// 从结果构造成功响应。
    pub fn success(id: Option<Value>, result: Value) -> Self {
        JsonRpcResponse::Success(JsonRpcSuccess::new(id, result))
    }

    /// 从错误构造失败响应。
    pub fn failure(id: Option<Value>, error: crate::a2a::error::A2AError) -> Self {
        JsonRpcResponse::Failure(JsonRpcFailure::new(id, error))
    }
}

/// HTTP 头中承载的服务参数名。
///
/// 规范 9.2 节：JSON-RPC 绑定下，服务参数通过标准 HTTP 请求头传递。
pub mod service_param {
    /// 协议版本头。
    pub const VERSION: &str = "A2A-Version";
    /// 扩展列表头，多个值用逗号分隔。
    pub const EXTENSIONS: &str = "A2A-Extensions";
}

/// 从 HTTP 头中解析服务参数。
///
/// 规范 9.2 节要求多个值用逗号分隔在同一个头字段里。
pub fn parse_service_params(headers: &std::collections::HashMap<String, String>) -> ServiceParams {
    let mut params = ServiceParams::default();

    for (name, value) in headers {
        match name.to_ascii_lowercase().as_str() {
            "a2a-version" => params.version = Some(value.clone()),
            "a2a-extensions" => {
                params.extensions = value
                    .split(',')
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
            _ => {}
        }
    }

    params
}

/// 服务参数。
#[derive(Debug, Clone, Default)]
pub struct ServiceParams {
    /// 客户端请求的协议版本。
    pub version: Option<String>,
    /// 客户端启用的扩展 URI 列表。
    pub extensions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_serializes_in_json_rpc_2_0_shape() {
        let request = JsonRpcRequest::new("req-1", method::SEND_MESSAGE, json!({"message": {}}));
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""), "{json}");
        assert!(json.contains("\"method\":\"SendMessage\""), "{json}");
        assert!(json.contains("\"id\":\"req-1\""), "{json}");
    }

    #[test]
    fn request_parses_from_json() {
        let raw = r#"{"jsonrpc":"2.0","id":7,"method":"GetTask","params":{"id":"t1"}}"#;
        let request: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "GetTask");
        assert_eq!(request.id, Some(json!(7)));
        assert_eq!(request.params.unwrap()["id"], json!("t1"));
    }

    #[test]
    fn validate_version_accepts_2_0_only() {
        let ok = JsonRpcRequest::new("r", method::GET_TASK, json!({}));
        assert!(ok.validate_version().is_ok());

        let bad = JsonRpcRequest {
            jsonrpc: "1.0".to_owned(),
            id: None,
            method: method::GET_TASK.to_owned(),
            params: None,
        };
        assert!(bad.validate_version().is_err());
    }

    #[test]
    fn success_and_failure_envelopes_are_distinguishable() {
        let success = JsonRpcResponse::success(Some(json!("r1")), json!({"ok": true}));
        let text = serde_json::to_string(&success).unwrap();
        assert!(text.contains("\"result\""), "{text}");
        assert!(!text.contains("\"error\""), "{text}");

        let failure = JsonRpcResponse::failure(
            Some(json!("r1")),
            crate::a2a::error::A2AError::TaskNotFound("t".into()),
        );
        let text = serde_json::to_string(&failure).unwrap();
        assert!(text.contains("\"error\""), "{text}");
        assert!(text.contains("-32001"), "{text}");
    }

    #[test]
    fn service_params_are_read_from_case_insensitive_headers() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("a2a-version".to_owned(), "0.3".to_owned());
        headers.insert(
            "A2A-Extensions".to_owned(),
            "https://e.test/a/v1,https://e.test/b/v1".to_owned(),
        );

        let params = parse_service_params(&headers);
        assert_eq!(params.version.as_deref(), Some("0.3"));
        assert_eq!(params.extensions.len(), 2);
        assert_eq!(params.extensions[0], "https://e.test/a/v1");
    }
}
