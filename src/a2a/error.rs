//! A2A 错误类型与 JSON-RPC 错误码映射。
//!
//! # 规范依据
//!
//! 规范 5.4 节给出了 A2A 错误类型到各绑定的标准映射。JSON-RPC 绑定用
//! JSON-RPC 2.0 的 `error` 对象承载，因此这里把 A2A 错误映射成
//! `(code, message, data)` 三元组。
//!
//! | A2A 错误                     | JSON-RPC code |
//! |------------------------------|---------------|
//! | `TaskNotFoundError`          | `-32001`      |
//! | `TaskNotCancelableError`     | `-32002`      |
//! | `PushNotificationNotSupportedError` | `-32003` |
//! | `UnsupportedOperationError`  | `-32004`      |
//! | `ContentTypeNotSupportedError` | `-32005`    |
//! | `InvalidAgentResponseError`  | `-32006`      |
//! | `ExtendedAgentCardNotConfiguredError` | `-32007` |
//! | `ExtensionSupportRequiredError` | `-32008`   |
//! | `VersionNotSupportedError`   | `-32009`      |
//!
//! 另外 JSON-RPC 2.0 自身的保留区间是 `-32768` ~ `-32000`，A2A 的错误码落在
//! 这个区间内，因此不会与标准错误（`-32700` 解析错误、`-32600` 非法请求等）
//! 冲突。

use serde::{Deserialize, Serialize};

/// A2A 错误类型。
///
/// 该类型实现了 [`std::error::Error`]，因此可以自动转成 `anyhow::Error`，
/// 不需要手写 `From` 实现（手写会与 anyhow 的 blanket impl 冲突）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum A2AError {
    /// 任务不存在。规范 5.4 节对应 `-32001`。
    TaskNotFound(String),
    /// 任务已进入终态，不能取消。规范 5.4 节对应 `-32002`。
    TaskNotCancelable(String),
    /// 不支持推送通知。规范 5.4 节对应 `-32003`。
    PushNotificationNotSupported,
    /// 不支持的操作（例如服务端未声明 `streaming` 能力）。对应 `-32004`。
    UnsupportedOperation(String),
    /// 不支持的媒体类型。对应 `-32005`。
    ContentTypeNotSupported(String),
    /// agent 返回了非法响应。对应 `-32006`。
    InvalidAgentResponse(String),
    /// 声明了扩展能力但没有配置扩展卡。对应 `-32007`。
    ExtendedAgentCardNotConfigured,
    /// 客户端要求的扩展服务端不支持。对应 `-32008`。
    ExtensionSupportRequired(String),
    /// 协议版本不支持。对应 `-32009`。
    VersionNotSupported(String),
    /// JSON-RPC 层的标准错误：请求格式非法。
    InvalidRequest(String),
    /// JSON-RPC 层的标准错误：方法不存在。
    MethodNotFound(String),
    /// JSON-RPC 层的标准错误：参数非法。
    InvalidParams(String),
    /// JSON-RPC 层的标准错误：内部错误。
    Internal(String),
    /// 鉴权失败。规范 5.4 节没有对应的 A2A 错误类型，
    /// 这里用 `-32010`（仍在 JSON-RPC 保留区间内）。
    AuthenticationFailed(String),
}

impl A2AError {
    /// JSON-RPC 错误码。
    pub fn code(&self) -> i32 {
        match self {
            A2AError::TaskNotFound(_) => -32001,
            A2AError::TaskNotCancelable(_) => -32002,
            A2AError::PushNotificationNotSupported => -32003,
            A2AError::UnsupportedOperation(_) => -32004,
            A2AError::ContentTypeNotSupported(_) => -32005,
            A2AError::InvalidAgentResponse(_) => -32006,
            A2AError::ExtendedAgentCardNotConfigured => -32007,
            A2AError::ExtensionSupportRequired(_) => -32008,
            A2AError::VersionNotSupported(_) => -32009,
            A2AError::InvalidRequest(_) => -32600,
            A2AError::MethodNotFound(_) => -32601,
            A2AError::InvalidParams(_) => -32602,
            A2AError::Internal(_) => -32603,
            A2AError::AuthenticationFailed(_) => {
                crate::a2a::auth::AUTHENTICATION_FAILED_CODE
            }
        }
    }

    /// 面向调用方的错误说明。
    pub fn message(&self) -> String {
        match self {
            A2AError::TaskNotFound(id) => format!("Task not found: {id}"),
            A2AError::TaskNotCancelable(id) => {
                format!("Task is not in a cancelable state: {id}")
            }            A2AError::PushNotificationNotSupported => {
                "Push notifications are not supported by this agent".to_owned()
            }
            A2AError::UnsupportedOperation(detail) => format!("Unsupported operation: {detail}"),
            A2AError::ContentTypeNotSupported(media) => {
                format!("Content type not supported: {media}")
            }
            A2AError::InvalidAgentResponse(detail) => {
                format!("Invalid agent response: {detail}")
            }
            A2AError::ExtendedAgentCardNotConfigured => {
                "Extended agent card is not configured".to_owned()
            }
            A2AError::ExtensionSupportRequired(uri) => {
                format!("Required extension is not supported: {uri}")
            }
            A2AError::VersionNotSupported(version) => {
                format!("Protocol version not supported: {version}")
            }
            A2AError::InvalidRequest(detail) => format!("Invalid request: {detail}"),
            A2AError::MethodNotFound(method) => format!("Method not found: {method}"),
            A2AError::InvalidParams(detail) => format!("Invalid params: {detail}"),
            A2AError::Internal(detail) => format!("Internal error: {detail}"),
            A2AError::AuthenticationFailed(detail) => format!("Authentication failed: {detail}"),
        }
    }

    /// 是否属于"能力未开启"这类**可预期**的错误。
    ///
    /// 服务端据此决定：能力相关的错误应该返回 4xx（客户端请求有问题），
    /// 而不是 5xx（服务端崩了）。
    pub fn is_client_fault(&self) -> bool {
        !matches!(self, A2AError::Internal(_) | A2AError::InvalidAgentResponse(_))
    }

    /// 由 JSON-RPC 错误码还原成 A2A 错误。
    ///
    /// 客户端收到远端的错误响应时调用：这样错误码（例如 `-32002`）
    /// 在跨进程传输后仍然保留语义，而不是被笼统地降级成"内部错误"。
    ///
    /// 注意：远端传来的 `message` 已经是"类型前缀 + 详情"的形式，
    /// 因此这里要从中**还原出标识符**，否则再次调用 [`A2AError::message`]
    /// 会拼出重复的前缀。
    pub fn from_rpc(code: i32, message: String) -> Self {
        match code {
            -32001 => A2AError::TaskNotFound(extract_detail(&message)),
            -32002 => A2AError::TaskNotCancelable(extract_detail(&message)),
            -32003 => A2AError::PushNotificationNotSupported,
            -32004 => A2AError::UnsupportedOperation(extract_detail(&message)),
            -32005 => A2AError::ContentTypeNotSupported(extract_detail(&message)),
            -32006 => A2AError::InvalidAgentResponse(extract_detail(&message)),
            -32007 => A2AError::ExtendedAgentCardNotConfigured,
            -32008 => A2AError::ExtensionSupportRequired(extract_detail(&message)),
            -32009 => A2AError::VersionNotSupported(extract_detail(&message)),
            -32600 => A2AError::InvalidRequest(extract_detail(&message)),
            -32601 => A2AError::MethodNotFound(extract_detail(&message)),
            -32602 => A2AError::InvalidParams(extract_detail(&message)),
            _ => A2AError::Internal(message),
        }
    }
}

impl std::fmt::Display for A2AError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for A2AError {}

/// 从错误消息里剥掉"类型前缀"，只留详情部分。
///
/// 例如 `"Task is not in a cancelable state: task-1"` → `"task-1"`。
///
/// 服务端的 [`A2AError::message`] 会生成带前缀的消息；客户端把它还原成
/// 错误类型时，如果保留了前缀，再次生成消息就会出现
/// `"Task is not in a cancelable state: Task is not in a cancelable state: ..."`
/// 这样的重复。
fn extract_detail(message: &str) -> String {
    // 前缀都是"XXX: "的形式，用最后一个 ": " 切分。
    match message.rfind(": ") {
        Some(position) => message[position + 2..].to_owned(),
        None => message.to_owned(),
    }
}

/// JSON-RPC 2.0 的错误对象。
///
/// 规范 9.5 节要求用标准 JSON-RPC 2.0 错误结构承载 A2A 错误。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// 数字错误码。
    pub code: i32,
    /// 人类可读的消息。
    pub message: String,
    /// 可选的附加数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcError {
    /// 构造一个 JSON-RPC 错误。
    ///
    /// 用于 [`A2AError`] 覆盖不到的场景，例如鉴权失败——
    /// 那是 HTTP 层的拒绝，没有对应的 A2A 标准错误类型。
    pub fn new(code: i32, message: impl Into<String>, data: Option<serde_json::Value>) -> Self {
        Self {
            code,
            message: message.into(),
            data,
        }
    }
}

impl From<&A2AError> for JsonRpcError {
    fn from(error: &A2AError) -> Self {
        Self {
            code: error.code(),
            message: error.message(),
            data: None,
        }
    }
}

impl From<A2AError> for JsonRpcError {
    fn from(error: A2AError) -> Self {
        Self::from(&error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_match_the_specification() {
        assert_eq!(A2AError::TaskNotFound("t".into()).code(), -32001);
        assert_eq!(A2AError::TaskNotCancelable("t".into()).code(), -32002);
        assert_eq!(A2AError::PushNotificationNotSupported.code(), -32003);
        assert_eq!(A2AError::UnsupportedOperation("x".into()).code(), -32004);
        assert_eq!(A2AError::ContentTypeNotSupported("x".into()).code(), -32005);
        assert_eq!(A2AError::InvalidAgentResponse("x".into()).code(), -32006);
        assert_eq!(A2AError::ExtendedAgentCardNotConfigured.code(), -32007);
        assert_eq!(A2AError::ExtensionSupportRequired("x".into()).code(), -32008);
        assert_eq!(A2AError::VersionNotSupported("x".into()).code(), -32009);
    }

    #[test]
    fn standard_json_rpc_codes_are_distinct_from_a2a_codes() {
        assert_eq!(A2AError::InvalidRequest("x".into()).code(), -32600);
        assert_eq!(A2AError::MethodNotFound("x".into()).code(), -32601);
        assert_eq!(A2AError::InvalidParams("x".into()).code(), -32602);
        assert_eq!(A2AError::Internal("x".into()).code(), -32603);
    }

    #[test]
    fn json_rpc_error_serializes_with_code_and_message() {
        let error = A2AError::TaskNotFound("task-1".into());
        let json = serde_json::to_string(&JsonRpcError::from(&error)).unwrap();
        assert!(json.contains("\"code\":-32001"), "{json}");
        assert!(json.contains("\"message\""), "{json}");
        assert!(json.contains("task-1"), "{json}");
    }

    #[test]
    fn client_fault_classification() {
        assert!(A2AError::TaskNotFound("t".into()).is_client_fault());
        assert!(A2AError::UnsupportedOperation("x".into()).is_client_fault());
        assert!(!A2AError::Internal("x".into()).is_client_fault());
        assert!(!A2AError::InvalidAgentResponse("x".into()).is_client_fault());
    }

    #[test]
    fn error_code_survives_a_roundtrip_through_rpc() {
        let original = A2AError::TaskNotCancelable("task-1".into());
        let rpc = JsonRpcError::from(&original);
        let restored = A2AError::from_rpc(rpc.code, rpc.message);

        assert_eq!(restored.code(), original.code());
        assert_eq!(restored.message(), original.message());
    }

    #[test]
    fn restored_error_does_not_duplicate_the_prefix() {
        let original = A2AError::TaskNotCancelable("task-1".into());
        let rpc = JsonRpcError::from(&original);
        let restored = A2AError::from_rpc(rpc.code, rpc.message);

        assert_eq!(
            restored.message(),
            "Task is not in a cancelable state: task-1",
            "the message must not repeat the prefix"
        );
    }

    #[test]
    fn from_rpc_maps_unknown_codes_to_internal() {
        let restored = A2AError::from_rpc(-32099, "something odd".to_owned());
        assert_eq!(restored.code(), -32603);
        assert!(matches!(restored, A2AError::Internal(_)));
    }
}
