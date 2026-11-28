//! A2A 协议的数据模型。
//!
//! # 与协议规范的对应关系
//!
//! A2A 的规范用 Protocol Buffers 定义数据模型，但**所有协议绑定都必须提供
//! 功能等价的表示**。JSON-RPC 绑定用 JSON，因此这里按规范 5.5 节的要求：
//!
//! - 字段名一律 **camelCase**（不是 protobuf 的 snake_case）；
//! - 枚举值一律 **SCREAMING_SNAKE_CASE**（ProtoJSON 规范的要求）；
//! - 时间戳一律 **ISO 8601 UTC**（规范 5.6.1 节，`2025-10-28T10:30:00.000Z`）。
//!
//! # 模块结构
//!
//! ```text
//!   message.rs ── Role / Part / Message       （规范 4.1.4 ~ 4.1.6）
//!   task.rs    ── TaskState / TaskStatus
//!                 Artifact / Task            （规范 4.1.1 ~ 4.1.3、4.1.7）
//!   stream.rs  ── StreamResponse / 事件
//!                 SendMessage*               （规范 3.2、4.2）
//! ```
//!
//! # 数据模型关系
//!
//! ```text
//!   Part ─┬─ text  (纯文本)
//!         ├─ raw  (二进制，JSON 中为 base64)
//!         ├─ url  (指向文件内容)
//!         └─ data (任意 JSON 结构)
//!
//!   Message ──── parts[] 组成一条消息
//!   Artifact ─── parts[] 组成一个任务产物
//!   Task ─────── status + artifacts[] + history[]
//!   TaskStatus ─ state + message? + timestamp?
//! ```
//!
//! 规范要求 `Part` **必须且只能**包含 `text` / `raw` / `url` / `data` 之一。
//! 这里用 `#[serde(untagged)]` 之外的显式建模来保证这一点：每个变体只允许
//! 一个字段非空，由 `Part::new_text` 等构造函数保证不变量。

pub mod message;
pub mod stream;
pub mod task;

pub use message::{Message, Part, Role};
pub use stream::{
    SendMessageConfiguration, SendMessageRequest, SendMessageResponse, StreamResponse,
    TaskArtifactUpdateEvent, TaskStatusUpdateEvent,
};
pub use task::{Artifact, Task, TaskState, TaskStatus};

/// A2A 协议版本号。
///
/// 规范 4.4.6 节要求 `AgentInterface.protocolVersion` 填写"当前主版本下最新的
/// 次版本号"。本实现对齐 JSON-RPC 绑定的 1.0。
pub const A2A_PROTOCOL_VERSION: &str = "1.0";

/// JSON-RPC 协议绑定标识。
pub const PROTOCOL_BINDING_JSONRPC: &str = "JSONRPC";

/// Agent Card 的 Well-Known 路径。
///
/// 规范 8.2 节与 14.3 节规定客户端通过该路径发现 Agent Card。
pub const WELL_KNOWN_AGENT_CARD_PATH: &str = "/.well-known/agent-card.json";

/// 纯文本的媒体类型。
pub const MEDIA_TYPE_TEXT: &str = "text/plain";
/// JSON 数据的媒体类型。
pub const MEDIA_TYPE_JSON: &str = "application/json";

/// 当前 UTC 时间的 ISO 8601 字符串（毫秒精度）。
///
/// 规范 5.6.1 节要求 `YYYY-MM-DDTHH:mm:ss.sssZ`，且时区必须是 UTC（`'Z'` 后缀，
/// 不允许偏移量）。
pub fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_is_iso8601_utc_with_millis() {
        let ts = now_iso8601();
        assert!(ts.ends_with('Z'), "{ts}");
        assert!(ts.contains('T'), "{ts}");
        // 毫秒精度：形如 2026-01-01T00:00:00.000Z
        assert_eq!(ts.len(), 24, "{ts}");
    }
}
