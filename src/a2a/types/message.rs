//! 消息与内容片段。
//!
//! 规范 4.1.4 ~ 4.1.6 节。

use serde::{Deserialize, Serialize};

/// 消息的发送方。
///
/// 规范 4.1.5 节：客户端发的是 `ROLE_USER`，服务端回的是 `ROLE_AGENT`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// 未指定。
    #[serde(rename = "ROLE_UNSPECIFIED")]
    Unspecified,
    /// 用户（客户端）发出。
    #[serde(rename = "ROLE_USER")]
    User,
    /// agent（服务端）发出。
    #[serde(rename = "ROLE_AGENT")]
    Agent,
}

impl Role {
    /// 按规范要求序列化为 SCREAMING_SNAKE_CASE 字符串。
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Unspecified => "ROLE_UNSPECIFIED",
            Role::User => "ROLE_USER",
            Role::Agent => "ROLE_AGENT",
        }
    }
}

/// 消息或产物中的一段内容。
///
/// 规范 4.1.6 节：`text` / `raw` / `url` / `data` 是 OneOf，**必须且只能**
/// 有一个被设置。为了在 Rust 侧守住这个不变量，这里不用 `Option` 字段的
/// 结构体，而是用枚举 + `#[serde(untagged)]`：
///
/// - 反序列化时按 `text` → `raw` → `url` → `data` 的顺序匹配第一个命中的；
/// - 序列化时只输出被设置的那一个。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    /// 纯文本内容。
    Text {
        /// 文本正文。
        text: String,
    },
    /// 二进制内容，JSON 中编码为 base64 字符串。
    Raw {
        /// base64 编码的二进制数据。
        raw: String,
    },
    /// 指向文件内容的 URL。
    Url {
        /// 文件地址。
        url: String,
    },
    /// 任意结构化 JSON 数据。
    Data {
        /// 结构化数据。
        data: serde_json::Value,
    },
}

impl Part {
    /// 构造纯文本片段。
    pub fn new_text(text: impl Into<String>) -> Self {
        Part::Text { text: text.into() }
    }

    /// 构造结构化数据片段。
    pub fn new_data(data: serde_json::Value) -> Self {
        Part::Data { data }
    }

    /// 构造 URL 片段。
    pub fn new_url(url: impl Into<String>) -> Self {
        Part::Url { url: url.into() }
    }

    /// 取出文本内容；非文本片段返回 `None`。
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Part::Text { text } => Some(text),
            _ => None,
        }
    }

    /// 提取所有文本片段拼成的字符串。
    ///
    /// 客户端把远端 agent 的回复展示给用户时用这个：结构化数据片段会被
    /// 转成紧凑 JSON，避免信息丢失。
    pub fn collect_text(&self) -> String {
        match self {
            Part::Text { text } => text.clone(),
            Part::Data { data } => data.to_string(),
            Part::Url { url } => url.clone(),
            Part::Raw { raw } => raw.clone(),
        }
    }

    /// 该片段是否为纯文本。
    pub fn is_text(&self) -> bool {
        matches!(self, Part::Text { .. })
    }
}

/// 一条消息。
///
/// 规范 4.1.4 节。`contextId` 是"一组相关交互的集合标识"，`taskId` 指向
/// 具体任务。二者都可选：客户端首条消息通常都不带，由服务端分配。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// 消息唯一标识，由创建方生成。
    #[serde(rename = "messageId")]
    pub message_id: String,
    /// 上下文标识。
    #[serde(rename = "contextId", skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// 任务标识。
    #[serde(rename = "taskId", skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// 发送方。
    pub role: Role,
    /// 消息内容。
    pub parts: Vec<Part>,
    /// 扩展元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Message {
    /// 构造一条只有文本内容的消息。
    pub fn new_text(
        role: Role,
        text: impl Into<String>,
        context_id: Option<String>,
        task_id: Option<String>,
    ) -> Self {
        Self {
            message_id: uuid::Uuid::new_v4().to_string(),
            context_id,
            task_id,
            role,
            parts: vec![Part::new_text(text)],
            metadata: None,
        }
    }

    /// 消息的全部文本内容。
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .map(Part::collect_text)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_serializes_in_screaming_snake_case() {
        assert_eq!(
            serde_json::to_string(&Role::User).unwrap(),
            "\"ROLE_USER\""
        );
        assert_eq!(
            serde_json::to_string(&Role::Agent).unwrap(),
            "\"ROLE_AGENT\""
        );
    }

    #[test]
    fn part_serializes_only_the_set_variant() {
        let part = Part::new_text("hello");
        let json = serde_json::to_string(&part).unwrap();
        assert_eq!(json, r#"{"text":"hello"}"#);
        assert!(part.is_text());
        assert_eq!(part.as_text(), Some("hello"));
    }

    #[test]
    fn part_roundtrips_through_json() {
        for part in [
            Part::new_text("t"),
            Part::new_data(serde_json::json!({"a": 1})),
            Part::new_url("https://example.com/x"),
        ] {
            let json = serde_json::to_string(&part).unwrap();
            let back: Part = serde_json::from_str(&json).unwrap();
            assert_eq!(json, serde_json::to_string(&back).unwrap());
        }
    }

    #[test]
    fn message_uses_camel_case_field_names() {
        let message = Message::new_text(Role::User, "hi", Some("ctx-1".into()), None);
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"messageId\""), "{json}");
        assert!(json.contains("\"contextId\""), "{json}");
        assert!(!json.contains("\"message_id\""), "{json}");
    }
}
