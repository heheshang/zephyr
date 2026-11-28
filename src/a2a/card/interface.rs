//! Agent Card 的周边类型：接口、能力声明、提供方。
//!
//! 规范 4.4 节把 Agent Card 拆成若干子对象：
//!
//! ```text
//!   AgentCard
//!   ├── supportedInterfaces : AgentInterface[]   （4.4.6）
//!   ├── capabilities        : AgentCapabilities  （4.4.3）
//!   │    └── extensions     : AgentExtension[]   （4.4.4）
//! ├── provider             : AgentProvider      （4.4.2）
//!   ├── skills              : AgentSkill[]       （见 skill.rs）
//!   └── securitySchemes     : SecurityScheme[]   （见 skill.rs）
//! ```
//!
//! 本文件放前三块（接口 / 能力 / 提供方），`skill.rs` 放技能与鉴权，
//! `card.rs` 放卡片本身。

use serde::{Deserialize, Serialize};

use crate::a2a::types::{A2A_PROTOCOL_VERSION, PROTOCOL_BINDING_JSONRPC};

/// agent 的一种协议接口。
///
/// 规范 4.4.6 节。`supportedInterfaces` 是有序列表，**第一项是首选**。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInterface {
    /// 接口地址。生产环境必须是绝对的 HTTPS URL。
    pub url: String,
    /// 协议绑定标识：`JSONRPC` / `GRPC` / `HTTP+JSON`。
    #[serde(rename = "protocolBinding")]
    pub protocol_binding: String,
    /// 协议版本。
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// 可选的租户标识。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

impl AgentInterface {
    /// 构造一个 JSON-RPC 接口。
    pub fn jsonrpc(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            protocol_binding: PROTOCOL_BINDING_JSONRPC.to_owned(),
            protocol_version: A2A_PROTOCOL_VERSION.to_owned(),
            tenant: None,
        }
    }
}

/// agent 支持的可选能力。
///
/// 规范 4.4.3 节。客户端据此判断能不能用流式、能不能用推送通知。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    /// 是否支持流式响应。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,
    /// 是否支持推送通知。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub push_notifications: Option<bool>,
    /// 支持的协议扩展。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<AgentExtension>,
    /// 是否支持在认证后提供扩展版 Agent Card。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extended_agent_card: Option<bool>,
}

/// 一个协议扩展声明。
///
/// 规范 4.4.4 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentExtension {
    /// 扩展的唯一 URI。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// 人类可读描述。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 客户端是否必须理解该扩展。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    /// 扩展的配置参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Map<String, serde_json::Value>>,
}

/// 服务提供方信息。
///
/// 规范 4.4.2 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentProvider {
    /// 提供方网站或文档地址。
    pub url: String,
    /// 提供方组织名。
    pub organization: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_declares_jsonrpc_binding_and_version() {
        let interface = AgentInterface::jsonrpc("https://x.test/a2a");
        assert_eq!(interface.protocol_binding, "JSONRPC");
        assert_eq!(interface.protocol_version, A2A_PROTOCOL_VERSION);

        let json = serde_json::to_string(&interface).unwrap();
        assert!(json.contains("\"protocolBinding\""), "{json}");
        assert!(json.contains("\"protocolVersion\""), "{json}");
        assert!(!json.contains("\"protocol_binding\""), "{json}");
    }

    #[test]
    fn capabilities_omit_empty_fields() {
        let capabilities = AgentCapabilities::default();
        let json = serde_json::to_string(&capabilities).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn extension_serializes_with_camel_case_keys() {
        let extension = AgentExtension {
            uri: Some("x".to_owned()),
            description: None,
            required: Some(true),
            params: None,
        };
        let json = serde_json::to_string(&extension).unwrap();
        assert!(json.contains("\"required\""), "{json}");
    }
}
