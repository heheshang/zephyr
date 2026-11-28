//! Agent Card：agent 的自描述清单。
//!
//! # 规范依据
//!
//! 规范 4.4 节定义了 Agent Card 的结构，8.2 节规定客户端通过
//! `/.well-known/agent-card.json` 发现它。
//!
//! Agent Card 是 A2A 与 MCP 最大的区别：**MCP 的 server 暴露"工具"，
//! A2A 的 agent 暴露"能力（skill）"**。skill 是描述性的——它告诉别人
//! "我能做什么、我擅长什么"，而不是"我有哪些可调用的函数"。
//!
//! # 与本项目的映射
//!
//! 项目里的 [`crate::agent::model::Agent`] 有 `name` / `description` /
//! `instructions` / `toolbox` / `sub_agents`。这里把它们翻译成 A2A 的概念：
//!
//! ```text
//!   Agent.name          ──►  AgentCard.name
//!   Agent.description   ──►  AgentCard.description
//!   Agent.sub_agents    ──►  一条委派 skill（tag 含 "delegation"）
//!   Agent.toolbox       ──►  每条工具一条能力 skill
//! ```
//!
//! 注意 `instructions`（系统提示词）**不进入** skill——它是写给模型看的
//! 内部指令，不是给客户端看的能力声明。详见 [`AgentCard::from_agent`]。
//!
//! 这样一来，一个纯本地的 `Agent` 就能零成本地暴露成 A2A 端点。
//!
//! # 模块结构
//!
//! ```text
//!   card.rs            ── 本文件：AgentCard 与其构造器
//!   card/interface.rs  ── AgentInterface / AgentCapabilities / AgentProvider
//!   card/skill.rs      ── AgentSkill 与鉴权方案声明
//!   card/build.rs      ── 从项目 Agent 生成 skill 列表的映射规则
//! ```

pub mod build;
pub mod interface;
pub mod skill;

use serde::{Deserialize, Serialize};

use crate::a2a::types::{MEDIA_TYPE_JSON, MEDIA_TYPE_TEXT};

pub use interface::{AgentCapabilities, AgentExtension, AgentInterface, AgentProvider};
pub use skill::{AgentSkill, SecurityRequirement, SecurityScheme, SecuritySchemeType};

/// Agent Card：agent 的自描述清单。
///
/// 规范 4.4.1 节。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    /// agent 名称。
    pub name: String,
    /// agent 描述。
    pub description: String,
    /// 支持的接口列表，第一项是首选。
    #[serde(rename = "supportedInterfaces")]
    pub supported_interfaces: Vec<AgentInterface>,
    /// 提供方。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    /// agent 版本。
    pub version: String,
    /// 文档地址。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    /// 能力声明。
    pub capabilities: AgentCapabilities,
    /// 默认输入媒体类型。
    #[serde(rename = "defaultInputModes")]
    pub default_input_modes: Vec<String>,
    /// 默认输出媒体类型。
    #[serde(rename = "defaultOutputModes")]
    pub default_output_modes: Vec<String>,
    /// 能力列表。
    pub skills: Vec<AgentSkill>,
    /// 图标地址。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
    /// 可用的鉴权方案。规范 4.4.7 节。
    ///
    /// 客户端据此知道该往请求里带什么凭据。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub security_schemes: Vec<SecurityScheme>,
    /// 本次交互要求满足的鉴权方案。规范 4.4.7 节。
    ///
    /// 空的表示"不需要鉴权"。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub security: Vec<SecurityRequirement>,
}

impl AgentCard {
    /// 构造一个最小可用的 Agent Card。
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        url: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            supported_interfaces: vec![AgentInterface::jsonrpc(url)],
            provider: None,
            version: version.into(),
            documentation_url: None,
            capabilities: AgentCapabilities::default(),
            default_input_modes: vec![MEDIA_TYPE_TEXT.to_owned(), MEDIA_TYPE_JSON.to_owned()],
            default_output_modes: vec![MEDIA_TYPE_TEXT.to_owned()],
            skills: Vec::new(),
            icon_url: None,
            security_schemes: Vec::new(),
            security: Vec::new(),
        }
    }

    /// 声明支持流式响应。
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.capabilities.streaming = Some(streaming);
        self
    }

    /// 追加一条 skill。
    pub fn with_skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }

    /// 设置提供方信息。
    pub fn with_provider(mut self, provider: AgentProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// 声明一种鉴权方案。
    pub fn with_security_scheme(mut self, scheme: SecurityScheme) -> Self {
        self.security_schemes.push(scheme);
        self
    }

    /// 声明本次交互要求的鉴权方案。
    pub fn with_security_requirement(mut self, requirement: SecurityRequirement) -> Self {
        self.security.push(requirement);
        self
    }

    /// 声明 HTTP Bearer Token 鉴权，并要求客户端必须提供。
    ///
    /// 这是 [`crate::a2a::auth::BearerTokenAuthenticator`] 对应的卡片写法：
    /// 客户端看到 `securitySchemes` 里有 `bearerAuth`，
    /// 就知道该发 `Authorization: Bearer <key>`。
    pub fn with_bearer_auth(self) -> Self {
        self.with_security_scheme(SecurityScheme {
            id: crate::a2a::auth::BEARER_SCHEME_ID.to_owned(),
            scheme_type: SecuritySchemeType::Http,
            description: Some("HTTP Bearer token (API key).".to_owned()),
            params: Some(serde_json::json!({
                "scheme": "bearer",
                "bearerFormat": "opaque",
            })),
        })
        .with_security_requirement(SecurityRequirement {
            scheme: crate::a2a::auth::BEARER_SCHEME_ID.to_owned(),
            scopes: Vec::new(),
        })
    }

    /// 首选接口地址。
    pub fn primary_url(&self) -> Option<&str> {
        self.supported_interfaces
            .first()
            .map(|interface| interface.url.as_str())
    }

    /// 是否声明支持流式。
    pub fn supports_streaming(&self) -> bool {
        self.capabilities.streaming.unwrap_or(false)
    }

    /// 从本项目的 [`crate::agent::model::Agent`] 生成 Agent Card。
    ///
    /// # 转换规则
    ///
    /// A2A 的 skill 是"**我能做什么**"这一粒度的能力声明，而不是
    /// "我有哪些可调用的函数"。因此这里的映射是：
    ///
    /// ```text
    ///   子 agent     ──►  一条委派 skill（tag 含 delegation）
    ///   工具         ──►  一条能力 skill（tag 为工具名）
    ///   Agent.description ──►  AgentCard.description
    /// ```
    ///
    /// # 为什么不用 instructions 充当 skill
    ///
    /// `instructions` 是**系统提示词**——写给模型看的内部指令，
    /// 常常包含"工作规则""注意事项"这类内容。把整段提示词塞进
    /// skill 的 `description`，远端客户端读到的就是一段内部实现细节，
    /// 既不能用来判断"该派什么活儿"，也把 agent 的实现泄露了出去。
    ///
    /// 因此 instructions **不进入** skill。只有在 agent 既没有子节点、
    /// 也没有工具（拿不出任何具体能力）时，才退回到 instructions 的
    /// **首行**作为兜底描述。
    ///
    /// # 确定性
    ///
    /// 工具箱是 `HashMap`，遍历顺序不确定。这里按名字排序，
    /// 保证同样的 agent 总是生成同样的卡片（便于比对与缓存）。
    pub fn from_agent(
        agent: &crate::agent::model::Agent,
        url: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        let description = if agent.description.trim().is_empty() {
            agent
                .instructions
                .first()
                .and_then(|instruction| instruction.lines().next())
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{} agent", agent.name))
        } else {
            agent.description.clone()
        };

        Self {
            name: agent.name.clone(),
            description,
            supported_interfaces: vec![AgentInterface::jsonrpc(url)],
            provider: None,
            version: version.into(),
            documentation_url: None,
            capabilities: AgentCapabilities::default(),
            default_input_modes: vec![MEDIA_TYPE_TEXT.to_owned(), MEDIA_TYPE_JSON.to_owned()],
            default_output_modes: vec![MEDIA_TYPE_TEXT.to_owned()],
            skills: build::build_skills(agent),
            icon_url: None,
            security_schemes: Vec::new(),
            security: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::model::Agent,
        tool::{ToolBox, calculator::CalculatorTool},
    };
    use std::sync::Arc;

    #[test]
    fn card_serializes_with_camel_case_keys() {
        let card = AgentCard::new("demo", "a demo agent", "https://x.test/a2a", "1.0.0");
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"supportedInterfaces\""), "{json}");
        assert!(json.contains("\"defaultInputModes\""), "{json}");
        assert!(json.contains("\"defaultOutputModes\""), "{json}");
        assert!(!json.contains("\"default_input_modes\""), "{json}");
    }

    #[test]
    fn interface_declares_jsonrpc_binding_and_version() {
        let card = AgentCard::new("demo", "d", "https://x.test/a2a", "1.0.0");
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"protocolBinding\":\"JSONRPC\""), "{json}");
        assert!(json.contains("\"protocolVersion\":\"1.0\""), "{json}");
        assert_eq!(card.primary_url(), Some("https://x.test/a2a"));
    }

    #[test]
    fn card_roundtrips_through_json() {
        let card = AgentCard::new("demo", "d", "https://x.test/a2a", "1.0.0")
            .with_streaming(true)
            .with_skill(AgentSkill::new(
                "s1",
                "Skill One",
                "does something",
                vec!["a".into(), "b".into()],
            ));

        let json = serde_json::to_string(&card).unwrap();
        let back: AgentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, card.name);
        assert!(back.supports_streaming());
        assert_eq!(back.skills.len(), 1);
        assert_eq!(back.skills[0].tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn card_from_agent_maps_sub_agents_to_delegation_skills() {
        let root = Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec!["You are a router.".to_owned()],
                Arc::new(ToolBox::new()),
            )
            .with_name("assistant".to_owned())
            .with_description("Routes work".to_owned()),
        );

        let math = Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec![],
                Arc::new(ToolBox::new()),
            )
            .with_name("math".to_owned())
            .with_description("Does math precisely".to_owned()),
        );

        root.set_sub_agents(vec![math]).unwrap();

        let card = AgentCard::from_agent(&root, "https://x.test/a2a", "1.0.0");
        assert_eq!(card.name, "assistant");
        assert_eq!(card.description, "Routes work");
        assert_eq!(card.skills.len(), 1);

        let skill = &card.skills[0];
        assert_eq!(skill.id, "assistant-math");
        assert_eq!(skill.name, "math");
        assert_eq!(skill.description, "Does math precisely");
        assert!(skill.tags.contains(&"delegation".to_owned()));
    }

    #[test]
    fn card_from_agent_maps_tools_to_capability_skills() {
        let mut toolbox = ToolBox::new();
        toolbox.insert("calculator".to_owned(), Box::new(CalculatorTool));

        let agent = Agent::new(
            "test-model".to_owned(),
            vec!["You are precise.".to_owned()],
            Arc::new(toolbox),
        )
        .with_name("math".to_owned())
        .with_description("Math expert".to_owned());

        let card = AgentCard::from_agent(&agent, "https://x.test/a2a", "1.0.0");
        assert_eq!(card.skills.len(), 1);

        let skill = &card.skills[0];
        assert_eq!(skill.id, "math-tool-calculator");
        assert_eq!(skill.name, "calculator");
        assert_eq!(skill.tags, vec!["calculator".to_owned()]);
        // 描述应取工具自己的能力说明，而不是系统提示词。
        assert!(
            skill.description.contains("arithmetic"),
            "unexpected description: {}",
            skill.description
        );
    }

    #[test]
    fn card_from_agent_keeps_instructions_out_of_skills() {
        let agent = Agent::new(
            "test-model".to_owned(),
            vec!["第一行摘要。\nSECRET INTERNAL PROMPT\n更多内部规则".to_owned()],
            Arc::new(ToolBox::new()),
        )
        .with_name("quiet".to_owned());

        let card = AgentCard::from_agent(&agent, "https://x.test/a2a", "1.0.0");
        // 没有任何工具也没有子 agent，走兜底分支。
        assert_eq!(card.skills.len(), 1);
        assert_eq!(card.skills[0].id, "quiet");
        // 兜底只取首行；提示词其余部分不应泄露到卡片里。
        assert_eq!(card.skills[0].description, "第一行摘要。");
        assert!(
            !card.skills[0].description.contains("SECRET INTERNAL PROMPT"),
            "unexpected description: {}",
            card.skills[0].description
        );
    }

    #[test]
    fn card_from_agent_produces_stable_skill_order() {
        let mut toolbox = ToolBox::new();
        toolbox.insert("zeta".to_owned(), Box::new(CalculatorTool));
        toolbox.insert("alpha".to_owned(), Box::new(CalculatorTool));

        let agent = Agent::new("test-model".to_owned(), vec![], Arc::new(toolbox))
            .with_name("stable".to_owned());

        let card = AgentCard::from_agent(&agent, "https://x.test/a2a", "1.0.0");
        let ids: Vec<&str> = card.skills.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["stable-tool-alpha", "stable-tool-zeta"]);
    }

    #[test]
    fn card_declares_bearer_auth_scheme() {
        let card = AgentCard::new("demo", "d", "https://x.test/a2a", "1.0.0").with_bearer_auth();

        assert_eq!(card.security_schemes.len(), 1);
        let scheme = &card.security_schemes[0];
        assert_eq!(scheme.id, "bearerAuth");
        assert_eq!(scheme.scheme_type, SecuritySchemeType::Http);

        assert_eq!(card.security.len(), 1);
        assert_eq!(card.security[0].scheme, "bearerAuth");

        // 序列化成 camelCase，符合规范 5.5 节。
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"securitySchemes\""), "{json}");
        assert!(json.contains("\"security\""), "{json}");
    }

    #[test]
    fn card_without_auth_has_no_security_fields() {
        let card = AgentCard::new("demo", "d", "https://x.test/a2a", "1.0.0");
        assert!(card.security_schemes.is_empty());
        assert!(card.security.is_empty());

        // 未启用鉴权时不应输出空字段。
        let json = serde_json::to_string(&card).unwrap();
        assert!(!json.contains("\"securitySchemes\""), "{json}");
        assert!(!json.contains("\"security\""), "{json}");
    }

    #[test]
    fn card_from_agent_falls_back_to_instructions_for_description() {
        let agent = Agent::new(
            "test-model".to_owned(),
            vec!["Only instruction.".to_owned()],
            Arc::new(ToolBox::new()),
        )
        .with_name("fallback".to_owned());

        let card = AgentCard::from_agent(&agent, "https://x.test/a2a", "1.0.0");
        assert_eq!(card.description, "Only instruction.");
    }
}
