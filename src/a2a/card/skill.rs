//! Agent Card 的 skill 与鉴权方案声明。
//!
//! 规范 4.4.5 节（skill）、4.4.7 节（鉴权方案）。

use serde::{Deserialize, Serialize};

/// agent 的一种能力（skill）。
///
/// 规范 4.4.5 节。必填字段是 `id` / `name` / `description` / `tags`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    /// 唯一标识。
    pub id: String,
    /// 人类可读名称。
    pub name: String,
    /// 详细描述。
    pub description: String,
    /// 关键词标签。
    pub tags: Vec<String>,
    /// 示例提示词。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    /// 该 skill 支持的输入媒体类型，覆盖 agent 默认值。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_modes: Option<Vec<String>>,
    /// 该 skill 支持的输出媒体类型，覆盖 agent 默认值。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modes: Option<Vec<String>>,
}

impl AgentSkill {
    /// 构造一条 skill。
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        tags: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            tags,
            examples: Vec::new(),
            input_modes: None,
            output_modes: None,
        }
    }

    /// 补充示例提示词。
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }
}

/// 鉴权方案声明。规范 4.4.7 节。
///
/// 目前只支持 HTTP Bearer Token（API Key）。规范里还列了 OAuth2、
/// mTLS 等方案，需要时再扩展。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecurityScheme {
    /// 方案标识，与 [`SecurityRequirement`] 里的名字对应。
    pub id: String,
    /// 方案类型。
    #[serde(rename = "type")]
    pub scheme_type: SecuritySchemeType,
    /// 人类可读描述。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 该方案的具体参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// 鉴权方案的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecuritySchemeType {
    /// HTTP 头里携带 Bearer Token。
    #[serde(rename = "http")]
    Http,
    /// OAuth2。
    #[serde(rename = "oauth2")]
    OAuth2,
    /// 双向 TLS。
    #[serde(rename = "mutualTLS")]
    MutualTls,
    /// OpenID Connect。
    #[serde(rename = "openIdConnect")]
    OpenIdConnect,
    /// API Key（放在 HTTP 头里）。
    #[serde(rename = "apiKey")]
    ApiKey,
}

/// 鉴权要求：某个 scheme 的哪些作用域。
///
/// 目前 Bearer Token 不区分作用域，因此只有一个可选字段。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecurityRequirement {
    /// 对应的 scheme 标识。
    pub scheme: String,
    /// 作用域列表。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
}
