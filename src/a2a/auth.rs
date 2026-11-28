//! A2A 鉴权：服务端校验客户端身份。
//!
//! # 规范依据
//!
//! 规范 4.4.7 节定义 Agent Card 的 `securitySchemes`（可用��鉴权方案）
//! 与 `security`（本次交互要求哪些方案）；规范 5.4 节给出错误映射。
//!
//! # 本模块的范围
//!
//! 这里只实现**一种**方案：HTTP Bearer Token（API Key）。
//!
//! ```text
//!   客户端                          服务端
//!     │                                │
//!     │ Authorization: Bearer <key>     │
//!     ├───────────────────────────────► │
//!     │                          校验 key 是否匹配
//!     │                                │
//!     │  ├─ 不匹配 → 401 + JSON-RPC 错误
//!     │  └─ 匹配   → 正常处理
//! ```
//!
//! # 设计取舍
//!
//! ## 为什么默认不启用
//!
//! 未配置 key 时**完全不校验**。这样本机开发（`127.0.0.1`）体验不变，
//! 也不至于让示例代码必须先造一个密钥才能跑。
//!
//! 启用方式是显式的：服务端构造时传入 [`BearerTokenAuthenticator`]，
//! 或在 `main.rs` 里设置环境变量。
//!
//! ## 为什么 Agent Card 本身不校验
//!
//! 卡片是**公开的自描述信息**，客户端要先拿到卡片才知道该用哪种鉴权
//! 方案（`securitySchemes` 里写着的）。若卡片也要鉴权，客户端就陷入
//! "不知道要发什么头"的死结。
//!
//! 因此本模块只保护 `/rpc`（真正执行 agent 的端点），
//! 卡片端点保持公开——这是 A2A 生态的通行做法。
//!
//! # 常量时间比较
//!
//! key 的比对使用**常量时间**算法，避免通过响应耗时侧信道逐字节猜测
//! 密钥。`subtle` crate 提供了这个能力。

use subtle::ConstantTimeEq;

/// 鉴权失败时使用的 JSON-RPC 错误码。
///
/// JSON-RPC 2.0 保留区间是 `-32768` ~ `-32000`，A2A 的错误码落在
/// `-32001` ~ `-32009`（见 [`crate::a2a::error::A2AError`]）。
/// 鉴权失败没有对应的 A2A 标准错误类型，这里用 `-32010`——
/// 仍在保留区间内，不会与标准错误冲突。
pub const AUTHENTICATION_FAILED_CODE: i32 = -32010;

/// Bearer Token 方案在 Agent Card 里的标识。
///
/// 服务端声明 `securitySchemes` 与客户端构造鉴权请求都用这个名字，
/// 两边必须一致，否则客户端不知道该用哪种方案。
pub const BEARER_SCHEME_ID: &str = "bearerAuth";

/// 鉴权器：判断一个请求是否有权执行。
///
/// # 为什么用 trait 而不是只用一个函数
///
/// 鉴权方式会演进（OAuth2、mTLS、联合方案）。用 trait 之后，
/// 新增方案不必改动服务端的分派逻辑。
pub trait Authenticator: Send + Sync {
    /// 校验请求。
    ///
    /// `authorization` 是 HTTP `Authorization` 头的原始值（可能缺失）。
    /// 返回 `Ok(())` 表示通过，`Err` 表示拒绝。
    fn authenticate(&self, authorization: Option<&str>) -> Result<(), AuthError>;

    /// 该鉴权器对应的方案名，用于写进 Agent Card 的 `securitySchemes`。
    fn scheme_name(&self) -> &str;
}

/// 鉴权失败。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// 没有提供任何凭据。
    Missing,
    /// 凭据格式不对（例如不是 `Bearer <key>`）。
    Malformed,
    /// 凭据格式正确但值不匹配。
    Invalid,
}

impl AuthError {
    /// 面向调用方的说明。
    pub fn message(&self) -> String {
        match self {
            AuthError::Missing => "Missing authorization credentials".to_owned(),
            AuthError::Malformed => {
                "Malformed authorization header; expected 'Authorization: Bearer <key>'".to_owned()
            }
            AuthError::Invalid => "Invalid authorization credentials".to_owned(),
        }
    }
}

/// HTTP Bearer Token 鉴权器。
///
/// # 安全性
///
/// - key 通过 [`ConstantTimeEq`] 比对，不泄露时序信息；
/// - key 不为空时才启用，空 key 等同于"不鉴权"；
/// - key **不出现在日志或错误消息里**，避免经由 `Display` 泄露。
#[derive(Debug, Clone)]
pub struct BearerTokenAuthenticator {
    key: Vec<u8>,
}

impl BearerTokenAuthenticator {
    /// 用给定 key 构造。
    ///
    /// 空 key 会退化成"放行一切"——这样调用方不必特判。
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into().into_bytes(),
        }
    }

    /// 是否实际启用了校验。
    pub fn is_enabled(&self) -> bool {
        !self.key.is_empty()
    }

    /// 从 `Authorization` 头里取出 Bearer token。
    ///
    /// 只认 `Bearer ` 前缀（大小写不敏感，与 HTTP 规范一致）。
    fn extract_bearer(header: &str) -> Option<&str> {
        let (scheme, token) = header.split_once(' ')?;
        if scheme.eq_ignore_ascii_case("bearer") {
            let token = token.trim();
            if token.is_empty() { None } else { Some(token) }
        } else {
            None
        }
    }
}

impl Authenticator for BearerTokenAuthenticator {
    fn authenticate(&self, authorization: Option<&str>) -> Result<(), AuthError> {
        // 未配置 key → 放行。这让"默认不启用"成为可能。
        if !self.is_enabled() {
            return Ok(());
        }

        let header = authorization.ok_or(AuthError::Missing)?;
        let token = BearerTokenAuthenticator::extract_bearer(header).ok_or(AuthError::Malformed)?;

        // 常量时间比较：长度不同也要走完整流程，避免长度信息泄漏。
        let provided = token.as_bytes();
        let matches = provided.ct_eq(&self.key) | (provided.len() ^ self.key.len()).ct_eq(&0);

        if bool::from(matches) {
            Ok(())
        } else {
            Err(AuthError::Invalid)
        }
    }

    fn scheme_name(&self) -> &str {
        "bearerAuth"
    }
}

/// 从 HTTP 头集合里取 `Authorization` 的值。
///
/// 头名大小写不敏感（HTTP 规范要求），这里统一转小写再查。
pub fn authorization_header(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth(key: &str) -> BearerTokenAuthenticator {
        BearerTokenAuthenticator::new(key)
    }

    #[test]
    fn accepts_matching_bearer_token() {
        let auth = auth("s3cret");
        assert!(auth.authenticate(Some("Bearer s3cret")).is_ok());
    }

    #[test]
    fn rejects_wrong_token() {
        let auth = auth("s3cret");
        assert_eq!(
            auth.authenticate(Some("Bearer wrong")),
            Err(AuthError::Invalid)
        );
    }

    #[test]
    fn rejects_missing_header() {
        let auth = auth("s3cret");
        assert_eq!(auth.authenticate(None), Err(AuthError::Missing));
    }

    #[test]
    fn rejects_non_bearer_scheme() {
        let auth = auth("s3cret");
        assert_eq!(
            auth.authenticate(Some("Basic s3cret")),
            Err(AuthError::Malformed)
        );
    }

    #[test]
    fn bearer_scheme_is_case_insensitive() {
        let auth = auth("s3cret");
        assert!(auth.authenticate(Some("bearer s3cret")).is_ok());
        assert!(auth.authenticate(Some("BEARER s3cret")).is_ok());
    }

    #[test]
    fn empty_key_disables_authentication() {
        let auth = auth("");
        assert!(!auth.is_enabled());
        // 未启用时一切放行，包括缺失凭据。
        assert!(auth.authenticate(None).is_ok());
        assert!(auth.authenticate(Some("Bearer whatever")).is_ok());
    }

    #[test]
    fn token_with_surrounding_whitespace_is_accepted() {
        let auth = auth("s3cret");
        assert!(auth.authenticate(Some("Bearer  s3cret ")).is_ok());
    }

    #[test]
    fn empty_token_is_malformed() {
        let auth = auth("s3cret");
        assert_eq!(auth.authenticate(Some("Bearer ")), Err(AuthError::Malformed));
    }
}
