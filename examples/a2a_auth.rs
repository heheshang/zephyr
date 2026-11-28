//! # A2A 鉴权端到端测试（`a2a_auth`）
//!
//! 验证服务端开启 Bearer Token 鉴权后的行为。
//!
//! # 场景
//!
//! ```text
//!   客户端                              服务端（ZEPHYR_A2A_TOKEN=secret）
//!     │                                      │
//!     ├── 无 Authorization 头              ──► 401
//!     ├── 错误的 token                      ──► 401
//!     ├── 非 Bearer 前缀                    ──► 401
//!     ├── 正确的 Bearer token               ──► 200，正常执行 agent
//!     │                                      │
//!     └── GET Agent Card（无凭据）          ──► 200（卡片公开）
//! ```
//!
//! 最后一条是**有意为之**：Agent Card 必须公开，否则客户端拿不到卡片
//! 就不知道该用什么凭据（规范 4.4.7 节把 `securitySchemes` 放在卡片里
//! 正是为了告诉客户端该怎么做）。
//!
//! # 用法
//!
//! ```bash
//! cargo run --example a2a_auth
//! ```

use zephyr::a2a::{client::A2AClient, types::TaskState};

/// 测试用的服务端地址。
///
/// 可以用 `A2A_AUTH_TEST_URL` 覆盖，方便指向任意端口的服务端。
const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8981";
/// 测试用的密钥。
const TOKEN: &str = "test-secret-token";
/// 错误密钥。
const WRONG_TOKEN: &str = "wrong-token";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let server_url =
        std::env::var("A2A_AUTH_TEST_URL").unwrap_or_else(|_| DEFAULT_SERVER_URL.to_owned());

    println!("\n========== A2A 鉴权端到端测试 ==========\n");
    println!("服务端: {server_url}");
    println!("密钥  : {TOKEN}\n");

    let mut passed = 0usize;
    let mut failed = 0usize;

    // --------------------------------------------------------
    // TEST 1: Agent Card 仍然公开（客户端要靠它才知道用哪种凭据）
    // --------------------------------------------------------
    println!("[TEST 1] Agent Card 公开可发现");
    match reqwest::get(format!("{server_url}/.well-known/agent-card.json")).await {
        Ok(response) => {
            let status = response.status();
            let body: serde_json::Value = response.json().await.unwrap_or_default();
            println!("  · HTTP 状态: {status}");

            let schemes = body
                .get("securitySchemes")
                .and_then(|value| value.as_array())
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| entry.get("id").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            println!("  · securitySchemes: {schemes:?}");

            let security = body
                .get("security")
                .and_then(|value| value.as_array())
                .map(|entries| entries.len())
                .unwrap_or(0);
            println!("  · security 要求数: {security}");

            check(
                status.is_success(),
                "Agent Card 无需凭据即可访问",
                &mut passed,
                &mut failed,
            );
            check(
                schemes.contains(&"bearerAuth"),
                "卡片声明了 bearerAuth 方案",
                &mut passed,
                &mut failed,
            );
            check(security > 0, "卡片声明了鉴权要求", &mut passed, &mut failed);
        }
        Err(error) => {
            println!("  ✗ 发现失败: {error}");
            failed += 1;
        }
    }

    // --------------------------------------------------------
    // TEST 2: 无凭据 → 拒绝
    // --------------------------------------------------------
    println!("\n[TEST 2] 无凭据访问 /rpc");
    match raw_rpc(&server_url, None).await {
        Ok(status) => {
            println!("  · HTTP 状态: {status}");
            check(
                status.as_u16() == 401,
                "无凭据被拒（401）",
                &mut passed,
                &mut failed,
            );
        }
        Err(error) => {
            println!("  ✗ 请求失败: {error}");
            failed += 1;
        }
    }

    // --------------------------------------------------------
    // TEST 3: 错误 token → 拒绝
    // --------------------------------------------------------
    println!("\n[TEST 3] 错误 token");
    match raw_rpc(&server_url, Some(WRONG_TOKEN)).await {
        Ok(status) => {
            println!("  · HTTP 状态: {status}");
            check(
                status.as_u16() == 401,
                "错误 token 被拒（401）",
                &mut passed,
                &mut failed,
            );
        }
        Err(error) => {
            println!("  ✗ 请求失败: {error}");
            failed += 1;
        }
    }

    // --------------------------------------------------------
    // TEST 4: 非 Bearer 前缀 → 拒绝
    // --------------------------------------------------------
    println!("\n[TEST 4] 非 Bearer 前缀");
    match raw_rpc_with_scheme(&server_url, "Basic").await {
        Ok(status) => {
            println!("  · HTTP 状态: {status}");
            check(
                status.as_u16() == 401,
                "Basic 前缀被拒（401）",
                &mut passed,
                &mut failed,
            );
        }
        Err(error) => {
            println!("  ✗ 请求失败: {error}");
            failed += 1;
        }
    }

    // --------------------------------------------------------
    // TEST 5: 正确 token → 通过
    // --------------------------------------------------------
    println!("\n[TEST 5] 正确 token");
    let client = match A2AClient::discover_with_auth(&server_url, TOKEN).await {
        Ok(client) => {
            println!("  ✓ 发现成功（带凭据）");
            passed += 1;
            client
        }
        Err(error) => {
            println!("  ✗ 发现失败: {error}");
            failed += 1;
            return report(passed, failed);
        }
    };

    match client.send_text("请用一句话介绍你自己").await {
        Ok(task) => {
            println!("  · 任务状态: {:?}", task.status.state);
            check(
                matches!(task.status.state, TaskState::Completed),
                "带凭据的调用成功执行 agent",
                &mut passed,
                &mut failed,
            );
        }
        Err(error) => {
            println!("  ✗ 调用失败: {error}");
            failed += 1;
        }
    }

    // --------------------------------------------------------
    // TEST 6: 客户端后续请求自动带凭据
    // --------------------------------------------------------
    println!("\n[TEST 6] 多轮请求自动携带凭据");
    match client.send_text("再补充一句").await {
        Ok(task) => {
            println!("  · 任务状态: {:?}", task.status.state);
            check(
                matches!(task.status.state, TaskState::Completed),
                "第二轮仍通过鉴权",
                &mut passed,
                &mut failed,
            );
        }
        Err(error) => {
            println!("  ✗ 调用失败: {error}");
            failed += 1;
        }
    }

    report(passed, failed)
}

/// 发一个裸 JSON-RPC 请求，返回 HTTP 状态码。
///
/// 不经过 [`A2AClient`]——这样才能验证"没带凭据"这个场景。
async fn raw_rpc(server_url: &str, token: Option<&str>) -> anyhow::Result<reqwest::StatusCode> {
    raw_rpc_with_scheme_inner(server_url, token, "Bearer").await
}

/// 发一个裸 JSON-RPC 请求，但用指定的认证方案前缀。
async fn raw_rpc_with_scheme(
    server_url: &str,
    scheme: &str,
) -> anyhow::Result<reqwest::StatusCode> {
    raw_rpc_with_scheme_inner(server_url, Some(TOKEN), scheme).await
}

/// 裸 JSON-RPC 请求的公共实现。
async fn raw_rpc_with_scheme_inner(
    server_url: &str,
    token: Option<&str>,
    scheme: &str,
) -> anyhow::Result<reqwest::StatusCode> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "m1",
                "role": "ROLE_USER",
                "parts": [{"text": "你好"}]
            }
        }
    });

    let mut builder = reqwest::Client::new()
        .post(format!("{server_url}/rpc"))
        .header("Content-Type", "application/json");

    if let Some(token) = token {
        builder = builder.header("Authorization", format!("{scheme} {token}"));
    }

    let response = builder.json(&body).send().await?;
    Ok(response.status())
}

/// 记录一条检查结果。
fn check(condition: bool, label: &str, passed: &mut usize, failed: &mut usize) {
    if condition {
        println!("  ✓ {label}");
        *passed += 1;
    } else {
        println!("  ✗ {label}");
        *failed += 1;
    }
}

/// 汇总测试结果。
fn report(passed: usize, failed: usize) -> anyhow::Result<()> {
    println!("\n========== 测试结果 ==========");
    println!("通过: {passed}");
    println!("失败: {failed}");
    if failed == 0 {
        println!("\n全部通过 ✓");
    } else {
        println!("\n存在失败项 ✗");
    }
    Ok(())
}
