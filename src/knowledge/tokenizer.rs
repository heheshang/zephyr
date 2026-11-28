//! 中英混合分词器。
//!
//! # 为什么需要自己分词
//!
//! SQLite FTS5 自带的分词器对中文支持很差：
//!
//! | 分词器       | "面积" 能搜到 "圆的面积公式" 吗 |
//! | ------------ | ----------------------------- |
//! | `unicode61`  | ❌ ���中文按整段当一个 token     |
//! | `trigram`    | ❌ trigram 要求 ≥3 字符        |
//!
//! 实测 `trigram` 能匹配 "面积公式"（4 字），但匹配不了 "面积"（2 字），
//! 而 2 字查询在中文里非常常见。
//!
//! # 方案
//!
//! 自己把文本切成 **bigram（相邻两字）**，写进 FTS5 的普通列：
//!
//! ```text
//! "圆的面积公式"  →  圆的 的面 面积 积公 公式
//! ```
//!
//! 查询 "面积" 也切成 `面积`，于是能精确命中。
//!
//! 英文不能这么切（"TCP" 切成 "TC"/"CP" 就搜不到了），
//! 所以**连续 ASCII 字母数字整体保留为一个 token**。
//!
//! ```text
//! "TCP 是面向连接的协议"  →  TCP 是面 面向 向连 连接 接的 的协 协议
//! ```
//!
//! # 代价
//!
//! bigram 会让索引变大、查询变慢，但对中文检索这是必要代价。
//! 另外 bigram 会产生一些无意义的跨字组合（"面积" 与 "积累" 共享 "积"），
//! 这一点由 BM25 的 IDF 加权来缓解——罕见字的 bigram 权重更高。

/// 把文本切成用于 FTS5 索引 / 查询的 token 序列。
///
/// 返回的 token 以空格分隔，且**每个 token 都被双引号包裹**。
/// 引号是必要的：FTS5 的 MATCH 语法里 `*`、`"`、`(` 等是特殊字符，
/// 用户查询可能包含它们（例如 `"C++"`、`a*b`），不加引号会引发解析错误。
/// 双引号内的内容按字面量匹配，不会被当作语法。
pub fn tokenize(text: &str) -> String {
    let tokens = tokenize_raw(text);
    tokens
        .into_iter()
        .map(|token| format!("\"{token}\""))
        .collect::<Vec<_>>()
        .join(" ")
}

/// 切出 token 列表（不含引号），便于测试与调试。
pub fn tokenize_raw(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut ascii_buf = String::new();
    let mut cjk_buf: Vec<char> = Vec::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            // 英文/数字：与前面的中文分开，先结算中文缓冲
            flush_cjk(&mut cjk_buf, &mut tokens);
            ascii_buf.push(ch);
        } else if ch.is_whitespace() {
            // 空白：分隔符，结算两边
            flush_ascii(&mut ascii_buf, &mut tokens);
            flush_cjk(&mut cjk_buf, &mut tokens);
        } else if is_cjk(ch) {
            // 中日韩文字：进 CJK 缓冲做 bigram
            flush_ascii(&mut ascii_buf, &mut tokens);
            cjk_buf.push(ch);
        } else {
            // 其他符号（标点、emoji 等）：纯分隔符，直接丢弃。
            // 保留它们会让 "Rust，异步" 切出 "，异" 这种无意义的 bigram。
            flush_ascii(&mut ascii_buf, &mut tokens);
            flush_cjk(&mut cjk_buf, &mut tokens);
        }
    }

    flush_ascii(&mut ascii_buf, &mut tokens);
    flush_cjk(&mut cjk_buf, &mut tokens);

    tokens
}

/// 判断是不是需要做 bigram 切分的中日韩字符。
///
/// 覆盖 CJK 统一表意文字、中日韩标点、假名、韩文等。
/// 用范围判断而不是 `is_alphanumeric()`——后者对中文有效，
/// 但对全角标点（如 `，`）会返回 false，标点不该参与 bigram。
fn is_cjk(ch: char) -> bool {
    let code = ch as u32;
    matches!(code,
        0x3040..=0x309F      // 日文假名
        | 0x30A0..=0x30FF    // 日文标点
        | 0x3400..=0x4DBF    // CJK 扩展 A
        | 0x4E00..=0x9FFF    // CJK 统一表意文字
        | 0xF900..=0xFAFF    // CJK 兼容表意文字
        | 0xAC00..=0xD7AF    // 韩文音节
        | 0x20000..=0x2A6DF  // CJK 扩展 B
    )
}

/// 结算 ASCII 缓冲：一个连续的字母数字串算一个 token。
fn flush_ascii(buf: &mut String, out: &mut Vec<String>) {
    if !buf.is_empty() {
        out.push(buf.clone());
        buf.clear();
    }
}

/// 结算 CJK 缓冲：切成 bigram；单字则整体保留。
fn flush_cjk(buf: &mut Vec<char>, out: &mut Vec<String>) {
    if buf.len() == 1 {
        out.push(buf[0].to_string());
    } else {
        for window in buf.windows(2) {
            out.push(window.iter().collect());
        }
    }
    buf.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chinese_is_split_into_bigrams() {
        assert_eq!(tokenize_raw("面积公式"), vec!["面积", "积公", "公式"]);
    }

    #[test]
    fn single_chinese_character_is_kept() {
        assert_eq!(tokenize_raw("圆"), vec!["圆"]);
    }

    #[test]
    fn english_words_are_kept_whole() {
        assert_eq!(tokenize_raw("TCP"), vec!["TCP"]);
        assert_eq!(tokenize_raw("HTTP 2"), vec!["HTTP", "2"]);
    }

    #[test]
    fn mixed_chinese_and_english() {
        assert_eq!(
            tokenize_raw("TCP 是面向连接的"),
            vec!["TCP", "是面", "面向", "向连", "连接", "接的"]
        );
    }

    #[test]
    fn punctuation_is_a_separator() {
        assert_eq!(tokenize_raw("Rust，异步。"), vec!["Rust", "异步"]);
    }

    #[test]
    fn empty_text_yields_no_tokens() {
        assert!(tokenize_raw("").is_empty());
        assert!(tokenize_raw("   ").is_empty());
    }

    #[test]
    fn tokens_are_quoted_for_fts5_safety() {
        // 引号包裹后，FTS5 不会把 token 里的特殊字符当语法
        assert_eq!(tokenize("TCP"), "\"TCP\"");
        assert_eq!(tokenize("面积"), "\"面积\"");
    }

    #[test]
    fn punctuation_is_dropped_as_a_separator() {
        // 纯符号不会产生 token，避免 FTS5 收到无意义的查询
        assert!(tokenize_raw("!!!").is_empty());
        assert!(tokenize_raw("*").is_empty());
        // 字母数字之间的符号被切开
        assert_eq!(tokenize_raw("a*b"), vec!["a", "b"]);
    }

    #[test]
    fn query_and_document_share_the_same_tokenizer() {
        // 查询词与文档用同一个分词器，2 字中文查询才能命中
        let doc = tokenize("圆的面积公式是什么");
        let query = tokenize("面积");
        assert!(doc.contains(&query));
    }
}
