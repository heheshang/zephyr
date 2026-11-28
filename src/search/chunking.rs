/// 带位置信息的切片。
///
/// `start` 是切片首字符在原文中的**字符下标**（不是字节下标），
/// `text` 是切片内容。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    pub start: usize,
    pub text: String,
}

/// 固定长度切片，**保留每片在原文中的起始下标**。
///
/// 与 [`fixed_length_chunking`] 的区别只在于额外返回位置信息，
/// 切片算法本身完全一致（按字符计数，步长 = `chunk_size - overlap`）。
///
/// 知识库需要这个下标：检索命中后要能回到原文定位，
/// 也便于将来做"命中片段前后各扩 N 字符"的上下文补全。
pub fn chunking_with_offsets(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<ChunkSpan> {
    assert!(chunk_size > 0, "chunk_size must be greater than 0");
    assert!(overlap < chunk_size, "overlap must be less than chunk_size");

    let chars: Vec<char> = text.chars().collect();
    let chars_len = chars.len();
    let mut chunks = Vec::new();
    let step = chunk_size - overlap;

    let mut start = 0usize;
    while start < chars_len {
        let end = (start + chunk_size).min(chars_len);
        chunks.push(ChunkSpan {
            start,
            text: chars[start..end].iter().collect(),
        });
        if end == chars_len {
            break;
        }
        start += step;
    }

    chunks
}

/// 固定长度切片：按 `chunk_size` 切分，相邻两片重叠 `overlap` 个字符。
///
/// 步长为 `chunk_size - overlap`；重叠部分保证语义连续，
/// 避免答案正好被切在边界上。
///
/// 注意这里按**字符**（`char`）计数，不是字节，所以对中文是安全的。
pub fn fixed_length_chunking(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    chunking_with_offsets(text, chunk_size, overlap)
        .into_iter()
        .map(|span| span.text)
        .collect()
}

#[cfg(test)]
mod chunking_tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let result = fixed_length_chunking("abcdefghij", 5, 0);

        assert_eq!(result, vec!["abcde".to_string(), "fghij".to_string(),]);
    }

    #[test]
    fn test_chunking_with_overlap() {
        let result = fixed_length_chunking("abcdefghij", 5, 2);

        assert_eq!(
            result,
            vec!["abcde".to_string(), "defgh".to_string(), "ghij".to_string(),]
        );
    }

    #[test]
    fn test_final_chunk_is_shorter() {
        let result = fixed_length_chunking("abcdefgh", 5, 0);

        assert_eq!(result, vec!["abcde".to_string(), "fgh".to_string(),]);
    }

    #[test]
    fn test_chunk_size_larger_than_text() {
        let result = fixed_length_chunking("abc", 10, 0);

        assert_eq!(result, vec!["abc".to_string()]);
    }

    #[test]
    fn test_empty_text() {
        let result = fixed_length_chunking("", 5, 2);

        assert!(result.is_empty());
    }

    #[test]
    fn test_unicode_text() {
        let result = fixed_length_chunking("你好世界朋友", 3, 1);

        assert_eq!(
            result,
            vec![
                "你好世".to_string(),
                "世界朋".to_string(),
                "朋友".to_string(),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "chunk_size must be greater than 0")]
    fn test_zero_chunk_size() {
        fixed_length_chunking("abc", 0, 0);
    }

    #[test]
    #[should_panic(expected = "overlap must be less than chunk_size")]
    fn test_overlap_equal_to_chunk_size() {
        fixed_length_chunking("abc", 3, 3);
    }

    #[test]
    #[should_panic(expected = "overlap must be less than chunk_size")]
    fn test_overlap_greater_than_chunk_size() {
        fixed_length_chunking("abc", 3, 4);
    }

    #[test]
    fn chunking_with_offsets_reports_positions() {
        let spans = chunking_with_offsets("abcdefghij", 5, 2);

        assert_eq!(
            spans,
            vec![
                ChunkSpan {
                    start: 0,
                    text: "abcde".to_string()
                },
                ChunkSpan {
                    start: 3,
                    text: "defgh".to_string()
                },
                ChunkSpan {
                    start: 6,
                    text: "ghij".to_string()
                },
            ]
        );
    }

    #[test]
    fn chunking_with_offsets_matches_plain_chunking() {
        let text = "Rust 是一门系统编程语言，强调安全与并发。";
        let plain = fixed_length_chunking(text, 10, 3);
        let spans = chunking_with_offsets(text, 10, 3);

        assert_eq!(plain, spans.into_iter().map(|s| s.text).collect::<Vec<_>>());
    }

    #[test]
    fn chunking_with_offsets_slices_the_original_text() {
        let text = "abcdefghij";
        let spans = chunking_with_offsets(text, 5, 2);

        // 位置下标能正确切回原文
        for span in spans {
            let chars: Vec<char> = text.chars().collect();
            let end = (span.start + 5).min(chars.len());
            let expected: String = chars[span.start..end].iter().collect();
            assert_eq!(span.text, expected);
        }
    }
}
