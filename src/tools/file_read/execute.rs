//! `read_file` 工具的执行逻辑。
//!
//! 本模块只负责「拿到参数 -> 真正读文件 -> 返回字符串」，不涉及 Tool trait 的
//! schema 定义（那部分在 `impl.rs`）以及工具注册（在 `src/tools.rs`）。
//!
//! 调用链：
//!
//! ```text
//! LLM 产出 tool_call(JSON)
//!        │
//!        ▼
//! Tool::execute(args_json)            // impl.rs：反序列化成 ReadFileArgs
//!        │
//!        ▼
//! read(&args)                        // 本文件入口：分派到具体读取实现
//!        ├── CSV  ──> read_csv_as_markdown()   // 转成 Markdown 表格
//!        └── 其它 ──> read_text_with_line_numbers() // 带行号的文本片段
//! ```

use std::{fs, path::Path};

use anyhow::Ok;

use crate::tools::file_read::r#impl::ReadFileArgs;

/// 读取文件的统一入口。
///
/// 根据文件扩展名做分派：
/// - `.csv`：交给 [`read_csv_as_markdown`]，输出 Markdown 表格；
/// - 其它：交给 [`read_text_with_line_numbers`]，输出带行号的纯文本。
///
/// 注意：本函数返回的 `Err` 会由调用方（`Tool::execute`）转换成
/// `"Error: {err}"` 字符串回灌给 LLM，因此这里的错误信息需要可读。
pub fn read(args: &ReadFileArgs) -> anyhow::Result<String> {
    // 把参数字符串转成 Path，方便后续做存在性判断与扩展名解析。
    let path = Path::new(&args.file_path);

    // 前置校验：文件不存在时直接 bail，避免后续 fs 调用抛出更底层的 OS 错误。
    if !path.exists() {
        anyhow::bail!("File not found: {}", args.file_path);
    }

    // 按扩展名分派。extension() 返回 Option<&OsStr>，再用 to_str() 转成
    // UTF-8 字符串；非 UTF-8 或没有扩展名时统一落到 `_` 分支按纯文本处理。
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => read_csv_as_markdown(path),
        _ => read_text_with_line_numbers(path, args.start_line, args.end_line),
    }
}

/// 读取文本文件，并按行号区间输出带行号的内容。
///
/// 行号语义：
/// - `start_line` 从 1 开始（与编辑器一致），内部换算成 0 基下标；
/// - `end_line` 为负数（默认 `-1`）表示「读到文件末尾」。
fn read_text_with_line_numbers(
    path: &Path,
    start_line: usize,
    end_line: i64,
) -> anyhow::Result<String> {
    // 一次性读入整个文件为 UTF-8 字符串；二进制/非 UTF-8 文件会在这里报错。
    let content = fs::read_to_string(path)?;
    // 按行切分成借用切片，避免为每行分配独立的 String。
    let lines: Vec<&str> = content.lines().collect();

    // 起始下标：1 基 -> 0 基。saturating_sub 保证 start_line == 0 时不会下溢。
    let start_index = start_line.saturating_sub(1);
    // 结束下标（不含）：负数表示到末尾；正数则钳制到文件行数以内。
    let end_index = if end_line < 0 {
        lines.len()
    } else {
        (end_line as usize).min(lines.len())
    };

    let mut result = String::new();
    // 取区间切片：若 start_index > end_index（例如起点超出文件长度），
    // get() 返回 None，这里用 unwrap_or(&[]) 兜底为空区间。
    for (offset, line) in lines
        .get(start_index..end_index)
        .unwrap_or(&[])
        .iter()
        .enumerate()
    {
        // 行号右对齐占 4 位，形如 `   1 | 内容`，便于 LLM 定位与后续再请求区间。
        result.push_str(&format!("{:>4} | {}\n", start_index + offset + 1, line));
    }

    Ok(result)
}

/// 读取 CSV 文件并转成 Markdown 表格字符串。
///
/// 先输出表头行与分隔行，再逐行输出数据行，方便 LLM 直接理解结构化数据。
fn read_csv_as_markdown(path: &Path) -> anyhow::Result<String> {
    // csv::Reader 会自动识别分隔符（逗号/制表符/竖线等）。
    let mut reader = csv::Reader::from_path(path)?;
    // headers() 返回借用 reader 内部的 CSVRecord，因此需要 clone() 取得拥有所有权的副本。
    let headers = reader.headers()?.clone();

    let mut table = String::new();
    // 表头行：`| col1 | col2 |`
    table.push_str(&format!(
        "| {} |\n",
        headers.iter().collect::<Vec<_>>().join(" | ")
    ));
    // 分隔行：`|---|---|`，列数与表头一致。
    table.push_str(&format!("|{}\n", "---|".repeat(headers.len())));

    // 逐条读取数据记录；记录条数与表头不一致时 csv crate 会给出可变长度记录。
    for record in reader.records() {
        // 单行解析失败（引号未闭合等）时向上返回错误。
        let record = record?;
        // 数据行：`| v1 | v2 |`
        table.push_str(&format!(
            "| {} |\n",
            record.iter().collect::<Vec<_>>().join(" | ")
        ));
    }

    Ok(table)
}
