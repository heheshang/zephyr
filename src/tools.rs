use std::{collections::HashMap, sync::Arc};

use crate::tools::{
    calculator::r#impl::CalculatorTool,
    file_delete::execute::DeleteFileTool,
    file_list::r#impl::ListFileTool,
    file_read::r#impl::ReadFileTool,
    file_unzip::r#impl::UnzipFileTool,
    mcp::{client::McpClient, tool::McpTool},
    read_image::r#impl::ReadImageTool,
    tool::Tool,
    web_search::r#impl::WebSearchTool,
};

pub mod calculator;
pub mod file_delete;
pub mod file_list;
pub mod file_read;
pub mod file_unzip;
pub mod mcp;
pub mod read_image;
pub mod tool;
pub mod web_search;

pub type ToolBox = HashMap<String, Box<dyn Tool>>;

pub async fn build_toolbox() -> Result<ToolBox, anyhow::Error> {
    let mut tools: Vec<Box<dyn Tool>> = vec![Box::new(CalculatorTool), Box::new(WebSearchTool)];
    let mcp_client = Arc::new(McpClient::connect().await?);

    for tool in mcp_client.list_tools().await? {
        tools.push(Box::new(McpTool::new(mcp_client.clone(), tool)));
    }

    Ok(tools
        .into_iter()
        .map(|t| (t.name().to_string(), t))
        .collect())
}

pub fn build_file_explorer_toolbox(vision_model: impl Into<String>) -> ToolBox {
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(UnzipFileTool),
        Box::new(ListFileTool),
        Box::new(ReadFileTool),
        Box::new(ReadImageTool::new(vision_model)),
        Box::new(DeleteFileTool),
    ];
    into_toolbox(tools)
}

fn into_toolbox(tools: Vec<Box<dyn Tool>>) -> ToolBox {
    tools
        .into_iter()
        .map(|t| (t.name().to_string(), t))
        .collect()
}
