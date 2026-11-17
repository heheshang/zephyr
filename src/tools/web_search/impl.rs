use schemars::schema_for;
use serde_json::Value;

use crate::{
    agent::context::ExecutionContext,
    tools::{
        tool::Tool,
        web_search::execute::{WebSearchArgs, web_search},
    },
};

pub struct WebSearchTool;

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current information on a given query."
    }

    fn parameters(&self) -> Value {
        serde_json::to_value(schema_for!(WebSearchArgs))
            .expect("Failed to serialize WebSearchArgs schema")
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &mut ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: WebSearchArgs = serde_json::from_str(args_json)?;
        match web_search(args).await {
            Ok(output) => Ok(serde_json::to_string(&output)?),
            Err(err) => Ok(format!("Error: {err}")),
        }
    }
}
