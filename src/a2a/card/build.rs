//! 从项目 [`crate::agent::model::Agent`] 生成 A2A skill 列表的映射规则。
//!
//! 规则详见 [`crate::a2a::card::AgentCard::from_agent`] 的文档。
//! 这里只放实现，方便 `card.rs` 保持"数据结构 + 构造器"的纯粹。

use crate::{
    a2a::card::skill::AgentSkill,
    agent::model::Agent,
};

/// 由 agent 构造 skill 列表。
///
/// 映射顺序：
///
/// 1. 子 agent → 委派能力（tag 含 `delegation`）；
/// 2. 工具 → 具体能力（tag 为工具名）；
/// 3. 兜底：既无子节点也无工具时，给一条描述自身的 skill。
pub fn build_skills(agent: &Agent) -> Vec<AgentSkill> {
    let mut skills = Vec::new();

    // ---- 1. 子 agent → 委派能力 ----
    //
    // 子 agent 是这个 agent 最主要的"能力"：根助手自己可能什么都不会，
    // 但它能转交给 researcher / math / coder，这对远端客户端来说是
    // 最需要知道的信息——它决定了"该派什么活儿过来"。
    for sub in agent.sub_agents() {
        let mut tags: Vec<String> = vec!["delegation".to_owned()];
        tags.extend(sorted_tool_names(sub));

        let sub_description = if sub.description.trim().is_empty() {
            format!("将任务转交给 '{}' 子 agent 处理。", sub.name)
        } else {
            sub.description.clone()
        };

        skills.push(AgentSkill::new(
            format!("{}-{}", agent.name, sub.name),
            sub.name.clone(),
            sub_description,
            tags,
        ));
    }

    // ---- 2. 工具 → 具体能力 ----
    //
    // 每个工具就是一个可独立完成的动作，因此一条 skill。
    // 描述直接取工具自己的 description——那是写给人看的、能力级别的
    // 说明，比提示词合适得多。
    let mut tool_names: Vec<&String> = agent.toolbox.keys().collect();
    tool_names.sort();

    for tool_name in tool_names {
        let tool = &agent.toolbox[tool_name];
        let tool_description = tool.description().trim();

        // 描述为空时给个兜底，至少让客户端知道有这个东西。
        let description = if tool_description.is_empty() {
            format!("调用 '{}' 工具。", tool_name)
        } else {
            tool_description.to_owned()
        };

        skills.push(AgentSkill::new(
            format!("{}-tool-{}", agent.name, tool_name),
            tool_name.clone(),
            description,
            vec![tool_name.clone()],
        ));
    }

    // ---- 3. 兜底 ----
    //
    // 既没有子节点也没有工具时，这个 agent 没有可枚举的能力，
    // 至少给一条描述自身的 skill，别让卡片是空的。
    if skills.is_empty() {
        let fallback_description = agent
            .instructions
            .first()
            .and_then(|instruction| instruction.lines().next())
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{} agent", agent.name));

        skills.push(AgentSkill::new(
            agent.name.clone(),
            agent.name.clone(),
            fallback_description,
            Vec::new(),
        ));
    }

    skills
}

/// 取 agent 工具箱里的工具名，按字典序排列。
///
/// 工具箱是 `HashMap`，直接遍历顺序不确定；排序后卡片是稳定的。
fn sorted_tool_names(agent: &Agent) -> Vec<String> {
    let mut names: Vec<String> = agent.toolbox.keys().cloned().collect();
    names.sort();
    names
}
