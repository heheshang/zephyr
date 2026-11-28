//! Agent 树：多 agent 之间的父子关系与转移目标。
//!
//! # 为什么单独成模块
//!
//! agent 的**核心职责**是"想 → 做 → 看结果"的循环（见 `runtime.rs`）；
//! 而 agent **树**解决的是另一个问题：谁能把活儿交给谁。
//!
//! 两者混在一起会让 `model.rs` 同时承担"行为"和"拓扑"两种关注点，
//! 因此这里把拓扑相关的组装、校验、查询都收拢到这个模块。
//!
//! # 树的结构
//!
//! ```text
//!   root (assistant)
//!   ├── researcher
//!   ├── math
//!   └── coder
//! ```
//!
//! # 转移目标
//!
//! 一个 agent 允许转移到的目标有三类：
//!
//! 1. 子节点；
//! 2. 父节点；
//! 3. 兄弟节点（可通过 `disallow_transfer_to_peers` 关闭）。
//!
//! # 组装的两步走
//!
//! Python 可以在构造函数里直接给子节点设置 `parent`，但 Rust 必须先有
//! `Arc<Self>` 才能创建 `Weak<Self>`。因此树的组装被拆成两步：
//!
//! ```ignore
//! let math = Arc::new(Agent::new(...).with_name("math".into()));
//! let router = Arc::new(Agent::new(...).with_name("router".into()));
//! router.set_sub_agents(vec![math])?;
//! ```

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
};

use crate::agent::model::Agent;
impl Agent {
    /// 组装 agent 树：把 `sub_agents` 挂到当前 agent 下面。
    ///
    /// 校验规则与书中一致：同一父节点下名称不能重复，一个 agent 只能有一个
    /// 父节点。这里额外检查整棵树内的名称唯一性以及环形引用，因为这两点
    /// 会直接影响按名称转移和 `Arc` 的生命周期。
    pub fn set_sub_agents(self: &Arc<Self>, sub_agents: Vec<Arc<Agent>>) -> anyhow::Result<()> {
        if self.sub_agents.get().is_some() {
            anyhow::bail!("agent '{}' already has sub-agents", self.name);
        }

        let mut direct_names = HashSet::new();
        for sub in &sub_agents {
            if !direct_names.insert(sub.name.clone()) {
                anyhow::bail!("duplicate sub-agent name: '{}'", sub.name);
            }
            if Arc::ptr_eq(sub, self) {
                anyhow::bail!("agent '{}' cannot be its own sub-agent", self.name);
            }
        }

        // 一个 agent 不能有两个父节点。先整体检查，再统一写入，
        // 避免校验失败时留下"挂了一半"的树。
        for sub in &sub_agents {
            if sub.parent.get().is_some() {
                anyhow::bail!(
                    "agent '{}' already has a parent; an agent can only appear once in the tree",
                    sub.name
                );
            }
        }

        // 名称在整个树中必须唯一，否则 transfer_to_agent("name") 会有歧义。
        let mut root = self.clone();
        while let Some(parent) = root.parent() {
            root = parent;
        }
        let mut names = HashMap::new();
        root.collect_names(&mut names)?;
        for sub in &sub_agents {
            if sub.contains_agent(Arc::as_ptr(self)) {
                anyhow::bail!(
                    "cannot attach '{}' under '{}': it would create a cycle",
                    sub.name,
                    self.name
                );
            }
            sub.collect_names(&mut names)?;
        }

        self.sub_agents
            .set(sub_agents.clone())
            .map_err(|_| anyhow::anyhow!("agent '{}' already has sub-agents", self.name))?;

        let parent = Arc::downgrade(self);
        for sub in sub_agents {
            // 前面已经确认所有子节点都没有 parent，这里不会失败。
            let _ = sub.parent.set(parent.clone());
        }

        Ok(())
    }

    /// 当前节点的子节点列表；未组装时返回空切片。
    pub fn sub_agents(&self) -> &[Arc<Agent>] {
        self.sub_agents.get().map(Vec::as_slice).unwrap_or_default()
    }

    /// 当前节点的父节点。弱引用升级失败说明父节点已被释放。
    pub fn parent(&self) -> Option<Arc<Agent>> {
        self.parent.get().and_then(Weak::upgrade)
    }

    /// 当前 agent 允许转移到的全部目标。
    ///
    /// 三类目标：子节点、父节点、兄弟节点（可关闭）。
    pub fn transfer_targets(&self) -> Vec<Arc<Agent>> {
        let mut targets = self.sub_agents().to_vec();

        if let Some(parent) = self.parent() {
            targets.push(parent.clone());

            if !self.disallow_transfer_to_peers {
                for sibling in parent.sub_agents() {
                    if !std::ptr::eq(sibling.as_ref(), self) {
                        targets.push(sibling.clone());
                    }
                }
            }
        }

        // 理论上树结构已经保证不会重复；这里仍按名称去重，
        // 保证生成的 enum 中不会出现重复值。
        let mut seen = HashSet::new();
        targets.retain(|agent| seen.insert(agent.name.clone()));
        targets
    }

    /// 在**允许的转移目标**中按名字查找。
    ///
    /// 书中从根节点搜索整棵树。这里直接搜索转移目标列表，效果相同且更安全：
    /// 不在目标列表里的 agent 本来就不允许通过工具转移过去。
    pub fn find_transfer_target(&self, name: &str) -> Option<Arc<Agent>> {
        self.transfer_targets()
            .into_iter()
            .find(|agent| agent.name == name)
    }

    /// 递归收集这棵子树里的所有 agent 名字。
    fn collect_names(&self, names: &mut HashMap<String, usize>) -> anyhow::Result<()> {
        let ptr = self as *const Agent as usize;
        if let Some(existing) = names.get(&self.name) {
            if *existing == ptr {
                anyhow::bail!(
                    "agent '{}' appears more than once in the agent tree",
                    self.name
                );
            }
            anyhow::bail!("duplicate agent name '{}' in agent tree", self.name);
        }
        names.insert(self.name.clone(), ptr);

        for sub in self.sub_agents() {
            sub.collect_names(names)?;
        }
        Ok(())
    }

    /// 这棵子树里是否包含某个 agent。
    fn contains_agent(&self, target: *const Agent) -> bool {
        if std::ptr::eq(self, target) {
            return true;
        }
        self.sub_agents()
            .iter()
            .any(|sub| sub.contains_agent(target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolBox;

    fn agent(name: &str) -> Arc<Agent> {
        Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec![format!("You are {name}.")],
                Arc::new(ToolBox::new()),
            )
            .with_name(name.to_owned())
            .with_description(format!("{name} description")),
        )
    }

    fn names(agents: &[Arc<Agent>]) -> Vec<String> {
        agents.iter().map(|agent| agent.name.clone()).collect()
    }

    #[test]
    fn parent_and_siblings_are_transfer_targets() {
        let math = agent("math");
        let search = agent("search");
        let router = agent("router");
        router
            .set_sub_agents(vec![math.clone(), search.clone()])
            .unwrap();

        assert_eq!(names(&router.transfer_targets()), ["math", "search"]);
        assert_eq!(
            names(&math.transfer_targets()),
            ["router", "search"],
            "math should reach its parent and its sibling"
        );
        assert!(math.parent().is_some());
        assert!(math.find_transfer_target("nobody").is_none());
    }

    #[tokio::test]
    async fn llm_request_includes_the_dynamic_transfer_tool() {
        let math = agent("math");
        let search = agent("search");
        let router = agent("router");
        router
            .set_sub_agents(vec![math.clone(), search.clone()])
            .unwrap();

        let mut context = crate::agent::context::ExecutionContext::default();
        let request = router.prepare_llm_request(&mut context).await.unwrap();
        let tools = serde_json::to_string(&request.tools).unwrap();
        assert!(tools.contains("transfer_to_agent"));
        assert!(tools.contains("\"math\""));
        assert!(tools.contains("\"search\""));

        // 叶子节点同样会得到父节点/兄弟节点的转移工具。
        let request = math.prepare_llm_request(&mut context).await.unwrap();
        let tools = serde_json::to_string(&request.tools).unwrap();
        assert!(tools.contains("\"router\""));
        assert!(tools.contains("\"search\""));
    }

    #[test]
    fn disallow_transfer_to_peers_removes_siblings() {
        let math = Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec!["You are math.".to_owned()],
                Arc::new(ToolBox::new()),
            )
            .with_name("math".to_owned())
            .with_disallow_transfer_to_peers(true),
        );
        let search = agent("search");
        let router = agent("router");
        router.set_sub_agents(vec![math.clone(), search]).unwrap();

        assert_eq!(names(&math.transfer_targets()), ["router"]);
    }

    #[test]
    fn an_agent_cannot_have_two_parents() {
        let math = agent("math");
        let router = agent("router");
        let other = agent("other");
        router.set_sub_agents(vec![math.clone()]).unwrap();

        assert!(other.set_sub_agents(vec![math]).is_err());
    }

    #[test]
    fn duplicate_names_across_branches_are_rejected() {
        let left = agent("left");
        let right = agent("right");
        let root = agent("root");
        root.set_sub_agents(vec![left.clone(), right.clone()])
            .unwrap();

        left.set_sub_agents(vec![agent("duplicate")]).unwrap();
        assert!(right.set_sub_agents(vec![agent("duplicate")]).is_err());
    }

    #[test]
    fn cycles_are_rejected() {
        let child = agent("child");
        let root = agent("root");
        root.set_sub_agents(vec![child.clone()]).unwrap();

        assert!(child.set_sub_agents(vec![root]).is_err());
    }
}
