use super::*;
use puffer_resources::{AgentSpec, PromptTemplate};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const REVIEWER_IDS: &[&str] = &[
    "reviewer-correctness",
    "reviewer-security",
    "reviewer-performance",
    "reviewer-concurrency",
    "reviewer-error-handling",
    "reviewer-architecture",
    "reviewer-rust-idioms",
    "reviewer-testing",
    "reviewer-api-contract",
    "reviewer-null-safety",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn load_prompt(relative_path: &str) -> PromptTemplate {
    let contents = fs::read_to_string(repo_root().join(relative_path)).unwrap();
    serde_yaml::from_str(&contents).unwrap()
}

fn load_agent(relative_path: &str) -> AgentSpec {
    let contents = fs::read_to_string(repo_root().join(relative_path)).unwrap();
    serde_yaml::from_str(&contents).unwrap()
}

#[test]
fn ultrareview_command_registered_as_prompt_with_pr_hint() {
    let commands = supported_commands();
    let cmd = find_command(&commands, "ultrareview").expect("ultrareview command registered");
    assert_eq!(cmd.kind, CommandKind::Prompt);
    assert_eq!(cmd.argument_hint.as_deref(), Some("[pr-number]"));
    assert!(cmd.aliases.is_empty());
    assert!(!cmd.hidden);
}

#[test]
fn ultrareview_command_sits_between_theme_and_usage() {
    let commands = supported_commands();
    let names: Vec<&str> = commands.iter().map(|c| c.name.as_str()).collect();
    let theme_idx = names.iter().position(|n| *n == "theme").expect("theme present");
    let ultra_idx = names
        .iter()
        .position(|n| *n == "ultrareview")
        .expect("ultrareview present");
    let usage_idx = names.iter().position(|n| *n == "usage").expect("usage present");
    assert!(
        theme_idx < ultra_idx && ultra_idx < usage_idx,
        "expected theme < ultrareview < usage, got indices {theme_idx} {ultra_idx} {usage_idx}"
    );
}

#[test]
fn ultrareview_prompt_renders_with_pr_number() {
    let prompt = load_prompt("resources/prompts/ultrareview.yaml");
    let mut vars = BTreeMap::new();
    vars.insert("ARGUMENTS".to_string(), "1234".to_string());
    let rendered = prompt.render(&vars);
    assert!(
        rendered.contains("PR number (optional): 1234"),
        "expected PR number substitution in rendered prompt: {rendered}"
    );
    assert!(rendered.contains("gh pr diff 1234"));
    assert!(rendered
        .to_lowercase()
        .to_lowercase()
        .contains("specialized reviewer subagents"));
}

#[test]
fn ultrareview_prompt_renders_without_arguments() {
    let prompt = load_prompt("resources/prompts/ultrareview.yaml");
    let rendered = prompt.render(&BTreeMap::new());
    assert!(rendered.contains("PR number (optional):"));
    assert!(rendered.contains("git diff HEAD"));
}

#[test]
fn ultrareview_prompt_lists_all_reviewers() {
    let prompt = load_prompt("resources/prompts/ultrareview.yaml");
    let rendered = prompt.render(&BTreeMap::new());
    for reviewer in REVIEWER_IDS {
        assert!(
            rendered.contains(reviewer),
            "coordinator prompt missing reference to {reviewer}"
        );
    }
}

#[test]
fn ultrareview_reviewer_agents_all_load_with_expected_fields() {
    for reviewer in REVIEWER_IDS {
        let path = format!("resources/agents/{reviewer}.yaml");
        let spec = load_agent(&path);
        assert_eq!(spec.id, *reviewer, "id mismatch in {path}");
        assert!(
            !spec.description.is_empty(),
            "missing description in {path}"
        );
        assert!(!spec.prompt.is_empty(), "missing prompt in {path}");
        assert_eq!(
            spec.isolation.as_deref(),
            Some("worktree"),
            "{path} must set isolation: worktree so reviewers run in isolated worktrees"
        );
    }
}

#[test]
fn ultrareview_reviewer_agents_restrict_to_read_only_tools() {
    let expected_disallowed: &[&str] = &[
        "Agent",
        "Edit",
        "Write",
        "NotebookEdit",
        "ExitPlanMode",
    ];
    for reviewer in REVIEWER_IDS {
        let path = format!("resources/agents/{reviewer}.yaml");
        let spec = load_agent(&path);
        for forbidden in expected_disallowed {
            assert!(
                spec.disallowed_tools
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(forbidden)),
                "{path} must disallow {forbidden} to stay read-only"
            );
        }
        let allowed_names: Vec<&str> = spec.tools.iter().map(|s| s.as_str()).collect();
        assert!(
            allowed_names.iter().all(|t| matches!(
                *t,
                "Read" | "Glob" | "Grep" | "Bash"
            )),
            "{path} allowed_tools must be a subset of read-only tools, got {allowed_names:?}"
        );
    }
}

#[test]
fn ultrareview_reviewer_agents_declare_explicit_out_of_scope_section() {
    for reviewer in REVIEWER_IDS {
        let path = format!("resources/agents/{reviewer}.yaml");
        let spec = load_agent(&path);
        assert!(
            spec.prompt.contains("OUT OF SCOPE")
                || spec.prompt.contains("out of scope"),
            "{path} must contain an explicit out-of-scope section to avoid lane overlap"
        );
        assert!(
            spec.prompt.contains("BLOCKER")
                && spec.prompt.contains("SHOULD-FIX")
                && spec.prompt.contains("NIT"),
            "{path} must define the BLOCKER/SHOULD-FIX/NIT severity vocabulary"
        );
    }
}
