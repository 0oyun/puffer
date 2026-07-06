//! Agent-facing browser actions layered over the managed Chrome sessions.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::daemon::{DaemonState, ServerEnvelope};

use super::dom_inspect::dom_inspect_expression;
use super::params::{optional_u32, required_string, required_string_array};
use super::ref_resolution::{
    fill_expression, focus_expression, hosted_fill_focus_check_expression,
    hosted_fill_point_expression, in_frame_prepare_fill_fn, in_frame_readback_fn,
    in_frame_select_fn, in_frame_set_checked_fn, main_doc_focus_clear_expression,
    main_doc_readback_expression, scroll_into_view_expression, select_expression,
    set_checkable_state_expression, target_point_expression, upload_input_handle_expression,
};
use super::screenshot::{parse_agent_screenshot_options, BrowserElementRef};
use super::session::BrowserSession;
use super::tabs::{backend_session_id, BrowserTabInfo, BrowserTabsState};
use super::{
    browser_debug, BrowserHistoryDirection, BrowserInputEvent, BrowserRegistry, DEFAULT_URL,
    INITIAL_HEIGHT, INITIAL_WIDTH,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserCheckableState {
    kind: String,
    checked: bool,
}

/// Handles `browser_agent`, the agent-oriented browser action endpoint.
pub(crate) fn handle_browser_agent(state: &Arc<DaemonState>, params: &Value) -> Result<Value> {
    let action = required_string(params, "action")?;
    let root_session_id = required_string(params, "sessionId")?;
    let width = optional_u32(params, "width").unwrap_or(INITIAL_WIDTH);
    let height = optional_u32(params, "height").unwrap_or(INITIAL_HEIGHT);
    browser_debug(
        "agent.request",
        format!(
            "action={} root_session_id={} tab_id={:?} width={} height={}",
            action,
            root_session_id,
            optional_string(params, "tabId"),
            width,
            height
        ),
    );
    match action.as_str() {
        "list" => {
            // Pull in any tab the user opened directly in the native browser so
            // the agent sees what the user is actually looking at (#649).
            state.browsers.sync_native_tabs(
                &state.event_sender(),
                &root_session_id,
                width,
                height,
            );
            Ok(serde_json::to_value(list_tabs_with_cli_fallback(
                state,
                &root_session_id,
            ))?)
        }
        "open" => {
            let tab_id =
                resolve_open_target_tab_id(&state.browsers, &root_session_id, params, true);
            arm_agent_recording(state, &root_session_id, &tab_id);
            let tab = open_agent_tab(
                state,
                &root_session_id,
                params,
                width,
                height,
                true,
                Some(tab_id),
            )?;
            state.browsers.arm_agent_recording(&tab.backend_session_id);
            publish_tabs(state, &root_session_id);
            Ok(serde_json::to_value(tab)?)
        }
        "new" => {
            let tab_id =
                resolve_open_target_tab_id(&state.browsers, &root_session_id, params, false);
            arm_agent_recording(state, &root_session_id, &tab_id);
            let tab = open_agent_tab(
                state,
                &root_session_id,
                params,
                width,
                height,
                false,
                Some(tab_id),
            )?;
            state.browsers.arm_agent_recording(&tab.backend_session_id);
            publish_tabs(state, &root_session_id);
            Ok(serde_json::to_value(tab)?)
        }
        "focus" => {
            let tab_id = required_string(params, "tabId")?;
            let tab = state.browsers.focus_tab(&root_session_id, &tab_id)?;
            publish_tabs(state, &root_session_id);
            Ok(serde_json::to_value(tab)?)
        }
        "close" => {
            let tab_id = optional_string(params, "tabId")
                .or_else(|| {
                    active_or_first(&state.browsers.list_tabs(&root_session_id))
                        .map(|tab| tab.tab_id)
                })
                .with_context(|| format!("no browser tabs for session `{root_session_id}`"))?;
            browser_debug(
                "agent.close.resolved",
                format!("root_session_id={} tab_id={}", root_session_id, tab_id),
            );
            let tabs = state.browsers.close_tab(&root_session_id, &tab_id)?;
            publish_tabs(state, &root_session_id);
            Ok(serde_json::to_value(tabs)?)
        }
        "quit" | "exit" => {
            state.browsers.close_root(&root_session_id)?;
            let tabs = state.browsers.list_tabs(&root_session_id);
            publish_tabs(state, &root_session_id);
            Ok(serde_json::to_value(tabs)?)
        }
        "navigate" => {
            let url = required_string(params, "url")?;
            let (tab_id, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            state.browsers.navigate(&backend_id, url)?;
            state.browsers.focus_tab(&root_session_id, &tab_id)?;
            publish_tabs(state, &root_session_id);
            let mut result = serde_json::to_value(state.browsers.list_tabs(&root_session_id))?;
            // Autonomously clear a reCAPTCHA the navigation landed on, so the agent
            // continues its task without the model having to handle the challenge.
            #[cfg(feature = "captcha-audio")]
            if let Some(outcome) = state.browsers.auto_solve_if_captcha(&backend_id) {
                if let Some(obj) = result.as_object_mut() {
                    obj.insert("captchaSolve".to_string(), outcome);
                }
            }
            Ok(result)
        }
        "reload" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            state.browsers.reload(&backend_id)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), true))
        }
        "back" | "forward" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let direction = if action == "back" {
                BrowserHistoryDirection::Back
            } else {
                BrowserHistoryDirection::Forward
            };
            state.browsers.history(&backend_id, direction)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), true))
        }
        "snapshot" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            state.browsers.agent_snapshot(&backend_id)
        }
        "domInspect" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let query = required_string(params, "query")?;
            Ok(state
                .browsers
                .get(&backend_id)?
                .evaluate(dom_inspect_expression(&query)?)?
                .value)
        }
        "consoleLogs" | "console" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            let clear = params
                .get("clear")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            state.browsers.console_logs(&backend_id, clear)
        }
        "waitNetworkIdle" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.wait_for_network_idle(
                &backend_id,
                network_idle_duration(params),
                navigation_timeout(params),
            )?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "screenshot" => {
            let (tab_id, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let options = parse_agent_screenshot_options(params)?;
            state
                .browsers
                .agent_screenshot(&backend_id, &tab_id, options)
        }
        "openScreenshot" => {
            let url = required_string(params, "url")?;
            let (tab_id, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            state.browsers.navigate(&backend_id, url)?;
            state.browsers.focus_tab(&root_session_id, &tab_id)?;
            publish_tabs(state, &root_session_id);
            state
                .browsers
                .wait_for_load(&backend_id, navigation_timeout(params))?;
            let options = parse_agent_screenshot_options(params)?;
            state
                .browsers
                .agent_screenshot(&backend_id, &tab_id, options)
        }
        "openConsoleLogs" => {
            let url = required_string(params, "url")?;
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            let _ = state.browsers.console_logs(&backend_id, true);
            state.browsers.arm_agent_recording(&backend_id);
            state.browsers.navigate(&backend_id, url)?;
            state
                .browsers
                .wait_for_load(&backend_id, navigation_timeout(params))?;
            let clear = params
                .get("clear")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            state.browsers.console_logs(&backend_id, clear)
        }
        "click" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state.browsers.agent_click(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), true))
        }
        // Coordinate-based click. Unlike `click` (which needs a snapshot `ref`),
        // this dispatches a real mouse click at viewport pixel (x, y); the
        // browser's hit-testing routes it into whatever is under the point —
        // including cross-origin iframes (reCAPTCHA checkbox / challenge tiles)
        // that the DOM snapshot cannot surface as refs. The move is split into a
        // few steps with small delays so it reads as human pointer motion rather
        // than an instantaneous teleport.
        "clickAt" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let x = coord_param(params, "x")
                .ok_or_else(|| anyhow::anyhow!("clickAt requires numeric `x`"))?;
            let y = coord_param(params, "y")
                .ok_or_else(|| anyhow::anyhow!("clickAt requires numeric `y`"))?;
            let click_count = optional_u32(params, "count").unwrap_or(1).max(1);
            state
                .browsers
                .agent_click_at(&backend_id, x, y, click_count)?;
            Ok(state.browsers.post_action_snapshot(
                &backend_id,
                json!({ "ok": true, "x": x, "y": y }),
                true,
            ))
        }
        // Move the pointer to (x, y) without clicking — useful to build up
        // human-like pointer telemetry before a click, which behavioral
        // anti-bot scorers (reCAPTCHA v3 / checkbox) weigh.
        "moveMouse" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let x = coord_param(params, "x")
                .ok_or_else(|| anyhow::anyhow!("moveMouse requires numeric `x`"))?;
            let y = coord_param(params, "y")
                .ok_or_else(|| anyhow::anyhow!("moveMouse requires numeric `y`"))?;
            state.browsers.agent_move_mouse(&backend_id, x, y)?;
            Ok(state
                .browsers
                .post_action_snapshot(&backend_id, json!({ "ok": true, "x": x, "y": y }), false))
        }
        // Autonomously solve a reCAPTCHA v2 on the page via its audio challenge +
        // a local whisper model (feature `captcha-audio`). `count` = max rounds.
        "solveCaptcha" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            #[cfg(feature = "captcha-audio")]
            {
                let max_rounds = optional_u32(params, "count").unwrap_or(3).max(1);
                let outcome = state.browsers.agent_solve_captcha(&backend_id, max_rounds)?;
                Ok(state
                    .browsers
                    .post_action_snapshot(&backend_id, outcome, true))
            }
            #[cfg(not(feature = "captcha-audio"))]
            {
                let _ = &backend_id;
                bail!("solveCaptcha requires building puffer with the `captcha-audio` feature")
            }
        }
        "dblclick" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state.browsers.agent_double_click(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), true))
        }
        "hover" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state.browsers.agent_hover(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "focus_ref" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state.browsers.agent_focus(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "type" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            if let Some(target) = optional_string(params, "ref") {
                state.browsers.agent_click(&backend_id, &target)?;
                thread::sleep(Duration::from_millis(40));
            }
            let text = required_string(params, "text")?;
            state
                .browsers
                .input(&backend_id, BrowserInputEvent::Text { text })?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "insertText" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let text = required_string(params, "text")?;
            state
                .browsers
                .input(&backend_id, BrowserInputEvent::Text { text })?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "fill" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            let text = required_string(params, "text")?;
            let outcome = state.browsers.agent_fill(&backend_id, &target, &text)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, outcome, false))
        }
        "select" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            let value = required_string(params, "value")?;
            state.browsers.agent_select(&backend_id, &target, &value)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "upload" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            let files = required_string_array(params, "files")?;
            state.browsers.agent_upload(&backend_id, &target, files)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "check" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state.browsers.agent_check(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "uncheck" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state.browsers.agent_uncheck(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "press" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let key = required_string(params, "key")?;
            state.browsers.agent_press(&backend_id, &key)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), true))
        }
        "keydown" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let key = required_string(params, "key")?;
            state.browsers.agent_key_down(&backend_id, &key)?;
            Ok(json!({ "ok": true }))
        }
        "keyup" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let key = required_string(params, "key")?;
            state.browsers.agent_key_up(&backend_id, &key)?;
            Ok(json!({ "ok": true }))
        }
        "scroll" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let direction = required_string(params, "direction")?;
            let px = optional_u32(params, "px").unwrap_or(600);
            state.browsers.agent_scroll(&backend_id, &direction, px)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "scrollIntoView" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let target = required_string(params, "ref")?;
            state
                .browsers
                .agent_scroll_into_view(&backend_id, &target)?;
            Ok(state.browsers.post_action_snapshot(&backend_id, json!({ "ok": true }), false))
        }
        "evaluate" | "eval" => {
            let (_, backend_id) =
                ensure_target_tab(state, &root_session_id, params, width, height)?;
            state.browsers.arm_agent_recording(&backend_id);
            let script = required_string(params, "script")?;
            let value = state.browsers.get(&backend_id)?.evaluate(script)?.value;
            Ok(json!({ "value": value }))
        }
        other => bail!("unsupported browser agent action `{other}`"),
    }
}

fn arm_agent_recording(state: &Arc<DaemonState>, root_session_id: &str, tab_id: &str) {
    state
        .browsers
        .arm_agent_recording(&backend_session_id(root_session_id, tab_id));
}

fn open_agent_tab(
    state: &Arc<DaemonState>,
    root_session_id: &str,
    params: &Value,
    width: u32,
    height: u32,
    reuse_existing: bool,
    resolved_tab_id: Option<String>,
) -> Result<BrowserTabInfo> {
    let background = background_requested(params);
    let activate = params
        .get("activate")
        .and_then(Value::as_bool)
        .unwrap_or(!background);
    if let Some(tab_id) = resolved_tab_id.or_else(|| optional_string(params, "tabId")) {
        return state.browsers.open_tab(
            state.event_sender(),
            root_session_id.to_string(),
            Some(tab_id),
            optional_string(params, "label"),
            optional_string(params, "url"),
            width,
            height,
            activate,
            background,
        );
    }
    if reuse_existing {
        if let Some(tab) = active_or_first(&state.browsers.list_tabs(root_session_id)) {
            return state.browsers.open_tab(
                state.event_sender(),
                root_session_id.to_string(),
                Some(tab.tab_id),
                optional_string(params, "label"),
                optional_string(params, "url"),
                width,
                height,
                activate,
                background,
            );
        }
    }
    state.browsers.open_tab(
        state.event_sender(),
        root_session_id.to_string(),
        None,
        optional_string(params, "label"),
        optional_string(params, "url"),
        width,
        height,
        activate,
        background,
    )
}

fn resolve_open_target_tab_id(
    browsers: &BrowserRegistry,
    root_session_id: &str,
    params: &Value,
    reuse_existing: bool,
) -> String {
    if let Some(tab_id) = optional_string(params, "tabId") {
        return tab_id;
    }
    if reuse_existing {
        if let Some(tab) = active_or_first(&browsers.list_tabs(root_session_id)) {
            return tab.tab_id;
        }
    }
    browsers.tabs.lock().unwrap().next_tab_id(root_session_id)
}

impl BrowserRegistry {
    /// Clicks an element ref from the last agent snapshot.
    pub(crate) fn agent_click(&self, backend_session_id: &str, ref_id: &str) -> Result<()> {
        let session = self.get(backend_session_id)?;
        let (x, y) = self.agent_target_point(backend_session_id, ref_id)?;
        dispatch_agent_mouse_click(&session, x, y, 1)?;
        Ok(())
    }

    /// Double-clicks an element ref from the last agent snapshot.
    pub(crate) fn agent_double_click(&self, backend_session_id: &str, ref_id: &str) -> Result<()> {
        let session = self.get(backend_session_id)?;
        let (x, y) = self.agent_target_point(backend_session_id, ref_id)?;
        dispatch_agent_mouse_click(&session, x, y, 1)?;
        dispatch_agent_mouse_click(&session, x, y, 2)?;
        Ok(())
    }

    /// Moves the pointer over an element ref from the last agent snapshot.
    pub(crate) fn agent_hover(&self, backend_session_id: &str, ref_id: &str) -> Result<()> {
        let session = self.get(backend_session_id)?;
        let (x, y) = self.agent_target_point(backend_session_id, ref_id)?;
        session.input(BrowserInputEvent::Mouse {
            event_type: "mouseMoved".to_string(),
            x,
            y,
            button: "none".to_string(),
            buttons: Some(0),
            click_count: 0,
        })?;
        Ok(())
    }

    /// Clicks at a raw viewport pixel coordinate (no snapshot ref). The
    /// browser's hit-testing routes the click into whatever is under the point,
    /// including cross-origin iframes that the DOM snapshot cannot expose as
    /// refs (e.g. the reCAPTCHA checkbox or challenge tiles).
    pub(crate) fn agent_click_at(
        &self,
        backend_session_id: &str,
        x: f64,
        y: f64,
        click_count: u32,
    ) -> Result<()> {
        let session = self.get(backend_session_id)?;
        dispatch_humanized_move(&session, x, y)?;
        thread::sleep(Duration::from_millis(45));
        for c in 1..=click_count.max(1) {
            session.input(BrowserInputEvent::Mouse {
                event_type: "mousePressed".to_string(),
                x,
                y,
                button: "left".to_string(),
                buttons: Some(1),
                click_count: c,
            })?;
            thread::sleep(Duration::from_millis(55));
            session.input(BrowserInputEvent::Mouse {
                event_type: "mouseReleased".to_string(),
                x,
                y,
                button: "left".to_string(),
                buttons: Some(0),
                click_count: c,
            })?;
        }
        Ok(())
    }

    /// Moves the pointer to a raw viewport pixel coordinate with human-like
    /// multi-step motion (no click). Useful to build pointer telemetry that
    /// behavioral anti-bot scorers weigh before a click.
    pub(crate) fn agent_move_mouse(&self, backend_session_id: &str, x: f64, y: f64) -> Result<()> {
        let session = self.get(backend_session_id)?;
        dispatch_humanized_move(&session, x, y)?;
        Ok(())
    }

    /// Autonomously solves a reCAPTCHA v2 challenge via its AUDIO option using a
    /// local whisper model (feature `captcha-audio`). Works because Puffer runs
    /// Chrome with site isolation disabled (`--disable-features=site-per-process`),
    /// so the cross-origin reСAPTCHA frames share the renderer and we can run JS in
    /// them via `Page.createIsolatedWorld`. The challenge mp3 is fetched inside the
    /// (google-origin) bframe and returned as base64, so the daemon needs no proxy.
    /// Returns `{ solved, rounds?, note?, log }`. Never throws on a normal miss —
    /// a `try again later` block returns `solved:false` so the caller can move on.
    #[cfg(feature = "captcha-audio")]
    /// ONE solve attempt: opens the challenge and solves it via audio (Whisper), or
    /// via the CLIP/CLIPSeg image path when `force_image` is set. Returns a `note`
    /// the orchestrator reads to decide retries — "audio-blocked" (doscaptcha "try
    /// again later") tells it to REFRESH THE PAGE and try again; after enough audio
    /// blocks the orchestrator forces the image path.
    fn solve_captcha_attempt(
        &self,
        backend_session_id: &str,
        max_rounds: u32,
        force_image: bool,
    ) -> Result<Value> {
        use base64::Engine as _;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let session = self.get(backend_session_id)?;
        // Page domain must be enabled so the (cross-site, same-process) reCAPTCHA
        // frames register in the frame registry — otherwise createIsolatedWorld
        // rejects their id with "No frame for given id found".
        let _ = session.cdp_call("Page.enable", json!({}));
        thread::sleep(Duration::from_millis(300));
        let mut log: Vec<String> = Vec::new();
        let checked_js = "document.querySelector('#recaptcha-anchor')?.getAttribute('aria-checked')";
        // The bframe (challenge) frame is pre-created and empty; it "has content"
        // only once a challenge actually opens. Used to tell an opened challenge
        // from a stale empty frame.
        let bframe_has_content = "!!(document.querySelector('#recaptcha-audio-button')||document.querySelector('table td')||document.querySelector('.rc-doscaptcha-header,.rc-doscaptcha-body'))";

        // The reCAPTCHA frames attach a few seconds AFTER navigation returns (the
        // api.js loads from google.com), so poll rather than sampling once too early
        // — otherwise an explicit solve right after a goto reports a false "no
        // recaptcha".
        let (mut anchor, mut bframe) = (None, None);
        for _ in 0..16 {
            let (a, b) = recaptcha_frame_ids(&session)?;
            if a.is_some() || b.is_some() {
                anchor = a;
                bframe = b;
                break;
            }
            thread::sleep(Duration::from_millis(500));
        }
        if anchor.is_none() && bframe.is_none() {
            return Ok(json!({ "solved": false, "note": "no recaptcha", "log": log, "debug": recaptcha_debug(&session) }));
        }
        if std::env::var("PUFFER_CAPTCHA_DEBUG").is_ok() {
            let ftree = session.cdp_call("Page.getFrameTree", json!({})).ok();
            let create_try = anchor.as_ref().or(bframe.as_ref()).map(|f| {
                match session.cdp_call(
                    "Page.createIsolatedWorld",
                    json!({ "frameId": f, "worldName": "dbg" }),
                ) {
                    Ok(v) => format!("OK:{v}"),
                    Err(e) => format!("ERR:{e:#}"),
                }
            });
            return Ok(json!({
                "debug": recaptcha_debug(&session),
                "anchor": anchor, "bframe": bframe,
                "createWorldTry": create_try,
                "frameTree": ftree,
            }));
        }
        if let Some(a) = &anchor {
            if eval_in_frame(&session, a, checked_js)?.as_str() == Some("true") {
                return Ok(json!({ "solved": true, "note": "already solved", "log": log }));
            }
        }
        // Open the challenge with a TRUSTED coordinate click on the checkbox
        // (behavioral scoring weighs this), computing coords from the anchor
        // iframe's rect in the top document. The bframe frame is PRE-created and
        // empty before any click, so treat a contentless bframe as "not opened"
        // and still click the checkbox — otherwise Puffer acts on a blank frame.
        let bframe_ready = match &bframe {
            Some(b) => eval_in_frame(&session, b, bframe_has_content)?.as_bool() == Some(true),
            None => false,
        };
        if !bframe_ready {
            bframe = None;
            // Click the checkbox and CONFIRM it took — the anchor's bframe frame
            // exists before any click, so "a bframe id exists" is not proof the
            // challenge opened. Retry the click (it sometimes misses) until either
            // the box is checked (silent pass) or the bframe actually has content.
            'checkbox: for attempt in 0..3u32 {
                let coords = session
                    .evaluate(
                        "(()=>{const f=[...document.querySelectorAll('iframe')].find(i=>(i.src||'').includes('api2/anchor'));if(!f)return null;const r=f.getBoundingClientRect();return {x:r.left+30,y:r.top+r.height/2};})()"
                            .to_string(),
                    )
                    .ok()
                    .map(|e| e.value);
                if let Some(obj) = coords.as_ref().and_then(Value::as_object) {
                    let x = obj.get("x").and_then(Value::as_f64).unwrap_or(0.0);
                    let y = obj.get("y").and_then(Value::as_f64).unwrap_or(0.0);
                    if x > 0.0 && y > 0.0 {
                        let _ = self.agent_click_at(backend_session_id, x, y, 1);
                        log.push(format!("click-checkbox{}", attempt + 1));
                    }
                }
                // human pause after clicking, then watch for the outcome
                thread::sleep(Duration::from_millis(rng.gen_range(900..1700)));
                for _ in 0..10 {
                    let (a2, b2) = recaptcha_frame_ids(&session)?;
                    if let Some(a) = &a2 {
                        if eval_in_frame(&session, a, checked_js)?.as_str() == Some("true") {
                            return Ok(json!({ "solved": true, "note": "silent pass", "log": log }));
                        }
                    }
                    if let Some(b) = &b2 {
                        if eval_in_frame(&session, b, bframe_has_content)?.as_bool() == Some(true) {
                            bframe = Some(b.clone());
                            break 'checkbox;
                        }
                    }
                    thread::sleep(Duration::from_millis(500));
                }
                log.push("checkbox-missed".to_string());
                thread::sleep(Duration::from_millis(rng.gen_range(500..1100)));
            }
        }
        let Some(bframe) = bframe else {
            return Ok(json!({ "solved": false, "note": "no challenge frame", "log": log }));
        };
        // The bframe exists early but empty — wait for the challenge UI (audio button,
        // image grid, or the doscaptcha block) to actually render before acting, so we
        // don't read a blank frame and give up.
        for _ in 0..16 {
            let ready = eval_in_frame(
                &session,
                &bframe,
                "!!(document.querySelector('#recaptcha-audio-button')||document.querySelector('table td')||document.querySelector('.rc-doscaptcha-header,.rc-doscaptcha-body'))",
            )
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
            if ready {
                break;
            }
            thread::sleep(Duration::from_millis(500));
        }
        // IMAGE path: taken when the orchestrator forces it (audio blocked earlier /
        // on cooldown) or PUFFER_CAPTCHA_IMAGE_ONLY is set. It never clicks the audio
        // button, so it can't trip the "try again later" doscaptcha. CLIP is
        // open-vocab (3×3) and CLIPSeg segments the 4×4 area grids.
        let image_only = std::env::var_os("PUFFER_CAPTCHA_IMAGE_ONLY").is_some();
        #[cfg(feature = "captcha-image")]
        if image_only || force_image {
            let task0 = eval_in_frame(
                &session,
                &bframe,
                "(()=>{const s=document.querySelector('.rc-imageselect-desc strong,strong');return s?s.innerText:'';})()",
            )
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();
            log.push(format!("img0:{task0}"));
            // Generous round budget so a legit multi-step challenge chain (each
            // Verify reveals another grid) can be followed to completion instead of
            // being abandoned half-solved (which reads as a bot). Real reloads are
            // separately capped, and verify_fails only accrues on real rejections.
            let (ok, ilog) = self.solve_image_challenge(backend_session_id, &bframe, &anchor, max_rounds.max(18));
            log.extend(ilog);
            return Ok(json!({
                "solved": ok,
                "note": if ok { "solved via image" } else { "image: unsolved" },
                "log": log,
            }));
        }
        // AUDIO attempt (single). On a doscaptcha block we return note "audio-blocked"
        // so the orchestrator can REFRESH THE PAGE (fresh widget) and retry; the image
        // fallback is the orchestrator's decision after N audio blocks, not inline.
        let ab = eval_in_frame(
            &session,
            &bframe,
            "(()=>{const b=document.querySelector('#recaptcha-audio-button');if(!b)return 'no-audio-btn';b.click();return 'clicked-audio';})()",
        )
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "eval-err".to_string());
        log.push(format!("audiobtn:{ab}"));
        // human pause while the audio panel loads / as if listening
        thread::sleep(Duration::from_millis(rng.gen_range(1500..2400)));

        for round in 0..max_rounds.max(1) {
            if eval_in_frame(
                &session,
                &bframe,
                "!!document.querySelector('.rc-doscaptcha-header,.rc-doscaptcha-body')",
            )?
            .as_bool()
                == Some(true)
            {
                log.push("blocked".to_string());
                // Audio is rate-limited ("try again later"). The whole widget is now
                // poisoned (image in it is blocked too), so the ONLY recovery is a
                // fresh page. Hand back "audio-blocked" — the orchestrator refreshes
                // the page and retries (audio again, or image after enough blocks).
                return Ok(json!({ "solved": false, "note": "audio-blocked", "log": log }));
            }
            // Fetch the challenge mp3 from inside the (google-origin) bframe → base64.
            let mut b64 = String::new();
            for _ in 0..8 {
                let v = eval_in_frame(
                    &session,
                    &bframe,
                    "(async()=>{const a=document.querySelector('.rc-audiochallenge-tdownload-link');if(!a)return '';try{const r=await fetch(a.href);const u=new Uint8Array(await r.arrayBuffer());let s='';for(let i=0;i<u.length;i++)s+=String.fromCharCode(u[i]);return btoa(s);}catch(e){return '';}})()",
                )?;
                if let Some(s) = v.as_str() {
                    if !s.is_empty() {
                        b64 = s.to_string();
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(700));
            }
            if b64.is_empty() {
                let state = eval_in_frame(
                    &session,
                    &bframe,
                    "JSON.stringify({dl:!!document.querySelector('.rc-audiochallenge-tdownload-link'),audio:!!document.querySelector('#audio-source,audio'),playbtn:!!document.querySelector('.rc-audiochallenge-play-button,button.rc-button-default'),instr:(document.querySelector('.rc-audiochallenge-instructions')||{}).innerText||'',err:(document.querySelector('.rc-audiochallenge-error-message')||{}).innerText||'',body:(document.body?document.body.innerText:'').slice(0,120)})",
                )
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
                log.push(format!("no-audio {state}"));
                break;
            }
            let mp3 = base64::engine::general_purpose::STANDARD
                .decode(b64.as_bytes())
                .unwrap_or_default();
            let text = match super::captcha_audio::transcribe(&mp3) {
                Ok(t) => t,
                Err(error) => {
                    log.push(format!("asr-err:{error:#}"));
                    break;
                }
            };
            log.push(format!("asr:{}", text.chars().take(40).collect::<String>()));
            if text.trim().is_empty() {
                let _ = eval_in_frame(
                    &session,
                    &bframe,
                    "(()=>{const r=document.querySelector('#recaptcha-reload-button');if(r)r.click();})()",
                );
                thread::sleep(Duration::from_millis(2500));
                continue;
            }
            // Human-like: type the answer, pause as if reviewing it, THEN submit —
            // filling and clicking verify in the same instant reads as a bot.
            let fill_js = format!(
                "(()=>{{const i=document.querySelector('#audio-response');if(!i)return 'no-input';i.focus();i.value={};i.dispatchEvent(new Event('input',{{bubbles:true}}));return 'filled';}})()",
                serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".to_string())
            );
            let _ = eval_in_frame(&session, &bframe, &fill_js);
            thread::sleep(Duration::from_millis(rng.gen_range(800..1600)));
            let _ = eval_in_frame(
                &session,
                &bframe,
                "(()=>{const v=document.querySelector('#recaptcha-verify-button');if(v)v.click();})()",
            );
            thread::sleep(Duration::from_millis(rng.gen_range(2800..3600)));
            if let Some(a) = &anchor {
                if eval_in_frame(&session, a, checked_js)?.as_str() == Some("true") {
                    log.push("PASS".to_string());
                    return Ok(json!({ "solved": true, "rounds": round + 1, "log": log }));
                }
            }
            if let Ok(tok) = session.evaluate(
                "(()=>{try{return (grecaptcha&&grecaptcha.getResponse&&grecaptcha.getResponse())||''}catch(e){return ''}})()"
                    .to_string(),
            ) {
                if tok.value.as_str().map(|s| !s.is_empty()).unwrap_or(false) {
                    log.push("PASS-token".to_string());
                    return Ok(json!({ "solved": true, "rounds": round + 1, "log": log }));
                }
            }
            log.push(format!("retry{}", round + 1));
            thread::sleep(Duration::from_millis(1200));
        }
        // Audio was served but we couldn't solve it (no mp3 / empty ASR / wrong
        // answers). Report "audio-failed" so the orchestrator refreshes + moves on.
        Ok(json!({ "solved": false, "note": "audio-failed", "log": log }))
    }

    /// Orchestrates the autonomous solve: try AUDIO first, and on a "try again
    /// later" doscaptcha block REFRESH THE PAGE and retry — up to `PUFFER_CAPTCHA_
    /// AUDIO_ATTEMPTS` (default 2) audio tries — then fall back to the IMAGE solver
    /// on the next fresh page. Also skips audio entirely for a cooldown window after
    /// a recent block (so repeated navigations don't keep re-tripping doscaptcha),
    /// and honours PUFFER_CAPTCHA_IMAGE_ONLY.
    #[cfg(feature = "captcha-audio")]
    pub(crate) fn agent_solve_captcha(&self, backend_session_id: &str, max_rounds: u32) -> Result<Value> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_AUDIO_BLOCK_MS: AtomicU64 = AtomicU64::new(0);
        let now_ms = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        };
        let cooldown_ms = std::env::var("PUFFER_CAPTCHA_AUDIO_COOLDOWN_S")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(480)
            * 1000;
        let max_audio = std::env::var("PUFFER_CAPTCHA_AUDIO_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(2);
        let image_only = std::env::var_os("PUFFER_CAPTCHA_IMAGE_ONLY").is_some();
        let mut audio_fails = 0u32;
        let mut agg: Vec<String> = Vec::new();
        // audio attempts (each may refresh) + up to 2 image attempts after.
        for attempt in 0..(max_audio + 2) {
            let cooling = now_ms().saturating_sub(LAST_AUDIO_BLOCK_MS.load(Ordering::Relaxed)) < cooldown_ms;
            let force_image = image_only || cooling || audio_fails >= max_audio;
            if attempt > 0 {
                agg.push(format!("--attempt{}{}", attempt + 1, if force_image { ":image" } else { ":audio" }));
            }
            let r = self.solve_captcha_attempt(backend_session_id, max_rounds, force_image)?;
            if let Some(l) = r.get("log").and_then(Value::as_array) {
                agg.extend(l.iter().filter_map(|v| v.as_str().map(String::from)));
            }
            if r.get("solved").and_then(Value::as_bool) == Some(true) {
                let mut out = r.clone();
                out["log"] = json!(agg);
                return Ok(out);
            }
            let note = r.get("note").and_then(Value::as_str).unwrap_or("");
            // Terminal states — nothing to retry.
            if matches!(note, "no recaptcha" | "already solved" | "silent pass" | "no challenge frame") {
                let mut out = r.clone();
                out["log"] = json!(agg);
                return Ok(out);
            }
            // The image attempt already retries internally (reloads, multi-step,
            // wall-clock deadline). Don't re-run it — that just multiplies the time
            // budget and re-solves fresh grids pointlessly. One image attempt is final.
            if force_image {
                let mut out = r.clone();
                out["log"] = json!(agg);
                return Ok(out);
            }
            // Both a doscaptcha block and a served-but-unsolved audio round count as
            // an audio failure; after `max_audio` of them the next attempt forces
            // image. Only a real block starts the skip-audio cooldown.
            if note == "audio-blocked" || note == "audio-failed" {
                audio_fails += 1;
            }
            if note == "audio-blocked" {
                LAST_AUDIO_BLOCK_MS.store(now_ms(), Ordering::Relaxed);
            }
            // Audio blocked or failed, or image attempt failed → refresh the page for
            // a clean widget and try again (next attempt forces image once audio is
            // exhausted / cooling).
            self.reload_page_for_captcha(backend_session_id);
        }
        Ok(json!({ "solved": false, "note": "exhausted", "log": agg }))
    }

    /// Refresh the page and wait for a fresh reCAPTCHA widget to attach — the only
    /// way to recover from a doscaptcha-poisoned widget.
    #[cfg(feature = "captcha-audio")]
    fn reload_page_for_captcha(&self, backend_session_id: &str) {
        let Ok(session) = self.get(backend_session_id) else {
            return;
        };
        let _ = session.cdp_call("Page.reload", json!({ "ignoreCache": false }));
        // Give the reload + reСAPTCHA script time to re-attach the frames.
        for _ in 0..20 {
            thread::sleep(Duration::from_millis(500));
            if let Ok((anchor, bframe)) = recaptcha_frame_ids(&session) {
                if anchor.is_some() || bframe.is_some() {
                    thread::sleep(Duration::from_millis(800));
                    return;
                }
            }
        }
    }

    /// IMAGE-challenge fallback (feature `captcha-image`): when the audio option is
    /// blocked, solve the visible grid with the native YOLO detector. Reads the
    /// grid geometry + task from the bframe, screenshots just the grid, detects the
    /// matching tiles, clicks them (humanized) and verifies — with human-like dwell
    /// throughout. Reloads on an empty detection, retries up to `max_rounds`.
    #[cfg(feature = "captcha-image")]
    fn solve_image_challenge(
        &self,
        backend_session_id: &str,
        bframe: &str,
        _anchor: &Option<String>,
        max_rounds: u32,
    ) -> (bool, Vec<String>) {
        use base64::Engine as _;
        use rand::Rng;
        let mut log: Vec<String> = Vec::new();
        let Ok(session) = self.get(backend_session_id) else {
            return (false, log);
        };
        let mut rng = rand::thread_rng();
        // Control-loop state across rounds. `verify_fails` bounds pointless
        // re-verifies of a rejected grid; whether to Verify is decided per-grid from
        // its own selection state (any_selected), not a cross-grid flag.
        let mut verify_fails = 0u32;
        // Cap reloads: hammering "new challenge" is the single biggest bot tell and
        // burns the session's reputation (→ doscaptcha block). A human reloads a
        // couple of times at most.
        let mut reloads = 0u32;
        let max_reloads = 3u32;
        // Cap the multi-step chain. A legit reCAPTCHA challenge resolves in 1–3
        // steps; an endless stream of new challenges that never accepts is the
        // low-trust "doscaptcha-lite" trap — grinding it just wastes time and is
        // itself a bot tell (a human never solves 8 in a row). Give up past this.
        let mut next_steps = 0u32;
        let max_next_steps = std::env::var("PUFFER_CLIP_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5u32);
        // Wall-clock deadline so a hard multi-step/dynamic chain returns with its log
        // instead of being killed by the caller's outer timeout (which loses the log
        // and reads as a hang). Env-tunable.
        let deadline = std::time::Instant::now()
            + Duration::from_secs(
                std::env::var("PUFFER_CLIP_DEADLINE_S").ok().and_then(|v| v.parse().ok()).unwrap_or(90),
            );
        for round in 0..max_rounds.max(1) {
            if std::time::Instant::now() >= deadline {
                log.push("img:time-cap".to_string());
                break;
            }
            // SETTLE FIRST: before reading the task/geometry or clicking anything,
            // wait until the grid is fully rendered — every tile image loaded AND the
            // src list unchanged across two reads. Doing this at the TOP means the
            // task, geometry and clicks below all act on ONE stable challenge, so a
            // click can't land on a grid still transitioning to the next challenge
            // (cross-challenge mis-touch, "上下题误触").
            {
                let mut prev_grid: Option<String> = None;
                for _ in 0..24 {
                    let st = eval_in_frame(
                        &session,
                        bframe,
                        "(()=>{const im=[...document.querySelectorAll('table td img')];const allc=im.length?im.every(i=>i.complete&&i.naturalWidth>0):false;return JSON.stringify({allc:allc,s:im.map(i=>i.src).join('|')});})()",
                    )
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .unwrap_or(Value::Null);
                    let all_loaded = st.get("allc").and_then(Value::as_bool).unwrap_or(false);
                    let cur_grid = st.get("s").and_then(Value::as_str).map(String::from);
                    if all_loaded && cur_grid.is_some() && cur_grid == prev_grid {
                        break;
                    }
                    if cur_grid.is_some() {
                        prev_grid = cur_grid;
                    }
                    thread::sleep(Duration::from_millis(170));
                }
            }
            // Grid geometry + tile count + task word + per-tile SELECTED state.
            // The td "…selected" class is reCAPTCHA's own record of what we've
            // already picked — the source of truth that unifies one-shot (tiles
            // stay selected → verify when all matches selected) and dynamic
            // (clicked tiles fade to a NEW image, unselected → keep clicking).
            let meta_raw = eval_in_frame(
                &session,
                bframe,
                "(()=>{const t=document.querySelector('table');if(!t)return '';const r=t.getBoundingClientRect();const tds=[...t.querySelectorAll('td')];const isSel=(td)=>{const c=td.className||'';if(/selected/i.test(c))return true;if(td.getAttribute('aria-pressed')==='true')return true;const cm=td.querySelector('.rc-imageselect-checkmark');if(cm){const st=getComputedStyle(cm);if(parseFloat(st.opacity||'0')>0.1&&st.display!=='none'&&st.visibility!=='hidden')return true;}return false;};const sel=tds.map(td=>isSel(td)?1:0);const s=document.querySelector('.rc-imageselect-desc strong,strong');const dbg=tds.slice(0,3).map(td=>td.className+'|ap='+td.getAttribute('aria-pressed')+'|cm='+((td.querySelector('.rc-imageselect-checkmark')||{}).style?getComputedStyle(td.querySelector('.rc-imageselect-checkmark')).opacity:'none'));return JSON.stringify({x:r.left,y:r.top,w:r.width,h:r.height,cells:tds.length,task:s?s.innerText:'',sel:sel,dbg:dbg});})()",
            )
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();
            let meta: Value = serde_json::from_str(&meta_raw).unwrap_or(Value::Null);
            let task = meta.get("task").and_then(Value::as_str).unwrap_or("").trim().to_string();
            let cells = meta.get("cells").and_then(Value::as_u64).unwrap_or(0) as usize;
            if task.is_empty() || cells == 0 {
                log.push("img:no-grid".to_string());
                break;
            }
            if !super::captcha_clip::can_handle(&task) {
                log.push(format!("img:cant-handle:{task}"));
                break;
            }
            let rows = if cells >= 16 { 4 } else { 3 };
            // bframe iframe position in the top document → absolute grid rect.
            let bf = session
                .evaluate(
                    "(()=>{const f=[...document.querySelectorAll('iframe')].find(i=>(i.src||'').includes('api2/bframe'));if(!f)return null;const r=f.getBoundingClientRect();return {x:r.left,y:r.top};})()"
                        .to_string(),
                )
                .ok()
                .map(|e| e.value)
                .unwrap_or(Value::Null);
            let gx = bf.get("x").and_then(Value::as_f64).unwrap_or(0.0) + meta.get("x").and_then(Value::as_f64).unwrap_or(0.0);
            let gy = bf.get("y").and_then(Value::as_f64).unwrap_or(0.0) + meta.get("y").and_then(Value::as_f64).unwrap_or(0.0);
            let gw = meta.get("w").and_then(Value::as_f64).unwrap_or(0.0);
            let gh = meta.get("h").and_then(Value::as_f64).unwrap_or(0.0);
            if gw < 20.0 || gh < 20.0 {
                log.push("img:bad-rect".to_string());
                break;
            }
            // (Grid already settled at the top of the round.) Screenshot just the grid.
            let png = session
                .cdp_call(
                    "Page.captureScreenshot",
                    json!({ "format": "png", "clip": { "x": gx, "y": gy, "width": gw, "height": gh, "scale": 1 } }),
                )
                .ok()
                .and_then(|v| v.get("data").and_then(Value::as_str).map(String::from))
                .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()).ok())
                .unwrap_or_default();
            if png.is_empty() {
                log.push("img:no-shot".to_string());
                break;
            }
            // Geometry diagnostics: dump the computed grid rect + a full-viewport
            // screenshot so a blank/mis-clipped grid can be traced (the clipped
            // grid PNG itself is saved by captcha_clip when PUFFER_CLIP_DEBUG set).
            if std::env::var_os("PUFFER_CLIP_DEBUG").is_some() {
                if let Some(dir) = std::env::var_os("HOME")
                    .map(|h| std::path::PathBuf::from(h).join(".puffer").join("clip_debug"))
                {
                    let _ = std::fs::create_dir_all(&dir);
                    let full = session
                        .cdp_call("Page.captureScreenshot", json!({ "format": "png" }))
                        .ok()
                        .and_then(|v| v.get("data").and_then(Value::as_str).map(String::from))
                        .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()).ok())
                        .unwrap_or_default();
                    if !full.is_empty() {
                        let _ = std::fs::write(dir.join("full_viewport.png"), &full);
                    }
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("geom.log")) {
                        use std::io::Write as _;
                        let _ = writeln!(
                            f,
                            "task={task} bf={} meta_x={} meta_y={} gx={gx:.1} gy={gy:.1} gw={gw:.1} gh={gh:.1} png_bytes={}",
                            bf,
                            meta.get("x").and_then(Value::as_f64).unwrap_or(-1.0),
                            meta.get("y").and_then(Value::as_f64).unwrap_or(-1.0),
                            png.len()
                        );
                    }
                }
            }
            // 4×4 "area" challenges (one photo sliced into 16) are segmented with
            // CLIPSeg — per-tile CLIP is weak when the object spans tiles. 3×3
            // classification stays on the CLIP per-tile path. PUFFER_CLIPSEG=0
            // forces the CLIP path for A/B during calibration.
            let use_seg = rows == 4 && std::env::var("PUFFER_CLIPSEG").as_deref() != Ok("0");
            let tiles = if use_seg {
                match super::captcha_clipseg::segment_tiles(&png, &task, rows) {
                    Ok(t) => t,
                    Err(e) => {
                        log.push(format!("img:clipseg-err:{e:#}"));
                        break;
                    }
                }
            } else {
                match super::captcha_clip::classify_tiles(&png, &task, rows) {
                    Ok(t) => t,
                    Err(e) => {
                        log.push(format!("img:detect-err:{e:#}"));
                        break;
                    }
                }
            };
            // Matched tiles not already selected (these need clicking).
            let sel: Vec<bool> = meta
                .get("sel")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) != 0).collect())
                .unwrap_or_default();
            let unselected: Vec<usize> = tiles
                .iter()
                .copied()
                .filter(|&i| sel.get(i).copied() != Some(true))
                .collect();
            log.push(format!("img:{task}:{}m/{}new/{} sel={}", tiles.len(), unselected.len(), cells,
                sel.iter().map(|b| if *b {'1'} else {'0'}).collect::<String>()));
            if std::env::var_os("PUFFER_CLIP_DEBUG").is_some() {
                if let Some(dbg) = meta.get("dbg") {
                    log.push(format!("tdcls:{dbg}"));
                }
            }

            if !unselected.is_empty() {
                for i in &unselected {
                    let (r, c) = (i / rows, i % rows);
                    let cx = gx + (c as f64 + 0.5) * gw / rows as f64;
                    let cy = gy + (r as f64 + 0.5) * gh / rows as f64;
                    let _ = self.agent_click_at(backend_session_id, cx, cy, 1);
                    thread::sleep(Duration::from_millis(rng.gen_range(40..110)));
                }
                verify_fails = 0;
                // Dynamic challenges REPLACE clicked tiles with new images that fade
                // in at STAGGERED times. Re-detecting too early reads a half-updated
                // grid and misaligns picks. Wait until the grid STABILIZES — all
                // images loaded AND the src list unchanged between two reads. This
                // handles full / partial / zero replacement and count mismatches
                // uniformly (unlike a "changed >= nclick" count, which never fires
                // when reCAPTCHA replaces fewer tiles than were clicked).
                let start_ms = std::env::var("PUFFER_CLIP_DYN_START").ok().and_then(|v| v.parse().ok()).unwrap_or(450u64);
                let settle_ms = std::env::var("PUFFER_CLIP_DYN_SETTLE").ok().and_then(|v| v.parse().ok()).unwrap_or(350u64);
                let max_wait_ms = std::env::var("PUFFER_CLIP_DYN_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(4500u64);
                thread::sleep(Duration::from_millis(start_ms));
                let mut elapsed = start_ms;
                let mut prev: Option<Vec<String>> = None;
                loop {
                    let st = eval_in_frame(
                        &session,
                        bframe,
                        "(()=>{const im=[...document.querySelectorAll('table td img')];const allc=im.length?im.every(i=>i.complete&&i.naturalWidth>0):false;return JSON.stringify({s:im.map(i=>i.src),allc:allc});})()",
                    )
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .unwrap_or(Value::Null);
                    let all_loaded = st.get("allc").and_then(Value::as_bool).unwrap_or(false);
                    let cur: Vec<String> = st
                        .get("s")
                        .and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    // Stable = loaded + a real (non-empty) read that matches the prior
                    // read. Requires ≥1 good read, so an eval hiccup (empty) can't
                    // short-circuit the wait.
                    let stable = all_loaded && !cur.is_empty() && prev.as_ref() == Some(&cur);
                    if stable && elapsed >= start_ms + 150 {
                        break;
                    }
                    if elapsed >= max_wait_ms {
                        break;
                    }
                    if !cur.is_empty() {
                        prev = Some(cur);
                    }
                    thread::sleep(Duration::from_millis(200));
                    elapsed += 200;
                }
                thread::sleep(Duration::from_millis(settle_ms));
                continue;
            }

            // No unselected matches remain on the CURRENT grid.
            let any_selected = sel.iter().any(|&b| b);
            if !any_selected {
                // Nothing selected here — a detection miss / hard grid, or a fresh
                // challenge that just replaced a verified one. Pressing Verify now
                // would submit an EMPTY answer (marked wrong), so reload a fresh grid
                // instead (bounded). Fixes "verify fired before the next challenge
                // was solved".
                if reloads >= max_reloads {
                    log.push("img:give-up-reloads".to_string());
                    break;
                }
                reloads += 1;
                log.push(format!("img:reload-empty{reloads}"));
                let _ = eval_in_frame(&session, bframe, "(()=>{const r=document.querySelector('#recaptcha-reload-button');if(r)r.click();})()");
                thread::sleep(Duration::from_millis(rng.gen_range(1100..1700)));
                continue;
            }
            // A real selection is complete → submit ONCE. Snapshot {task, img srcs}
            // to classify the outcome afterward.
            let pre_verify = eval_in_frame(
                &session,
                bframe,
                "(()=>{const s=document.querySelector('.rc-imageselect-desc strong,strong');const im=[...document.querySelectorAll('table td img')].map(i=>i.src);return JSON.stringify({t:s?s.innerText:'',im:im});})()",
            )
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .unwrap_or(Value::Null);
            thread::sleep(Duration::from_millis(rng.gen_range(220..480)));
            let _ = eval_in_frame(&session, bframe, "(()=>{const v=document.querySelector('#recaptcha-verify-button');if(v)v.click();})()");
            // Poll for the outcome. Priorities: (1) the TOP-document response token is
            // set ⇒ SOLVED (works even when `anchor` is None — the critical fix); (2)
            // the "incorrect response" banner shows ⇒ REJECTED; (3) the grid actually
            // changed (task or a NON-EMPTY img list differs) ⇒ advanced to the next
            // step. Waiting here also stops a premature second Verify.
            let mut advanced = false;
            let mut rejected = false;
            for _ in 0..16 {
                thread::sleep(Duration::from_millis(200));
                if recaptcha_solved(&session) {
                    log.push("img:PASS".to_string());
                    return (true, log);
                }
                let cur = eval_in_frame(
                    &session,
                    bframe,
                    "(()=>{const s=document.querySelector('.rc-imageselect-desc strong,strong');const im=[...document.querySelectorAll('table td img')].map(i=>i.src);const e=document.querySelector('.rc-imageselect-incorrect-response');const errVis=e?(getComputedStyle(e).display!=='none'&&getComputedStyle(e).visibility!=='hidden'&&parseFloat(getComputedStyle(e).opacity||'1')>0.1):false;return JSON.stringify({t:s?s.innerText:'',im:im,err:errVis});})()",
                )
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .unwrap_or(Value::Null);
                if cur.get("err").and_then(Value::as_bool) == Some(true) {
                    rejected = true;
                    break;
                }
                let im_nonempty = cur.get("im").and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false);
                let task_changed = cur.get("t") != pre_verify.get("t");
                let imgs_changed = cur.get("im") != pre_verify.get("im");
                if im_nonempty && (task_changed || imgs_changed) {
                    advanced = true;
                    break;
                }
            }
            if advanced && !rejected {
                // Legit multi-step chain — solve the next grid without counting a
                // failure or giving up (fixes "puffer stops while challenges keep
                // coming"). But bail out of an ENDLESS chain (low-trust trap).
                next_steps += 1;
                if next_steps > max_next_steps {
                    log.push(format!("img:chain-too-long({next_steps})"));
                    break;
                }
                log.push("img:next-step".to_string());
                verify_fails = 0;
                thread::sleep(Duration::from_millis(rng.gen_range(250..500)));
                continue;
            }
            // Rejected (banner) or unchanged grid → the selection was wrong.
            verify_fails += 1;
            log.push(format!("img:verify-fail{verify_fails}(r{})", round + 1));
            if verify_fails >= 2 {
                if reloads >= max_reloads {
                    log.push("img:give-up-reloads".to_string());
                    break;
                }
                reloads += 1;
                let _ = eval_in_frame(&session, bframe, "(()=>{const r=document.querySelector('#recaptcha-reload-button');if(r)r.click();})()");
                thread::sleep(Duration::from_millis(rng.gen_range(1100..1700)));
                verify_fails = 0;
            } else {
                thread::sleep(Duration::from_millis(rng.gen_range(450..800)));
            }
        }
        (false, log)
    }

    /// Autonomous hook: if a reCAPTCHA is present on the page, solve it and return
    /// the outcome; otherwise `None`. Called after navigation so Puffer clears a
    /// captcha on its own before continuing the task (silent to the model).
    #[cfg(feature = "captcha-audio")]
    pub(crate) fn auto_solve_if_captcha(&self, backend_session_id: &str) -> Option<Value> {
        // Escape hatch: PUFFER_CAPTCHA_NO_AUTO=1 lets a human open the challenge
        // (click the checkbox) so Puffer only solves the shown grid — used to test
        // the image solver without Puffer's own CDP checkbox click tripping the block.
        if std::env::var_os("PUFFER_CAPTCHA_NO_AUTO").is_some() {
            return None;
        }
        let session = self.get(backend_session_id).ok()?;
        // The reCAPTCHA frames attach a few seconds AFTER navigation returns, so
        // poll the frame tree (up to ~8s) rather than sampling once too early.
        let mut present = false;
        for _ in 0..16 {
            thread::sleep(Duration::from_millis(500));
            if let Ok((anchor, bframe)) = recaptcha_frame_ids(&session) {
                if anchor.is_some() || bframe.is_some() {
                    present = true;
                    break;
                }
            }
        }
        if !present {
            return None;
        }
        self.agent_solve_captcha(backend_session_id, 3).ok()
    }

    /// Focuses an element ref from the last agent snapshot.
    pub(crate) fn agent_focus(&self, backend_session_id: &str, ref_id: &str) -> Result<()> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        let session = self.get(backend_session_id)?;
        if let Some(backend_node_id) = in_frame_backend_node_id(&target) {
            session.cdp_call("DOM.focus", json!({ "backendNodeId": backend_node_id }))?;
            return Ok(());
        }
        session.evaluate(focus_expression(&target)?)?;
        Ok(())
    }

    /// Fills an input-like element ref from the last agent snapshot.
    ///
    /// Editable controls in the top document are filled directly. When the
    /// ref resolves to a hosted payment field — a cross-origin iframe (e.g.
    /// Shopify/Stripe PCI card fields) whose real `<input>` the top document
    /// cannot reach — the fill switches to trusted input: focus the frame
    /// with a real mouse click, select existing content with a triple
    /// click, and commit the text through `Input.insertText`, which the
    /// browser routes to the focused frame.
    pub(crate) fn agent_fill(
        &self,
        backend_session_id: &str,
        ref_id: &str,
        text: &str,
    ) -> Result<Value> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        let session = self.get(backend_session_id)?;
        if target.in_frame {
            return in_frame_fill(&session, &target, text);
        }
        // Fast path: the native value-setter + dispatched input/change events
        // fill most fields. `fill_expression` reads the value back synchronously
        // and throws "value did not stick" when the set was ignored. A
        // keystroke-guarded MAIN-DOCUMENT input (e.g. Olive Young #cardNo)
        // rejects any value not produced by genuine per-character keystrokes, so
        // it either throws here or briefly accepts and reverts. In both cases
        // fall back to a per-character trusted-keystroke fill (#675).
        match session.evaluate(fill_expression(&target, text)?) {
            Ok(evaluation) => {
                let outcome = evaluation.value;
                let hosted = outcome
                    .get("hostedFrameFill")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if hosted {
                    let x = outcome
                        .get("x")
                        .and_then(Value::as_f64)
                        .context("hosted frame fill point missing x")?;
                    let y = outcome
                        .get("y")
                        .and_then(Value::as_f64)
                        .context("hosted frame fill point missing y")?;
                    return hosted_frame_fill(&session, x, y, text);
                }
                // The synchronous in-page readback passed, but a keystroke-guarded
                // input can accept the programmatic value and revert it a tick
                // later. Re-read after a beat; if it reverted to empty, fall back
                // to the keystroke path rather than reporting a false success.
                if !text.is_empty() {
                    thread::sleep(Duration::from_millis(120));
                    if main_doc_field_is_empty(&session, &target)? {
                        return main_doc_keystroke_fill(&session, &target, text);
                    }
                }
                Ok(json!({ "ok": true }))
            }
            Err(err) if is_value_did_not_stick(&err) => {
                main_doc_keystroke_fill(&session, &target, text)
            }
            Err(err) => Err(err),
        }
    }

    /// Selects one option in a native `<select>` ref from the last agent snapshot.
    pub(crate) fn agent_select(
        &self,
        backend_session_id: &str,
        ref_id: &str,
        value: &str,
    ) -> Result<()> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        let session = self.get(backend_session_id)?;
        if target.in_frame {
            return in_frame_select(&session, &target, value);
        }
        session.evaluate(select_expression(&target, value)?)?;
        Ok(())
    }

    /// Uploads one or more files into a native file input ref from the last agent snapshot.
    pub(crate) fn agent_upload(
        &self,
        backend_session_id: &str,
        ref_id: &str,
        files: Vec<String>,
    ) -> Result<()> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        let session = self.get(backend_session_id)?;
        let expression = upload_input_handle_expression(&target)?;
        session.upload(expression, files)
    }

    /// Checks one checkbox-like ref from the last agent snapshot.
    pub(crate) fn agent_check(&self, backend_session_id: &str, ref_id: &str) -> Result<()> {
        self.set_checkable(backend_session_id, ref_id, true)
    }

    /// Unchecks one checkbox-like ref from the last agent snapshot.
    pub(crate) fn agent_uncheck(&self, backend_session_id: &str, ref_id: &str) -> Result<()> {
        self.set_checkable(backend_session_id, ref_id, false)
    }

    fn set_checkable(&self, backend_session_id: &str, ref_id: &str, checked: bool) -> Result<()> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        let session = self.get(backend_session_id)?;
        if let Some(backend_node_id) = in_frame_backend_node_id(&target) {
            let object_id = resolve_in_frame_object_id(&session, backend_node_id)?;
            session.cdp_call(
                "Runtime.callFunctionOn",
                json!({
                    "objectId": object_id,
                    "functionDeclaration": in_frame_set_checked_fn(checked),
                    "returnByValue": true,
                }),
            )?;
            return Ok(());
        }
        session.evaluate(set_checkable_state_expression(&target, checked)?)?;
        Ok(())
    }

    /// Presses one keyboard key in the target browser tab.
    pub(crate) fn agent_press(&self, backend_session_id: &str, key: &str) -> Result<()> {
        self.agent_key_down(backend_session_id, key)?;
        self.agent_key_up(backend_session_id, key)
    }

    /// Holds one keyboard key down in the target browser tab.
    pub(crate) fn agent_key_down(&self, backend_session_id: &str, key: &str) -> Result<()> {
        let combo = parse_key_combo(key);
        let code = key_code(&combo.key);
        self.get(backend_session_id)?.input(BrowserInputEvent::Key {
            event_type: "rawKeyDown".to_string(),
            key: combo.key.clone(),
            code,
            text: key_text(&combo.key).filter(|_| combo.modifiers == 0),
            modifiers: combo.modifiers,
            commands: combo.commands,
        })
    }

    /// Releases one keyboard key in the target browser tab.
    pub(crate) fn agent_key_up(&self, backend_session_id: &str, key: &str) -> Result<()> {
        let combo = parse_key_combo(key);
        let code = key_code(&combo.key);
        self.get(backend_session_id)?.input(BrowserInputEvent::Key {
            event_type: "keyUp".to_string(),
            key: combo.key,
            code,
            text: None,
            modifiers: combo.modifiers,
            commands: Vec::new(),
        })
    }

    /// Scrolls the target tab by a fixed amount in one direction.
    pub(crate) fn agent_scroll(
        &self,
        backend_session_id: &str,
        direction: &str,
        px: u32,
    ) -> Result<()> {
        let (delta_x, delta_y) = scroll_delta(direction, px)?;
        let session = self.get(backend_session_id)?;
        let state = session.state();
        session.input(BrowserInputEvent::Wheel {
            x: f64::from(state.width.max(1)) / 2.0,
            y: f64::from(state.height.max(1)) / 2.0,
            delta_x,
            delta_y,
        })
    }

    /// Scrolls an element ref into view from the last agent snapshot.
    pub(crate) fn agent_scroll_into_view(
        &self,
        backend_session_id: &str,
        ref_id: &str,
    ) -> Result<()> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        // An in-frame ref already carries a captured top-viewport quad center, so
        // it was rendered and in view at snapshot time; the top-document scroll
        // helper can't reach it. Treat scroll-into-view as a no-op.
        if target.in_frame {
            return Ok(());
        }
        self.get(backend_session_id)?
            .evaluate(scroll_into_view_expression(&target)?)?;
        Ok(())
    }

    fn lookup_ref(&self, backend_session_id: &str, ref_id: &str) -> Result<BrowserElementRef> {
        self.agent_refs
            .lock()
            .unwrap()
            .get(backend_session_id)
            .and_then(|refs| refs.iter().find(|item| item.ref_id == ref_id).cloned())
            .with_context(|| format!("no browser ref `{ref_id}`; run snapshot again"))
    }

    fn agent_target_point(&self, backend_session_id: &str, ref_id: &str) -> Result<(f64, f64)> {
        let target = self.lookup_ref(backend_session_id, ref_id)?;
        if target.in_frame {
            // The top document can't re-resolve a cross-origin node, so trust the
            // field's quad center captured at snapshot time. A coordinate click
            // there is routed into the OOPIF by browser-side hit testing.
            if target.x.is_finite() && target.y.is_finite() {
                return Ok((target.x, target.y));
            }
            bail!("in-frame ref `{ref_id}` has no finite viewport point; re-snapshot");
        }
        let evaluated = self
            .get(backend_session_id)?
            .evaluate(target_point_expression(&target)?)?;
        let x = evaluated
            .value
            .get("x")
            .and_then(Value::as_f64)
            .context("browser ref target point missing x")?;
        let y = evaluated
            .value
            .get("y")
            .and_then(Value::as_f64)
            .context("browser ref target point missing y")?;
        if !x.is_finite() || !y.is_finite() {
            bail!("browser ref target point is not finite");
        }
        Ok((x, y))
    }
}

/// Fills one field that lives inside a cross-origin payment iframe (#656),
/// addressed by CDP node identity because the top document can't see into the
/// OOPIF. Focuses the field via `DOM.focus`, verifies focus actually landed
/// (the #580 guard) and clears it, then types one real keystroke per character
/// (`Input.dispatchKeyEvent`) — a guarded widget like Amazon's APX reverts a
/// bulk `Input.insertText` value, so per-character typing is required. Finally
/// reads the value back to confirm it persisted, which — unlike a top-document
/// hosted fill — is possible here.
fn in_frame_fill(session: &BrowserSession, target: &BrowserElementRef, text: &str) -> Result<Value> {
    let backend_node_id = target
        .backend_node_id
        .context("in-frame ref is missing a backend node id; re-snapshot")?;
    session
        .cdp_call("DOM.focus", json!({ "backendNodeId": backend_node_id }))
        .context("focus the in-frame field")?;
    let object_id = resolve_in_frame_object_id(session, backend_node_id)?;
    let prepared = session.cdp_call(
        "Runtime.callFunctionOn",
        json!({
            "objectId": object_id,
            "functionDeclaration": in_frame_prepare_fill_fn(),
            "returnByValue": true,
        }),
    )?;
    let focused = prepared
        .pointer("/result/value/focused")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !focused {
        bail!(
            "fill failed: focus did not land on the in-frame field (it may be covered or still \
             loading). No text was sent."
        );
    }
    // Type one real keystroke per character instead of a single Input.insertText.
    // A guarded card widget (e.g. Amazon's APX) builds its submitted value from
    // genuine keydown events and reverts a bulk programmatic/insertText value, so
    // insertText leaves the field empty at submit time even though it briefly
    // shows in the DOM (#656 follow-up).
    for ch in text.chars() {
        type_char(session, ch)?;
    }
    // Let the widget's re-render settle, then read the value back. Unlike a
    // top-document hosted fill, an in-frame field's value CAN be read — so a
    // reverted/guarded fill is caught honestly instead of being reported as a
    // false success (which previously caused retries until Amazon rate-limited).
    thread::sleep(Duration::from_millis(180));
    let readback = session.cdp_call(
        "Runtime.callFunctionOn",
        json!({
            "objectId": object_id,
            "functionDeclaration": in_frame_readback_fn(),
            "returnByValue": true,
        }),
    )?;
    let value = readback
        .pointer("/result/value/value")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !text.is_empty() && value.is_empty() {
        bail!(
            "fill failed: the in-frame field reverted to empty after typing — the payment widget \
             rejected the input and did not keep it. No value stuck."
        );
    }
    Ok(json!({
        "ok": true,
        "mode": "inFrame",
        "note": "Typed into a field inside a cross-origin payment iframe and confirmed the value \
                 persisted."
    }))
}

/// True when a `fill_expression` failure is the in-page "value did not stick"
/// rejection — the signal that a guarded/controlled main-document input ignored
/// the native value-setter — rather than a transport/resolution error. Only
/// that specific rejection arms the keystroke fallback (#675), so a missing ref
/// or a CDP transport failure still surfaces unchanged.
fn is_value_did_not_stick(err: &anyhow::Error) -> bool {
    err.to_string().contains("value did not stick")
}

/// Re-reads a resolved main-document field's value and reports whether it is
/// empty. Used to catch a guarded input that accepted the native value-setter
/// synchronously but reverted it a tick later (#675).
fn main_doc_field_is_empty(session: &BrowserSession, target: &BrowserElementRef) -> Result<bool> {
    let readback = session.evaluate(main_doc_readback_expression(target)?)?.value;
    let value = readback
        .get("value")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Ok(value.is_empty())
}

/// Fills a keystroke-guarded MAIN-DOCUMENT input (#675) the native value-setter
/// path can't satisfy — the field reverts any value not produced by genuine
/// per-character keystrokes (Olive Young #cardNo; same guard family as Amazon's
/// APX in #656, but that fix only covered the cross-origin in-frame path).
///
/// Unlike the in-frame path, the top document CAN resolve this node, so the
/// click is a real coordinate mouse press at the field's viewport center
/// (trusted input the guard accepts) and focus/readback go through the
/// top-document ref resolver. The sequence: real click to focus → verify
/// `document.activeElement` IS the target and clear residual (#580 guard, so
/// keystrokes can't leak into the wrong element) → triple-click select-all →
/// one trusted keystroke per character → read the value back and BAIL HONESTLY
/// if it still reverted (never a false success, #580).
///
/// This fires ONLY after the native-setter fill was rejected, so fields that
/// fill fine keep the fast path and are unaffected.
fn main_doc_keystroke_fill(
    session: &BrowserSession,
    target: &BrowserElementRef,
    text: &str,
) -> Result<Value> {
    let (x, y) = main_doc_target_point(session, target)?;
    // A real coordinate mouse press is trusted input the keystroke guard
    // accepts; it also routes focus by browser-side hit testing exactly like a
    // user click.
    dispatch_agent_mouse_click(session, x, y, 1)?;
    // Verify focus actually landed on the resolved field before sending any
    // keystrokes, and clear any residual value so typing replaces rather than
    // appends (#580: never let keystrokes leak into whatever else holds focus).
    let prepared = session
        .evaluate(main_doc_focus_clear_expression(target)?)?
        .value;
    let focused = prepared
        .get("focused")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !focused {
        bail!(
            "fill failed: clicking the field at ({x:.0}, {y:.0}) did not move focus onto it (it \
             may be covered by an overlay or still loading). No text was sent."
        );
    }
    // Triple-click select-all clears anything the guard re-inserted after the
    // programmatic clear above, so the keystrokes replace it.
    dispatch_agent_mouse_click(session, x, y, 2)?;
    dispatch_agent_mouse_click(session, x, y, 3)?;
    for ch in text.chars() {
        type_char(session, ch)?;
    }
    // Let the guard's re-render settle, then read the value back. If it still
    // reverted to empty even after genuine per-character keystrokes, bail with
    // the same honest error rather than claiming a fill that did not stick
    // (#580: no false success).
    thread::sleep(Duration::from_millis(180));
    if !text.is_empty() && main_doc_field_is_empty(session, target)? {
        bail!(
            "fill failed: value did not stick — the field reverted to empty even after typing one \
             trusted keystroke per character, so the page is guarding it in a way this fill cannot \
             satisfy. No value stuck."
        );
    }
    Ok(json!({
        "ok": true,
        "mode": "keystroke",
        "note": "The field rejected a programmatic value, so it was filled by typing one trusted \
                 keystroke per character; the value was read back and confirmed to persist."
    }))
}

/// Resolves the current viewport center point of a resolved MAIN-DOCUMENT field
/// for the keystroke fallback's real mouse click. Reuses the same
/// scroll-into-view + clamp logic as the top-document target-point path so the
/// click lands where the field is actually rendered.
fn main_doc_target_point(session: &BrowserSession, target: &BrowserElementRef) -> Result<(f64, f64)> {
    let evaluated = session.evaluate(target_point_expression(target)?)?.value;
    let x = evaluated
        .get("x")
        .and_then(Value::as_f64)
        .context("main-document field target point missing x")?;
    let y = evaluated
        .get("y")
        .and_then(Value::as_f64)
        .context("main-document field target point missing y")?;
    if !x.is_finite() || !y.is_finite() {
        bail!("main-document field target point is not finite; re-snapshot");
    }
    Ok((x, y))
}

/// Types one character into the focused field as a real keystroke (`keyDown`
/// carrying the text, then `keyUp`). The dispatched events are top-level — they
/// route to whatever frame currently holds focus — so this serves both the
/// in-frame payment path (#656) and the main-document keystroke fallback (#675).
/// `nativeVirtualKeyCode` is never set (see input.rs / #636), so this stays on
/// the renderer-only path and never triggers OS key-repeat storms.
fn type_char(session: &BrowserSession, ch: char) -> Result<()> {
    let (key_down, key_up) = type_char_events(ch);
    session.input(key_down)?;
    session.input(key_up)
}

/// Builds the `keyDown`+`keyUp` pair for one trusted character keystroke. Pure
/// so the #636 invariant (the `Key` variant carries no native key code, only a
/// renderer-visible `code`/`text`) is unit-testable without a live browser. The
/// `keyDown` carries the character as `text`; the `keyUp` carries none.
fn type_char_events(ch: char) -> (BrowserInputEvent, BrowserInputEvent) {
    let key = ch.to_string();
    let code = dom_code_for_char(ch);
    (
        BrowserInputEvent::Key {
            event_type: "keyDown".to_string(),
            key: key.clone(),
            code: code.clone(),
            text: Some(key.clone()),
            modifiers: 0,
            commands: Vec::new(),
        },
        BrowserInputEvent::Key {
            event_type: "keyUp".to_string(),
            key,
            code,
            text: None,
            modifiers: 0,
            commands: Vec::new(),
        },
    )
}

/// Best-effort DOM `code` (e.g. `Digit4`, `KeyA`) for a character so guarded
/// fields that inspect `event.code`/`keyCode` see a plausible key. Empty for
/// other characters; the carried `text` still inserts them.
fn dom_code_for_char(ch: char) -> String {
    if ch.is_ascii_digit() {
        format!("Digit{ch}")
    } else if ch.is_ascii_alphabetic() {
        format!("Key{}", ch.to_ascii_uppercase())
    } else {
        String::new()
    }
}

/// Selects one option on a native `<select>` inside a cross-origin payment
/// iframe (the Amazon expiry month/year). A native popup can't be driven by
/// coordinate clicks, so the value is set inside the frame via `callFunctionOn`,
/// firing `input`/`change`. Fails loudly when no option matches (#580: never
/// claim success silently).
fn in_frame_select(session: &BrowserSession, target: &BrowserElementRef, value: &str) -> Result<()> {
    let backend_node_id = target
        .backend_node_id
        .context("in-frame ref is missing a backend node id; re-snapshot")?;
    let object_id = resolve_in_frame_object_id(session, backend_node_id)?;
    let outcome = session.cdp_call(
        "Runtime.callFunctionOn",
        json!({
            "objectId": object_id,
            "functionDeclaration": in_frame_select_fn(value)?,
            "returnByValue": true,
        }),
    )?;
    let matched = outcome
        .pointer("/result/value/matched")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !matched {
        let available = outcome
            .pointer("/result/value/available")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if available.is_empty() {
            bail!("select failed: no option matched \"{value}\" in the in-frame dropdown");
        }
        bail!("select failed: no option matched \"{value}\". Available: {available}");
    }
    let selected = outcome
        .pointer("/result/value/value")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Confirm the selection persisted: a guarded/controlled dropdown can revert a
    // programmatic value after re-render, which would otherwise submit as the
    // default (#656 follow-up). Read it back after a beat and fail honestly.
    thread::sleep(Duration::from_millis(180));
    let readback = session.cdp_call(
        "Runtime.callFunctionOn",
        json!({
            "objectId": object_id,
            "functionDeclaration": in_frame_readback_fn(),
            "returnByValue": true,
        }),
    )?;
    let now = readback
        .pointer("/result/value/value")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if now != selected {
        bail!(
            "select failed: the in-frame dropdown reverted after selecting \"{value}\" (the widget \
             did not keep the selection)."
        );
    }
    Ok(())
}

/// Returns the CDP backend node id when a ref points inside a cross-origin
/// payment iframe, so ref actions can route through node-identity CDP calls
/// instead of the top-document `findTarget` path that can't reach it.
fn in_frame_backend_node_id(target: &BrowserElementRef) -> Option<i64> {
    if target.in_frame {
        target.backend_node_id
    } else {
        None
    }
}

/// Resolves a backend node inside a cross-origin frame to a JS object id in that
/// frame's own context, so `Runtime.callFunctionOn` runs inside the OOPIF.
fn resolve_in_frame_object_id(session: &BrowserSession, backend_node_id: i64) -> Result<String> {
    let resolved = session.cdp_call("DOM.resolveNode", json!({ "backendNodeId": backend_node_id }))?;
    resolved
        .pointer("/object/objectId")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .context("in-frame node could not be resolved to a script object")
}

/// Completes a fill that targets a hosted (cross-origin) field iframe.
///
/// The value cannot be read back from the top document, so this path is
/// strict about the one thing it can verify: focus must actually land inside
/// the target frame before any text is sent, otherwise the keystrokes would
/// leak into whatever element holds focus (#580-class silent misfill).
fn hosted_frame_fill(session: &BrowserSession, x: f64, y: f64, text: &str) -> Result<Value> {
    // Right after the probe's scrollIntoView the browser-side hit test can
    // lag the new layout by a frame, routing a correctly-placed click to the
    // parent document instead of the iframe. Give the compositor a beat,
    // then retry the whole click with refreshed coordinates if focus does
    // not arrive.
    let mut focused = false;
    let mut point = (x, y);
    for attempt in 0..4 {
        if attempt > 0 {
            let refreshed = session
                .evaluate(hosted_fill_point_expression().to_string())?
                .value;
            if let (Some(x), Some(y)) = (
                refreshed.get("x").and_then(Value::as_f64),
                refreshed.get("y").and_then(Value::as_f64),
            ) {
                point = (x, y);
            }
        }
        thread::sleep(Duration::from_millis(80));
        dispatch_agent_mouse_click(session, point.0, point.1, 1)?;
        for _ in 0..6 {
            thread::sleep(Duration::from_millis(50));
            let check = session
                .evaluate(hosted_fill_focus_check_expression().to_string())?
                .value;
            if check
                .get("focused")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                focused = true;
                break;
            }
        }
        if focused {
            break;
        }
    }
    if !focused {
        let (x, y) = point;
        bail!(
            "fill failed: clicking the hosted field iframe at ({x:.0}, {y:.0}) did not move focus \
             into the frame; the field may be covered by an overlay or still loading. \
             No text was sent."
        );
    }
    // Clear any prior content by selecting it with a triple click — the
    // standard select-all gesture for single-line fields. Mouse events are
    // routed by browser-side hit testing into the frame, and `insertText`
    // then replaces the live selection in the focused frame. No synthetic
    // keyboard events: key dispatch can stall the CT Chromium input pipeline
    // (observed as a key-event storm wedging CDP), and platform select-all
    // shortcuts never reach cross-origin frames anyway.
    dispatch_agent_mouse_click(session, point.0, point.1, 2)?;
    dispatch_agent_mouse_click(session, point.0, point.1, 3)?;
    session.input(BrowserInputEvent::Text {
        text: text.to_string(),
    })?;
    Ok(json!({
        "ok": true,
        "mode": "hostedFrame",
        "note": "Typed into a hosted (cross-origin) field iframe after verifying focus. \
                 The frame's value cannot be read back from the top document; confirm via \
                 the page's own field validation (e.g. error labels) in the next snapshot."
    }))
}

fn dispatch_agent_mouse_click(
    session: &BrowserSession,
    x: f64,
    y: f64,
    click_count: u32,
) -> Result<()> {
    session.input(BrowserInputEvent::Mouse {
        event_type: "mouseMoved".to_string(),
        x,
        y,
        button: "none".to_string(),
        buttons: Some(0),
        click_count: 0,
    })?;
    session.input(BrowserInputEvent::Mouse {
        event_type: "mousePressed".to_string(),
        x,
        y,
        button: "left".to_string(),
        buttons: Some(1),
        click_count,
    })?;
    session.input(BrowserInputEvent::Mouse {
        event_type: "mouseReleased".to_string(),
        x,
        y,
        button: "left".to_string(),
        buttons: Some(0),
        click_count,
    })
}

/// Reads a coordinate param tolerantly: accepts a JSON float, integer, or a
/// numeric string (models sometimes emit `"339"` instead of `339`). Returns
/// None only when the value is absent or unparseable.
fn coord_param(params: &Value, key: &str) -> Option<f64> {
    let v = params.get(key)?;
    v.as_f64()
        .or_else(|| v.as_i64().map(|n| n as f64))
        .or_else(|| v.as_u64().map(|n| n as f64))
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
}

/// Moves the pointer to (x, y) along a RANDOMIZED curved (quadratic Bézier) path
/// with per-step sub-pixel jitter and variable inter-step dwell, approaching from
/// a random nearby point and settling exactly on the target. Real, non-repeating
/// pointer telemetry — behavioral anti-bot scoring (reCAPTCHA checkbox / v3) both
/// penalizes teleports and pattern-matches a fixed synthetic curve, so the path
/// must vary each time rather than be deterministic.
fn dispatch_humanized_move(session: &BrowserSession, x: f64, y: f64) -> Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let start_x = x - rng.gen_range(18.0..64.0);
    let start_y = y - rng.gen_range(-30.0..44.0);
    // Control point off the straight line gives the path a natural curve.
    let ctrl_x = (start_x + x) / 2.0 + rng.gen_range(-28.0..28.0);
    let ctrl_y = (start_y + y) / 2.0 + rng.gen_range(-28.0..28.0);
    let steps = rng.gen_range(5..=9);
    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let e = t * t * (3.0 - 2.0 * t); // smoothstep velocity profile
        let iu = 1.0 - e;
        let mx = iu * iu * start_x + 2.0 * iu * e * ctrl_x + e * e * x + rng.gen_range(-1.3..1.3);
        let my = iu * iu * start_y + 2.0 * iu * e * ctrl_y + e * e * y + rng.gen_range(-1.3..1.3);
        session.input(BrowserInputEvent::Mouse {
            event_type: "mouseMoved".to_string(),
            x: mx,
            y: my,
            button: "none".to_string(),
            buttons: Some(0),
            click_count: 0,
        })?;
        thread::sleep(Duration::from_millis(rng.gen_range(4..12)));
    }
    // Settle exactly on the target so the press lands where intended.
    session.input(BrowserInputEvent::Mouse {
        event_type: "mouseMoved".to_string(),
        x,
        y,
        button: "none".to_string(),
        buttons: Some(0),
        click_count: 0,
    })
}

fn ensure_target_tab(
    state: &Arc<DaemonState>,
    root_session_id: &str,
    params: &Value,
    width: u32,
    height: u32,
) -> Result<(String, String)> {
    let background = background_requested(params);
    let tabs = state.browsers.list_tabs(root_session_id);
    if let Some(tab_id) = optional_string(params, "tabId").or_else(|| tab_id_from_page(params)) {
        let backend_id = backend_session_id(root_session_id, &tab_id);
        let restore_url = tabs
            .tabs
            .iter()
            .find(|tab| tab.tab_id == tab_id)
            .map(|tab| tab.url.clone())
            .unwrap_or_else(|| DEFAULT_URL.to_string());
        ensure_backend_session(
            state,
            root_session_id,
            &tab_id,
            &backend_id,
            restore_url,
            width,
            height,
            background,
        )?;
        return Ok((tab_id, backend_id));
    }
    if let Some(tab) = active_or_first(&tabs) {
        ensure_backend_session(
            state,
            root_session_id,
            &tab.tab_id,
            &tab.backend_session_id,
            tab.url.clone(),
            width,
            height,
            background,
        )?;
        return Ok((tab.tab_id, tab.backend_session_id));
    }
    let tab = state.browsers.open_tab(
        state.event_sender(),
        root_session_id.to_string(),
        None,
        None,
        Some(DEFAULT_URL.to_string()),
        width,
        height,
        !background,
        background,
    )?;
    publish_tabs(state, root_session_id);
    Ok((tab.tab_id, tab.backend_session_id))
}

fn tab_id_from_page(params: &Value) -> Option<String> {
    let page = params.get("page")?;
    if let Some(tab_id) = page
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(tab_id.to_string());
    }
    page.get("tabId")
        .or_else(|| page.get("tab_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn ensure_backend_session(
    state: &Arc<DaemonState>,
    root_session_id: &str,
    tab_id: &str,
    backend_id: &str,
    restore_url: String,
    width: u32,
    height: u32,
    background: bool,
) -> Result<()> {
    if state.browsers.resize(backend_id, width, height).is_ok() {
        return Ok(());
    }
    let browser_state = state.browsers.open(
        state.event_sender(),
        backend_id.to_string(),
        Some(restore_url),
        width,
        height,
        !background,
    )?;
    let native_cef_session_id = state
        .browsers
        .tabs
        .lock()
        .unwrap()
        .tab(root_session_id, tab_id)
        .and_then(|tab| tab.native_cef_session_id);
    state.browsers.tabs.lock().unwrap().open_tab(
        root_session_id,
        Some(tab_id.to_string()),
        None,
        backend_id.to_string(),
        native_cef_session_id,
        browser_state,
        false,
    );
    publish_tabs(state, root_session_id);
    Ok(())
}

fn active_or_first(tabs: &BrowserTabsState) -> Option<BrowserTabInfo> {
    tabs.tabs
        .iter()
        .find(|tab| tab.active)
        .or_else(|| tabs.tabs.first())
        .cloned()
}

/// Workspace-stable CLI-browser root session ids start with this prefix
/// (`cli-browser-<slug>-<uuid>`, see `client::new_browser_session_id`). Chat
/// session UUIDs never do, so this distinguishes a Bash-`browser` keyspace from
/// a typed-Browser/pane keyspace without a registry lookup.
const CLI_BROWSER_SESSION_PREFIX: &str = "cli-browser-";

fn is_cli_browser_session(session_id: &str) -> bool {
    session_id.starts_with(CLI_BROWSER_SESSION_PREFIX)
}

/// Lists tabs for `root_session_id`, falling back to the workspace-stable
/// cli-browser keyspace when the chat-session UUID owns no tabs of its own
/// (issue #667). The Bash `browser` skill registers tabs under the cli-browser
/// id, but the visible in-app pane lists/subscribes by chat-session UUID — so
/// without this fallback the pane shows a blank `about:blank` even while a CLI
/// tab is live. Precedence is preserved: a chat-session's own tabs always win;
/// the CLI keyspace is consulted only when the chat keyspace is empty. When the
/// fallback fires it also registers an event bridge so future CLI tab-list
/// changes are mirrored onto this pane's `browser:<chat-uuid>:tabs` channel.
fn list_tabs_with_cli_fallback(
    state: &Arc<DaemonState>,
    root_session_id: &str,
) -> BrowserTabsState {
    let primary = state.browsers.list_tabs(root_session_id);
    if !primary.tabs.is_empty() {
        return primary;
    }
    // Only a typed/chat keyspace falls back; a CLI keyspace listing itself must
    // not bridge onto itself (and `register_tab_bridge` guards that too).
    if is_cli_browser_session(root_session_id) {
        return primary;
    }
    let Ok(cli_session_id) = crate::daemon_browser::default_cli_session_id(state.config_paths())
    else {
        return primary;
    };
    if cli_session_id == root_session_id {
        return primary;
    }
    let cli_tabs = state.browsers.list_tabs(&cli_session_id);
    if cli_tabs.tabs.is_empty() {
        return primary;
    }
    // Late-join reconcile: remember this pane so subsequent CLI tab updates are
    // mirrored to its channel (registration is idempotent).
    state
        .browsers
        .register_tab_bridge(&cli_session_id, root_session_id);
    cli_tabs
}

pub(crate) fn publish_tabs(state: &Arc<DaemonState>, root_session_id: &str) {
    let payload = serde_json::to_value(state.browsers.list_tabs(root_session_id))
        .unwrap_or_else(|_| json!({ "tabs": [] }));
    state.publish_event(ServerEnvelope::Event {
        event: format!("browser:{root_session_id}:tabs"),
        payload: payload.clone(),
    });
    // Mirror CLI-browser tab updates onto every chat-session pane bridged to
    // this keyspace, so the visible pane gets live updates with no frontend
    // change (issue #667). Only the cli-browser keyspace carries viewers.
    if is_cli_browser_session(root_session_id) {
        for viewer in state.browsers.bridged_viewers(root_session_id) {
            state.publish_event(ServerEnvelope::Event {
                event: format!("browser:{viewer}:tabs"),
                payload: payload.clone(),
            });
        }
    }
}

fn optional_string(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn background_requested(params: &Value) -> bool {
    params
        .get("background")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn navigation_timeout(params: &Value) -> Duration {
    Duration::from_millis(u64::from(
        optional_u32(params, "timeoutMs")
            .unwrap_or(30_000)
            .clamp(1, 120_000),
    ))
}

fn network_idle_duration(params: &Value) -> Duration {
    Duration::from_millis(u64::from(
        optional_u32(params, "idleMs")
            .unwrap_or(500)
            .clamp(0, 30_000),
    ))
}

/// Returns the text payload for one synthesized key event when applicable.
pub(super) fn key_text(key: &str) -> Option<String> {
    (key.len() == 1).then(|| key.to_string())
}

/// A parsed `Modifier+Key` combo for press/keydown/keyup actions.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct KeyCombo {
    pub(super) key: String,
    pub(super) modifiers: u32,
    pub(super) commands: Vec<String>,
}

/// Parses agent key strings like `Meta+A`, `Ctrl+Shift+Z`, or plain `Enter`
/// into a CDP key plus modifier mask. Editing shortcuts that depend on
/// platform-level translation (macOS `Cmd+A`) are mapped to explicit CDP
/// editing `commands` so they work in any focused frame, including
/// cross-origin payment iframes.
pub(super) fn parse_key_combo(raw: &str) -> KeyCombo {
    let mut modifiers = 0u32;
    let mut key = raw.to_string();
    if raw.len() > 1 && raw.contains('+') {
        let parts: Vec<&str> = raw.split('+').collect();
        let mut parsed_modifiers = 0u32;
        let mut consumed = 0usize;
        while consumed < parts.len() - 1 {
            match modifier_mask(parts[consumed]) {
                Some(mask) => {
                    parsed_modifiers |= mask;
                    consumed += 1;
                }
                None => break,
            }
        }
        if consumed > 0 {
            // The unconsumed tail is the key; `Meta++` leaves ["", ""],
            // which joins back into the literal `+` key.
            let tail = parts[consumed..].join("+");
            modifiers = parsed_modifiers;
            key = if tail.is_empty() {
                "+".to_string()
            } else {
                tail
            };
        }
    }
    let commands = editing_commands(&key, modifiers);
    KeyCombo {
        key,
        modifiers,
        commands,
    }
}

fn modifier_mask(name: &str) -> Option<u32> {
    match name.to_ascii_lowercase().as_str() {
        "alt" | "option" | "opt" => Some(1),
        "control" | "ctrl" => Some(2),
        "meta" | "cmd" | "command" | "super" | "win" => Some(4),
        "shift" => Some(8),
        _ => None,
    }
}

/// Maps primary-modifier editing shortcuts to CDP editing commands. CDP key
/// events are injected below the platform shortcut layer, so on macOS
/// `Cmd+A` never reaches the renderer as select-all unless the command is
/// attached explicitly.
fn editing_commands(key: &str, modifiers: u32) -> Vec<String> {
    let primary = modifiers & (2 | 4) != 0;
    let extra = modifiers & !(2 | 4);
    if !primary || extra != 0 {
        return Vec::new();
    }
    match key.to_ascii_lowercase().as_str() {
        "a" => vec!["selectAll".to_string()],
        _ => Vec::new(),
    }
}

/// Converts one named scroll direction into wheel deltas.
pub(super) fn scroll_delta(direction: &str, px: u32) -> Result<(f64, f64)> {
    match direction {
        "up" => Ok((0.0, -f64::from(px))),
        "down" => Ok((0.0, f64::from(px))),
        "left" => Ok((-f64::from(px), 0.0)),
        "right" => Ok((f64::from(px), 0.0)),
        other => bail!("unsupported scroll direction `{other}`; use up, down, left, or right"),
    }
}

fn key_code(key: &str) -> String {
    match key {
        "Enter" => "Enter",
        "Escape" => "Escape",
        "Tab" => "Tab",
        "Backspace" => "Backspace",
        "Delete" => "Delete",
        "ArrowUp" => "ArrowUp",
        "ArrowDown" => "ArrowDown",
        "ArrowLeft" => "ArrowLeft",
        "ArrowRight" => "ArrowRight",
        value if value.len() == 1 && value.chars().all(|c| c.is_ascii_alphabetic()) => {
            return format!("Key{}", value.to_ascii_uppercase());
        }
        value if value.len() == 1 && value.chars().all(|c| c.is_ascii_digit()) => {
            return format!("Digit{value}");
        }
        _ => key,
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::super::BrowserState;
    use super::*;

    fn test_browser_state(url: &str) -> BrowserState {
        BrowserState {
            url: url.to_string(),
            title: String::new(),
            loading: false,
            width: INITIAL_WIDTH,
            height: INITIAL_HEIGHT,
        }
    }

    #[test]
    fn open_target_resolution_reserves_first_new_tab() {
        let profile = tempfile::tempdir().unwrap();
        let browsers = BrowserRegistry::new(
            profile.path().to_path_buf(),
            false,
            crate::daemon_browser::BrowserLaunchSettings::default(),
        );
        let params = json!({});

        let tab_id = resolve_open_target_tab_id(&browsers, "root", &params, true);
        assert_eq!(tab_id, "t1");
        let backend_id = backend_session_id("root", &tab_id);
        let tab = browsers.tabs.lock().unwrap().open_tab(
            "root",
            Some(tab_id.clone()),
            None,
            backend_id.clone(),
            None,
            test_browser_state("about:blank"),
            true,
        );

        assert_eq!(tab.backend_session_id, backend_id);
        assert_eq!(browsers.tabs.lock().unwrap().next_tab_id("root"), "t2");
    }

    #[test]
    fn open_target_resolution_reuses_active_tab() {
        let profile = tempfile::tempdir().unwrap();
        let browsers = BrowserRegistry::new(
            profile.path().to_path_buf(),
            false,
            crate::daemon_browser::BrowserLaunchSettings::default(),
        );
        browsers.tabs.lock().unwrap().open_tab(
            "root",
            Some("existing".to_string()),
            None,
            backend_session_id("root", "existing"),
            None,
            test_browser_state("https://example.com"),
            true,
        );

        let tab_id = resolve_open_target_tab_id(&browsers, "root", &json!({}), true);

        assert_eq!(tab_id, "existing");
    }

    #[test]
    fn page_handle_can_resolve_target_tab_id() {
        assert_eq!(
            tab_id_from_page(&json!({ "page": { "tabId": "t3" } })).as_deref(),
            Some("t3")
        );
        assert_eq!(
            tab_id_from_page(&json!({ "page": "t4" })).as_deref(),
            Some("t4")
        );
    }

    #[test]
    fn dom_code_for_char_maps_digits_and_letters_only() {
        assert_eq!(dom_code_for_char('4'), "Digit4");
        assert_eq!(dom_code_for_char('a'), "KeyA");
        assert_eq!(dom_code_for_char('Z'), "KeyZ");
        // Non-alphanumeric characters carry no DOM code; the keystroke's text
        // still inserts them.
        assert_eq!(dom_code_for_char(' '), "");
        assert_eq!(dom_code_for_char('-'), "");
    }

    #[test]
    fn type_char_events_carry_text_on_keydown_only_and_no_native_key_code() {
        // The keystroke pair drives both the in-frame (#656) and main-document
        // (#675) trusted fills. The keyDown carries the character as text; the
        // keyUp carries none. Crucially, the `Key` variant has no field for a
        // native virtual key code at all, so it can never reach the OS NSEvent
        // path that caused the macOS autorepeat storm (#636).
        let (down, up) = type_char_events('4');
        match down {
            BrowserInputEvent::Key {
                event_type,
                key,
                code,
                text,
                modifiers,
                commands,
            } => {
                assert_eq!(event_type, "keyDown");
                assert_eq!(key, "4");
                assert_eq!(code, "Digit4");
                assert_eq!(text.as_deref(), Some("4"));
                assert_eq!(modifiers, 0);
                assert!(commands.is_empty());
            }
            _ => panic!("expected keyDown Key event"),
        }
        match up {
            BrowserInputEvent::Key {
                event_type, text, ..
            } => {
                assert_eq!(event_type, "keyUp");
                assert_eq!(text, None);
            }
            _ => panic!("expected keyUp Key event"),
        }
    }

    #[test]
    fn is_value_did_not_stick_arms_fallback_only_for_the_revert_rejection() {
        // Only the in-page "value did not stick" rejection arms the keystroke
        // fallback (#675). A missing ref or a transport error must surface
        // unchanged so the fallback never fires for unrelated failures.
        assert!(is_value_did_not_stick(&anyhow::anyhow!(
            "fill failed: value did not stick (the field may be inside a cross-origin iframe \
             or guarded by the page)"
        )));
        assert!(!is_value_did_not_stick(&anyhow::anyhow!(
            "No element matched browser ref @e7"
        )));
        assert!(!is_value_did_not_stick(&anyhow::anyhow!(
            "timed out waiting for browser evaluation"
        )));
    }
}

/// Runs `expr` inside a cross-origin frame via an isolated world. Only works
/// because Puffer disables site isolation, so the frame shares the renderer.
/// Shared by the audio solver and the image fallback.
#[cfg(any(feature = "captcha-audio", feature = "captcha-image"))]
fn eval_in_frame(session: &BrowserSession, frame_id: &str, expr: &str) -> Result<Value> {
    let world = session.cdp_call(
        "Page.createIsolatedWorld",
        json!({ "frameId": frame_id, "worldName": "puffer_captcha", "grantUniveralAccess": true }),
    )?;
    let ctx = world
        .get("executionContextId")
        .and_then(Value::as_i64)
        .ok_or_else(|| anyhow::anyhow!("no execution context for frame {frame_id}"))?;
    let resp = session.cdp_call(
        "Runtime.evaluate",
        json!({ "expression": expr, "contextId": ctx, "returnByValue": true, "awaitPromise": true }),
    )?;
    Ok(resp.pointer("/result/value").cloned().unwrap_or(Value::Null))
}

/// Universal "captcha solved" signal that does NOT depend on the (optional) anchor
/// frame: reCAPTCHA sets the response token on the TOP document once ANY path
/// (silent checkbox pass, audio, or image) succeeds. The image solver must use this
/// — `anchor` is legitimately `None` when the frame poll captured only the bframe,
/// and gating success on `Some(anchor)` would report a solved captcha as unsolved.
#[cfg(any(feature = "captcha-audio", feature = "captcha-image"))]
fn recaptcha_solved(session: &BrowserSession) -> bool {
    session
        .evaluate(
            "(()=>{try{return !!(window.grecaptcha&&grecaptcha.getResponse&&grecaptcha.getResponse())}catch(e){return false}})()"
                .to_string(),
        )
        .ok()
        .and_then(|e| e.value.as_bool())
        .unwrap_or(false)
}

/// Reads an attribute off a `DOM.getDocument` node (flat `[k,v,k,v,...]` array).
#[cfg(feature = "captcha-audio")]
fn dom_attr(node: &Value, name: &str) -> String {
    let Some(arr) = node.get("attributes").and_then(Value::as_array) else {
        return String::new();
    };
    let mut it = arr.iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        if k.as_str() == Some(name) {
            return v.as_str().unwrap_or_default().to_string();
        }
    }
    String::new()
}

/// Finds the reCAPTCHA v2 anchor (checkbox) and bframe (challenge) CDP frame ids
/// by piercing the DOM (`DOM.getDocument { pierce: true }`). `Page.getFrameTree`
/// does not surface these cross-site frames in Puffer's headless setup, but the
/// pierced DOM does (site isolation is disabled, so they share the renderer).
/// `(anchor, bframe)`.
#[cfg(feature = "captcha-audio")]
fn recaptcha_frame_ids(
    session: &BrowserSession,
) -> Result<(Option<String>, Option<String>)> {
    fn walk(node: &Value, anchor: &mut Option<String>, bframe: &mut Option<String>) {
        let is_iframe = node
            .get("nodeName")
            .and_then(Value::as_str)
            .map(|n| n.eq_ignore_ascii_case("iframe"))
            .unwrap_or(false);
        if is_iframe {
            let frame_id = node.get("frameId").and_then(Value::as_str);
            let content = node.get("contentDocument");
            let url = content
                .and_then(|c| c.get("documentURL"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| dom_attr(node, "src"));
            if let Some(fid) = frame_id {
                if url.contains("/recaptcha/") && url.contains("/anchor") && anchor.is_none() {
                    *anchor = Some(fid.to_string());
                }
                if url.contains("/recaptcha/") && url.contains("/bframe") && bframe.is_none() {
                    *bframe = Some(fid.to_string());
                }
            }
            if let Some(content) = content {
                walk(content, anchor, bframe);
            }
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for child in children {
                walk(child, anchor, bframe);
            }
        }
    }
    let doc = session.cdp_call("DOM.getDocument", json!({ "depth": -1, "pierce": true }))?;
    let (mut anchor, mut bframe) = (None, None);
    if let Some(root) = doc.get("root") {
        walk(root, &mut anchor, &mut bframe);
    }
    Ok((anchor, bframe))
}

/// Debug: dump iframes (pierced DOM) + frame-tree urls to diagnose why a captcha
/// wasn't found. Temporary aid for bring-up.
#[cfg(feature = "captcha-audio")]
fn recaptcha_debug(session: &BrowserSession) -> Vec<String> {
    fn walk(node: &Value, out: &mut Vec<String>) {
        if node
            .get("nodeName")
            .and_then(Value::as_str)
            .map(|n| n.eq_ignore_ascii_case("iframe"))
            .unwrap_or(false)
        {
            let fid = node.get("frameId").and_then(Value::as_str).unwrap_or("NONE");
            let src: String = dom_attr(node, "src").chars().take(55).collect();
            let durl: String = node
                .get("contentDocument")
                .and_then(|c| c.get("documentURL"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(55)
                .collect();
            out.push(format!(
                "iframe fid={fid} content={} src={src} docURL={durl}",
                node.get("contentDocument").is_some()
            ));
        }
        if let Some(content) = node.get("contentDocument") {
            walk(content, out);
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for child in children {
                walk(child, out);
            }
        }
    }
    let mut out = Vec::new();
    if let Ok(doc) = session.cdp_call("DOM.getDocument", json!({ "depth": -1, "pierce": true })) {
        if let Some(root) = doc.get("root") {
            walk(root, &mut out);
        }
    }
    fn ftree(node: &Value, out: &mut Vec<String>) {
        if let Some(frame) = node.get("frame") {
            out.push(format!(
                "ftree url={}",
                frame.get("url").and_then(Value::as_str).unwrap_or("")
            ));
        }
        if let Some(children) = node.get("childFrames").and_then(Value::as_array) {
            for child in children {
                ftree(child, out);
            }
        }
    }
    if let Ok(tree) = session.cdp_call("Page.getFrameTree", json!({})) {
        if let Some(root) = tree.get("frameTree") {
            ftree(root, &mut out);
        }
    }
    out
}
