use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const DEFAULT_PORT: u16 = 5050;
const CAFE_PORT: u16 = 8787;
const DASHBOARD_WAIT_SECS: u64 = 20;
const CAFE_WAIT_SECS: u64 = 90;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NavRoute {
    pub id: String,
    pub label: String,
    pub path: String,
    pub icon: String,
    pub group: String,
    pub activity: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DashboardStatus {
    pub running: bool,
    pub port: u16,
    pub base_url: String,
    pub listener_active: Option<bool>,
    pub message: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CafeAgentStatus {
    pub online: bool,
    pub status: String,
    pub sales_rows: Option<i64>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    pub ok: bool,
    pub ptt_enabled: bool,
    pub ptt_hotkey_display: String,
    pub continuous_listening: bool,
    pub whisper_lazy_load: bool,
    pub whisper_model: String,
    pub wake_word: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListenerStatus {
    pub is_listening: bool,
    pub process_alive: bool,
    pub pid: Option<u32>,
}

pub struct BackendState {
    pub jarvis_root: PathBuf,
    pub dashboard_child: Mutex<Option<Child>>,
    pub cafe_child: Mutex<Option<Child>>,
}

pub fn resolve_jarvis_root() -> PathBuf {
    if let Ok(root) = std::env::var("JARVIS_ROOT") {
        let p = PathBuf::from(root);
        if p.join("src").is_dir() {
            return p;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("src").is_dir() {
            return cwd;
        }
        if let Some(parent) = cwd.parent().and_then(|p| p.parent()) {
            if parent.join("src").is_dir() {
                return parent.to_path_buf();
            }
        }
    }
    PathBuf::from(".")
}

fn resolve_python(root: &Path) -> PathBuf {
    let win = root.join(".venv").join("Scripts").join("python.exe");
    if win.is_file() {
        return win;
    }
    let unix = root.join(".venv").join("bin").join("python");
    if unix.is_file() {
        return unix;
    }
    if cfg!(windows) {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn apply_hidden(cmd: &mut Command) {
    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn port_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().expect("valid addr"),
        Duration::from_millis(500),
    )
    .is_ok()
}

fn http_get_json_host(host: &str, port: u16, path: &str) -> Option<serde_json::Value> {
    let mut stream = TcpStream::connect(format!("{host}:{port}")).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).ok()?;
    let body = buf.split("\r\n\r\n").nth(1)?.trim();
    serde_json::from_str(body).ok()
}

fn http_get_json(port: u16, path: &str) -> Option<serde_json::Value> {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).ok()?;
    let body = buf.split("\r\n\r\n").nth(1)?.trim();
    serde_json::from_str(body).ok()
}

fn status_from_port(port: u16, message: &str) -> DashboardStatus {
    let base_url = format!("http://127.0.0.1:{port}");
    let running = port_open(port);
    let listener_active = if running {
        http_get_json(port, "/api/dashboard/status").and_then(|v| {
            v.get("listening")
                .or_else(|| v.get("listener_active"))
                .and_then(|x| x.as_bool())
        })
    } else {
        None
    };
    DashboardStatus {
        running,
        port,
        base_url,
        listener_active,
        message: message.to_string(),
    }
}

fn spawn_dashboard(root: &Path, port: u16) -> Result<Child, String> {
    let python = resolve_python(root);
    let mut cmd = Command::new(&python);
    cmd.args(["-m", "desktop_app.memory_viewer", &port.to_string()])
        .env("PYTHONPATH", root.join("src"))
        .env("JARVIS_ROOT", root)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_hidden(&mut cmd);
    cmd.spawn()
        .map_err(|e| format!("Failed to start dashboard ({python:?}): {e}"))
}

fn wait_for_port(port: u16, max_secs: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(max_secs);
    while Instant::now() < deadline {
        if port_open(port) {
            return true;
        }
        thread::sleep(Duration::from_millis(250));
    }
    false
}

fn resolve_cargo() -> PathBuf {
    if cfg!(windows) {
        if let Ok(home) = std::env::var("USERPROFILE") {
            let exe = PathBuf::from(home)
                .join(".cargo")
                .join("bin")
                .join("cargo.exe");
            if exe.is_file() {
                return exe;
            }
        }
    }
    PathBuf::from("cargo")
}

fn spawn_cafe_orchestrator(root: &Path) -> Result<Child, String> {
    let cafe_dir = root.join("cafe-agent");
    if !cafe_dir.is_dir() {
        return Err(format!(
            "cafe-agent directory not found at {}",
            cafe_dir.display()
        ));
    }
    let example = cafe_dir.join("config.example.toml");
    let cfg = cafe_dir.join("config.toml");
    if !cfg.is_file() {
        if example.is_file() {
            std::fs::copy(&example, &cfg)
                .map_err(|e| format!("Failed to create cafe-agent/config.toml: {e}"))?;
        } else {
            return Err("cafe-agent/config.example.toml missing".into());
        }
    }

    let release_name = if cfg!(windows) {
        "cafe-orchestrator.exe"
    } else {
        "cafe-orchestrator"
    };
    let release_bin = cafe_dir.join("target").join("release").join(release_name);

    let mut cmd = if release_bin.is_file() {
        Command::new(&release_bin)
    } else {
        let cargo = resolve_cargo();
        let mut c = Command::new(&cargo);
        c.args(["run", "-p", "orchestrator", "--quiet"]);
        c.current_dir(&cafe_dir);
        c
    };

    cmd.env("JARVIS_ROOT", root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_hidden(&mut cmd);
    cmd.spawn()
        .map_err(|e| format!("Failed to start cafe-orchestrator: {e}"))
}

fn run_python_script(root: &Path, script_rel: &str, args: &[&str]) -> Result<Output, String> {
    let python = resolve_python(root);
    let script = root.join(script_rel);
    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .args(args)
        .env("PYTHONPATH", root.join("src"))
        .env("JARVIS_ROOT", root)
        .current_dir(root);
    apply_hidden(&mut cmd);
    cmd.output()
        .map_err(|e| format!("Failed to run {script_rel}: {e}"))
}

fn parse_listener_json(stdout: &[u8]) -> ListenerStatus {
    let v: serde_json::Value = serde_json::from_slice(stdout).unwrap_or_default();
    ListenerStatus {
        is_listening: v
            .get("is_listening")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        process_alive: v
            .get("process_alive")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        pid: v.get("pid").and_then(|x| x.as_u64()).map(|p| p as u32),
    }
}

#[tauri::command]
fn get_nav_routes(port: Option<u16>) -> Vec<NavRoute> {
    let port = port.unwrap_or(DEFAULT_PORT);
    let base = format!("http://127.0.0.1:{port}");
    vec![
        NavRoute {
            id: "home".into(),
            label: "Home".into(),
            path: String::new(),
            icon: "⌂".into(),
            group: String::new(),
            activity: "explorer".into(),
        },
        NavRoute {
            id: "command".into(),
            label: "Command Centre".into(),
            path: format!("{base}/dashboard"),
            icon: "▣".into(),
            group: "Workspace".into(),
            activity: "explorer".into(),
        },
        NavRoute {
            id: "memory".into(),
            label: "Diary & memories".into(),
            path: format!("{base}/?tab=memories"),
            icon: "📓".into(),
            group: "Knowledge".into(),
            activity: "explorer".into(),
        },
        NavRoute {
            id: "graph".into(),
            label: "Knowledge graph".into(),
            path: format!("{base}/?tab=graph"),
            icon: "◇".into(),
            group: "Knowledge".into(),
            activity: "explorer".into(),
        },
        NavRoute {
            id: "meals".into(),
            label: "Meals".into(),
            path: format!("{base}/?tab=meals"),
            icon: "◆".into(),
            group: "Knowledge".into(),
            activity: "explorer".into(),
        },
        NavRoute {
            id: "chat".into(),
            label: "Chat & status".into(),
            path: format!("{base}/dashboard"),
            icon: "💬".into(),
            group: "Assistant".into(),
            activity: "assistant".into(),
        },
        NavRoute {
            id: "sulainis".into(),
            label: "Sulainis".into(),
            path: format!("{base}/sulainis/"),
            icon: "⚡".into(),
            group: "Operator".into(),
            activity: "operator".into(),
        },
        NavRoute {
            id: "pulse".into(),
            label: "Pulse".into(),
            path: format!("{base}/pulse/"),
            icon: "◉".into(),
            group: "Operator".into(),
            activity: "operator".into(),
        },
        NavRoute {
            id: "logs".into(),
            label: "Logs".into(),
            path: format!("{base}/dashboard"),
            icon: "≡".into(),
            group: "System".into(),
            activity: "explorer".into(),
        },
        NavRoute {
            id: "settings".into(),
            label: "Settings".into(),
            path: String::new(),
            icon: "⚙".into(),
            group: "System".into(),
            activity: "explorer".into(),
        },
    ]
}

#[tauri::command]
fn ensure_dashboard(
    state: tauri::State<'_, BackendState>,
    port: Option<u16>,
) -> Result<DashboardStatus, String> {
    let port = port.unwrap_or(DEFAULT_PORT);
    if port_open(port) {
        return Ok(status_from_port(port, "Dashboard already running"));
    }

    {
        let mut guard = state
            .dashboard_child
            .lock()
            .map_err(|_| "Dashboard lock poisoned".to_string())?;
        if guard.is_none() {
            let child = spawn_dashboard(&state.jarvis_root, port)?;
            *guard = Some(child);
        }
    }

    if wait_for_port(port, DASHBOARD_WAIT_SECS) {
        return Ok(status_from_port(
            port,
            "Dashboard started by Jarvis shell",
        ));
    }

    Ok(DashboardStatus {
        running: false,
        port,
        base_url: format!("http://127.0.0.1:{port}"),
        listener_active: None,
        message: "Dashboard process started but port not reachable yet".into(),
    })
}

#[tauri::command]
fn get_listener_status(state: tauri::State<'_, BackendState>) -> Result<ListenerStatus, String> {
    let out = run_python_script(&state.jarvis_root, "scripts/shell_daemon.py", &["status"])?;
    Ok(parse_listener_json(&out.stdout))
}

#[tauri::command]
fn get_voice_config(_state: tauri::State<'_, BackendState>, port: Option<u16>) -> VoiceConfig {
    let port = port.unwrap_or(DEFAULT_PORT);
    if let Some(v) = http_get_json(port, "/api/dashboard/voice-config") {
        return VoiceConfig {
            ok: v.get("ok").and_then(|x| x.as_bool()).unwrap_or(true),
            ptt_enabled: v
                .get("ptt_enabled")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            ptt_hotkey_display: v
                .get("ptt_hotkey_display")
                .and_then(|x| x.as_str())
                .unwrap_or("Ctrl+Shift+J")
                .to_string(),
            continuous_listening: v
                .get("continuous_listening")
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
            whisper_lazy_load: v
                .get("whisper_lazy_load")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            whisper_model: v
                .get("whisper_model")
                .and_then(|x| x.as_str())
                .unwrap_or("medium")
                .to_string(),
            wake_word: v
                .get("wake_word")
                .and_then(|x| x.as_str())
                .unwrap_or("Jarvis")
                .to_string(),
        };
    }
    VoiceConfig {
        ok: false,
        ptt_enabled: true,
        ptt_hotkey_display: "Ctrl+Shift+J".into(),
        continuous_listening: true,
        whisper_lazy_load: false,
        whisper_model: "medium".into(),
        wake_word: "Jarvis".into(),
    }
}

#[tauri::command]
fn start_listener(state: tauri::State<'_, BackendState>) -> Result<ListenerStatus, String> {
    let port = DEFAULT_PORT;
    if !port_open(port) {
        return Err(
            "Dashboard is not running on port 5050. Click «Restart dashboard» or wait for it to start, then try again.".into(),
        );
    }
    let out = run_python_script(&state.jarvis_root, "scripts/shell_daemon.py", &["start"])?;
    if !out.status.success() {
        return Err(format!(
            "start_listener failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    get_listener_status(state)
}

#[tauri::command]
fn stop_listener(state: tauri::State<'_, BackendState>) -> Result<ListenerStatus, String> {
    let out = run_python_script(&state.jarvis_root, "scripts/shell_daemon.py", &["stop"])?;
    if !out.status.success() {
        return Err(format!(
            "stop_listener failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    get_listener_status(state)
}

#[tauri::command]
fn open_settings(state: tauri::State<'_, BackendState>) -> Result<(), String> {
    let python = resolve_python(&state.jarvis_root);
    let script = state.jarvis_root.join("scripts/shell_settings.py");
    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .env("PYTHONPATH", state.jarvis_root.join("src"))
        .env("JARVIS_ROOT", &state.jarvis_root)
        .current_dir(&state.jarvis_root);
    apply_hidden(&mut cmd);
    cmd.spawn()
        .map_err(|e| format!("Failed to open settings: {e}"))?;
    Ok(())
}

#[tauri::command]
fn get_jarvis_root(state: tauri::State<'_, BackendState>) -> String {
    state.jarvis_root.display().to_string()
}

fn cafe_status_from_health() -> CafeAgentStatus {
    match http_get_json_host("127.0.0.1", CAFE_PORT, "/health") {
        Some(v) if v.get("status").and_then(|s| s.as_str()) == Some("ok") => CafeAgentStatus {
            online: true,
            status: "online".into(),
            sales_rows: v.get("sales_rows").and_then(|n| n.as_i64()),
        },
        Some(_) => CafeAgentStatus {
            online: false,
            status: "unexpected response".into(),
            sales_rows: None,
        },
        None => CafeAgentStatus {
            online: false,
            status: "offline".into(),
            sales_rows: None,
        },
    }
}

#[tauri::command]
fn get_cafe_agent_status() -> CafeAgentStatus {
    cafe_status_from_health()
}

#[tauri::command]
fn ensure_cafe_orchestrator(state: tauri::State<'_, BackendState>) -> Result<CafeAgentStatus, String> {
    if port_open(CAFE_PORT) {
        return Ok(cafe_status_from_health());
    }

    {
        let mut guard = state
            .cafe_child
            .lock()
            .map_err(|_| "Cafe orchestrator lock poisoned".to_string())?;
        if guard.is_none() {
            let child = spawn_cafe_orchestrator(&state.jarvis_root)?;
            *guard = Some(child);
        }
    }

    if wait_for_port(CAFE_PORT, CAFE_WAIT_SECS) {
        return Ok(cafe_status_from_health());
    }

    Err(
        "Cafe orchestrator started but port 8787 is not reachable yet (first run may compile Rust)".into(),
    )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let jarvis_root = resolve_jarvis_root();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendState {
            jarvis_root,
            dashboard_child: Mutex::new(None),
            cafe_child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            get_nav_routes,
            ensure_dashboard,
            get_listener_status,
            start_listener,
            stop_listener,
            open_settings,
            get_jarvis_root,
            get_cafe_agent_status,
            ensure_cafe_orchestrator,
            get_voice_config,
        ])
        .setup(|app| {
            let _ = app;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Jarvis shell");
}
