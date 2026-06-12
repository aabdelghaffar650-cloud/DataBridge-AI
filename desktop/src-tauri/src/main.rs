
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{Manager, WindowEvent};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

fn log_line(msg: &str) {
    let base = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
    let dir = PathBuf::from(base).join("DataBridgeAI");
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(dir.join("tauri_launcher.log")) {
        let _ = writeln!(f, "{}", msg);
    }
}

fn first_existing(paths: Vec<PathBuf>) -> Option<PathBuf> {
    for p in paths {
        log_line(&format!("checking {}", p.display()));
        if p.exists() {
            log_line(&format!("found {}", p.display()));
            return Some(p);
        }
    }
    None
}

fn spawn_streamlit(app_handle: &tauri::AppHandle) -> Option<Child> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let res_dir = app_handle.path().resource_dir().ok();

    log_line(&format!("exe_dir={}", exe_dir.display()));
    if let Some(ref r) = res_dir { log_line(&format!("resource_dir={}", r.display())); }

    let mut py = Vec::new();
    let mut launch = Vec::new();

    if let Some(r) = res_dir.clone() {
        // Prefer pythonw.exe to prevent a console window.
        py.push(r.join("python").join("Scripts").join("pythonw.exe"));
        py.push(r.join("python").join("pythonw.exe"));
        py.push(r.join("python").join("Scripts").join("python.exe"));
        py.push(r.join("python").join("python.exe"));
        launch.push(r.join("app").join("launch_streamlit.py"));
    }

    py.extend(vec![
        exe_dir.join("python").join("Scripts").join("pythonw.exe"),
        exe_dir.join("python").join("pythonw.exe"),
        exe_dir.join("python").join("Scripts").join("python.exe"),
        exe_dir.join("_up_").join("python").join("Scripts").join("pythonw.exe"),
        exe_dir.join("_up_").join("python").join("pythonw.exe"),
        exe_dir.join("_up_").join("python").join("Scripts").join("python.exe"),
        exe_dir.join("_up_").join("_up_").join("python").join("Scripts").join("pythonw.exe"),
        exe_dir.join("_up_").join("_up_").join("python").join("pythonw.exe"),
        exe_dir.join("_up_").join("_up_").join("python").join("Scripts").join("python.exe"),
        exe_dir.join("resources").join("python").join("Scripts").join("pythonw.exe"),
        exe_dir.join("resources").join("python").join("pythonw.exe"),
        exe_dir.join("resources").join("python").join("Scripts").join("python.exe"),
    ]);

    launch.extend(vec![
        exe_dir.join("app").join("launch_streamlit.py"),
        exe_dir.join("_up_").join("app").join("launch_streamlit.py"),
        exe_dir.join("_up_").join("_up_").join("app").join("launch_streamlit.py"),
        exe_dir.join("resources").join("app").join("launch_streamlit.py"),
    ]);

    let python = first_existing(py)?;
    let launcher = first_existing(launch)?;
    let app_dir = launcher.parent()?.to_path_buf();

    log_line(&format!("START python={} launcher={}", python.display(), launcher.display()));

    let mut cmd = Command::new(python);
    cmd.arg(launcher)
        .current_dir(app_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    cmd.spawn().ok()
}

fn main() {
    let child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let child_setup = child.clone();
    let child_exit = child.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(move |app| {
            let handle = app.handle().clone();
            thread::spawn(move || {
                let proc = spawn_streamlit(&handle);
                if let Ok(mut guard) = child_setup.lock() { *guard = proc; }

                for _ in 0..120 {
                    thread::sleep(Duration::from_secs(1));
                    if std::net::TcpStream::connect("127.0.0.1:8501").is_ok() {
                        if let Some(w) = handle.get_webview_window("main") {
                            let _ = w.navigate("http://127.0.0.1:8501".parse().unwrap());
                        }
                        break;
                    }
                }
            });
            Ok(())
        })
        .on_window_event(move |_window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                if let Ok(mut guard) = child_exit.lock() {
                    if let Some(child) = guard.as_mut() { let _ = child.kill(); }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running DataBridge AI");
}
