#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::OpenOptions;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::webview::{DownloadEvent, WebviewWindowBuilder};
use tauri::{Manager, WindowEvent};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

fn log_line(message: &str) {
    let base = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string());
    let directory = PathBuf::from(base).join("DataBridgeAI");
    let _ = std::fs::create_dir_all(&directory);
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("tauri_launcher.log"))
    {
        let _ = writeln!(file, "{}", message);
    }
}

fn first_existing(paths: Vec<PathBuf>) -> Option<PathBuf> {
    for path in paths {
        log_line(&format!("checking {}", path.display()));
        if path.is_file() {
            log_line(&format!("found {}", path.display()));
            return Some(path);
        }
    }
    None
}

fn reserve_available_port() -> Option<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

fn spawn_streamlit(app_handle: &tauri::AppHandle, port: u16) -> Option<Child> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let resource_dir = app_handle.path().resource_dir().ok();

    log_line(&format!("exe_dir={}", exe_dir.display()));
    if let Some(ref path) = resource_dir {
        log_line(&format!("resource_dir={}", path.display()));
    }

    let mut python_candidates = Vec::new();
    let mut launcher_candidates = Vec::new();

    if let Some(path) = resource_dir.clone() {
        python_candidates.push(path.join("python").join("pythonw.exe"));
        python_candidates.push(path.join("python").join("python.exe"));
        python_candidates.push(path.join("python").join("Scripts").join("pythonw.exe"));
        python_candidates.push(path.join("python").join("Scripts").join("python.exe"));
        launcher_candidates.push(path.join("app").join("launch_streamlit.py"));
    }

    python_candidates.extend(vec![
        exe_dir.join("python").join("pythonw.exe"),
        exe_dir.join("python").join("python.exe"),
        exe_dir.join("resources").join("python").join("pythonw.exe"),
        exe_dir.join("resources").join("python").join("python.exe"),
        exe_dir.join("_up_").join("python").join("pythonw.exe"),
        exe_dir.join("_up_").join("python").join("python.exe"),
        exe_dir
            .join("_up_")
            .join("_up_")
            .join("python")
            .join("pythonw.exe"),
        exe_dir
            .join("_up_")
            .join("_up_")
            .join("python")
            .join("python.exe"),
    ]);

    launcher_candidates.extend(vec![
        exe_dir.join("app").join("launch_streamlit.py"),
        exe_dir
            .join("resources")
            .join("app")
            .join("launch_streamlit.py"),
        exe_dir
            .join("_up_")
            .join("app")
            .join("launch_streamlit.py"),
        exe_dir
            .join("_up_")
            .join("_up_")
            .join("app")
            .join("launch_streamlit.py"),
    ]);

    let python = first_existing(python_candidates)?;
    let launcher = first_existing(launcher_candidates)?;
    let app_dir = launcher.parent()?.to_path_buf();

    log_line(&format!(
        "START python={} launcher={} port={}",
        python.display(),
        launcher.display(),
        port
    ));

    let mut command = Command::new(python);
    command
        .arg(launcher)
        .current_dir(app_dir)
        .env("DATABRIDGE_PORT", port.to_string())
        .env("PYTHONUTF8", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }

    match command.spawn() {
        Ok(child) => Some(child),
        Err(error) => {
            log_line(&format!("ERROR spawning launcher: {}", error));
            None
        }
    }
}

fn terminate_process_tree(child: &mut Child) {
    if child.try_wait().ok().flatten().is_some() {
        return;
    }

    #[cfg(target_os = "windows")]
    {
        let pid = child.id().to_string();
        let mut command = Command::new("taskkill");
        command
            .args(["/PID", &pid, "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(CREATE_NO_WINDOW);
        let _ = command.status();
    }

    let _ = child.kill();
    let _ = child.wait();
}

fn main() {
    let child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let child_setup = Arc::clone(&child);
    let child_exit = Arc::clone(&child);

    tauri::Builder::default()
        .setup(move |app| {
            // The main window is created manually so we can intercept WebView downloads
            // and ask for an explicit destination instead of silently using Downloads.
            WebviewWindowBuilder::from_config(app.handle(), &app.config().app.windows[0])?
                .on_download(|_webview, event| match event {
                    DownloadEvent::Requested { url, destination } => {
                        let suggested_name = destination
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("databridge-download")
                            .to_string();
                        let suggested_directory = destination.parent().map(PathBuf::from);

                        let mut dialog = rfd::FileDialog::new()
                            .set_title("Save DataBridge AI download as")
                            .set_file_name(suggested_name);
                        if let Some(directory) = suggested_directory {
                            dialog = dialog.set_directory(directory);
                        }

                        match dialog.save_file() {
                            Some(chosen_path) => {
                                log_line(&format!(
                                    "DOWNLOAD requested={} destination={}",
                                    url,
                                    chosen_path.display()
                                ));
                                *destination = chosen_path;
                                true
                            }
                            None => {
                                log_line(&format!("DOWNLOAD cancelled {}", url));
                                false
                            }
                        }
                    }
                    DownloadEvent::Finished { url, path, success } => {
                        log_line(&format!(
                            "DOWNLOAD finished={} path={:?} success={}",
                            url, path, success
                        ));
                        true
                    }
                    _ => true,
                })
                .build()?;

            let handle = app.handle().clone();
            thread::spawn(move || {
                let Some(port) = reserve_available_port() else {
                    log_line("ERROR no loopback port is available");
                    return;
                };
                let process = spawn_streamlit(&handle, port);
                if process.is_none() {
                    log_line("ERROR Streamlit launcher did not start");
                    return;
                }
                if let Ok(mut guard) = child_setup.lock() {
                    *guard = process;
                }

                for _ in 0..240 {
                    thread::sleep(Duration::from_millis(500));
                    let exited = child_setup
                        .lock()
                        .ok()
                        .and_then(|mut guard| guard.as_mut().and_then(|proc| proc.try_wait().ok().flatten()))
                        .is_some();
                    if exited {
                        log_line("ERROR Streamlit launcher exited before readiness");
                        return;
                    }
                    if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                        if let Some(window) = handle.get_webview_window("main") {
                            let url = format!("http://127.0.0.1:{}", port);
                            match url.parse() {
                                Ok(parsed) => {
                                    let _ = window.navigate(parsed);
                                    log_line(&format!("READY {}", url));
                                }
                                Err(error) => log_line(&format!("ERROR invalid local URL: {}", error)),
                            }
                        }
                        return;
                    }
                }
                log_line("ERROR Streamlit startup timeout");
            });
            Ok(())
        })
        .on_window_event(move |_window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                if let Ok(mut guard) = child_exit.lock() {
                    if let Some(process) = guard.as_mut() {
                        terminate_process_tree(process);
                    }
                    *guard = None;
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running DataBridge AI");
}
