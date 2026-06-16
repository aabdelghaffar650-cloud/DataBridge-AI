# Per-user installer fix

This build changes the NSIS installer mode from perMachine to perUser.

Why:
- Avoids installing into `C:\Program Files`
- Avoids admin permission issues
- Reduces locked-file install errors with bundled Python packages
- Installs under the user's local app programs directory

Build:
1. Run `prepare_embedded_python.bat`
2. Run `build_desktop_clean.bat`

Before installing this version:
- Uninstall old DataBridge AI
- Delete `C:\Program Files\DataBridge AI` if it remains
- Then install the new per-user installer
