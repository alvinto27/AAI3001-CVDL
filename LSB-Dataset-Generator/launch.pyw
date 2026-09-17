"""Report startup failures even when launched without a console."""
from pathlib import Path
import traceback

try:
    from gui import App
    App().mainloop()
except Exception:
    error = traceback.format_exc()
    log = Path(__file__).with_name("startup-error.log")
    log.write_text(error, encoding="utf-8")
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, f"The application could not start.\n\nSee {log}\n\n{error[-1200:]}", "LSB Dataset Generator", 16)
