# launcher.py
import sys
import os
import threading
import webbrowser

# Set working directory to folder containing this exe
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(2.5, open_browser).start()

    from app import create_app          # ← correct: uses factory function
    from core.config import get_settings

    flask_app = create_app()
    s = get_settings()

    flask_app.run(
        host="127.0.0.1",
        port=s.FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )