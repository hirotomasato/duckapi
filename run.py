"""Bootstrap the server: check the port, build the venv, seed .env, run uvicorn.

Replaces the old start.bat/stop.bat, whose hardcoded interpreter path pointed at
another user's profile and so only worked by falling through to PATH.

Usage:
    python run.py                  # start, open the dashboard when healthy
    python run.py --no-browser     # start headless
    python run.py --port 9000
    python run.py --reload         # uvicorn autoreload (dev)
    python run.py --check          # setup + port check only, don't boot
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
LOG = ROOT / "server.log"
DEFAULT_PORT = 8080


def venv_python() -> Path | None:
    """The venv interpreter, or None if the venv was never created."""
    exe = VENV / "Scripts" / "python.exe"
    if exe.exists():
        return exe
    exe = VENV / "bin" / "python"  # non-Windows layout
    return exe if exe.exists() else None


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    """True if something already accepts connections on `port`.

    Probes by connecting rather than parsing netstat: the question is "will
    uvicorn fail to bind", and only a real connect answers that.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def ensure_venv() -> Path:
    """Return the venv interpreter, creating it from example.env's requirements
    on first run. Idempotent: an existing venv is reused untouched."""
    if py := venv_python():
        return py

    print("Creating venv and installing dependencies (first run only)...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = venv_python()
    if not py:  # pragma: no cover - venv module just created it
        raise RuntimeError(f"venv created but no interpreter found under {VENV}")
    subprocess.run(
        [str(py), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )
    return py


def ensure_env() -> None:
    """Seed .env from example.env when absent. Never overwrites an existing one -
    it holds the user's API key and proxy settings."""
    env = ROOT / ".env"
    if env.exists():
        return
    shutil.copyfile(ROOT / "example.env", env)
    print(f"Created .env from example.env (open access, no API key set).")


def wait_healthy(port: int, timeout: float = 90.0) -> bool:
    """Poll /health until the server answers or `timeout` elapses.

    The dashboard is opened only after this passes, so the browser never lands
    on a connection-refused page during the ~10s prewarm.
    """
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def serve(py: Path, port: int, reload: bool) -> int:
    """Run uvicorn as a child process, teeing its output to server.log and the
    console. A child (rather than uvicorn.run in-process) keeps Ctrl-C simple:
    the signal reaches uvicorn directly and this process just reaps it.
    """
    cmd = [
        str(py), "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0", "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    print(f"Starting uvicorn on port {port}...")
    print(f"  dashboard : http://localhost:{port}/")
    print(f"  log file  : {LOG}")

    # Truncate rather than append: one run's log is one run's story, and a
    # stale tail from a crashed run is actively misleading while debugging.
    with LOG.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return proc.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            proc.terminate()
            try:
                return proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Start the DuckAI2API server.")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", DEFAULT_PORT)))
    ap.add_argument("--no-browser", action="store_true", help="don't open the dashboard")
    ap.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")
    ap.add_argument("--check", action="store_true", help="setup + port check only, don't boot")
    args = ap.parse_args()

    if port_busy(args.port):
        print(f"Port {args.port} is already in use - something else is listening there.")
        print(f"Close it first, or start on another port:  python run.py --port {args.port + 1}")
        return 1

    ensure_env()
    py = ensure_venv()
    print(f"Interpreter: {py}")

    if args.check:
        print(f"Port {args.port} is free. Setup OK.")
        return 0

    # Open the dashboard only once the server actually answers, so the browser
    # never lands on a connection-refused page during the ~10s prewarm.
    if not args.no_browser and not args.reload:
        threading.Thread(
            target=lambda: (wait_healthy(args.port) and webbrowser.open(f"http://localhost:{args.port}/")),
            daemon=True,
        ).start()

    return serve(py, args.port, args.reload)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
