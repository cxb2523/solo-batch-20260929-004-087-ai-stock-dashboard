"""Entry point: serve the dashboard on port 8050 via ``python app.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8050
DASHBOARD = Path(__file__).with_name("stock_dashboard.py")


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
        f"--server.port={PORT}",
        f"--server.address={HOST}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
