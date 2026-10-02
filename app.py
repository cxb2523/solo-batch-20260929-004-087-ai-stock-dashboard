"""Entry point: ``python app.py`` serves the dashboard on port 8050."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> int:
    script = str(Path(__file__).with_name("stock_dashboard.py"))
    sys.argv = [
        "streamlit",
        "run",
        script,
        "--server.port=8050",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--global.developmentMode=false",
    ]
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
