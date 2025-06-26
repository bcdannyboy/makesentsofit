"""Streamlit dashboard package for MakeSenseOfIt."""

from pathlib import Path
from . import streamlit_app


def launch_dashboard(analysis_file: Path, user_network_file: Path | None = None) -> None:
    """Launch the Streamlit dashboard as a subprocess."""
    import subprocess

    args = ["streamlit", "run", str(Path(__file__).parent / "streamlit_app.py"), "--", "--analysis", str(analysis_file)]
    if user_network_file:
        args += ["--user-network", str(user_network_file)]
    subprocess.Popen(args)

