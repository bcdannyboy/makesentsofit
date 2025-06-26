from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def load_analysis(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # If the JSON has top-level "posts" key, use that
    if isinstance(data, dict) and "posts" in data:
        data = data["posts"]
    return pd.DataFrame(data)


@st.cache_data(ttl=3600)
def load_user_network(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

