from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Paths, default_paths


def resolve_players_csv(paths: Paths | None = None) -> Path:
    paths = paths or default_paths()
    for candidate in paths.players_csv_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find players.csv. Expected one of: "
        + ", ".join(str(p) for p in paths.players_csv_candidates)
    )


def load_players_df(csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or resolve_players_csv()
    df = pd.read_csv(csv_path, header=0)
    return df

