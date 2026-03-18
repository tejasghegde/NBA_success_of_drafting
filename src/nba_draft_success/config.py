from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def players_csv_candidates(self) -> tuple[Path, ...]:
        # Keep backwards compatibility with the current repo (players.csv at root),
        # while supporting a cleaner future layout (data/players.csv).
        return (self.data_dir / "players.csv", self.repo_root / "players.csv")


def default_paths() -> Paths:
    # src/nba_draft_success/config.py -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    return Paths(repo_root=repo_root)

