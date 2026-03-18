from __future__ import annotations

import re
import string
import time
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.basketball-reference.com"


@dataclass(frozen=True)
class ScrapeConfig:
    timeout_s: float = 15.0
    sleep_s: float = 0.25
    user_agent: str = "nba-draft-success (educational project)"


def _get(url: str, cfg: ScrapeConfig) -> requests.Response:
    headers = {"User-Agent": cfg.user_agent}
    r = requests.get(url, headers=headers, timeout=cfg.timeout_s)
    r.raise_for_status()
    if cfg.sleep_s:
        time.sleep(cfg.sleep_s)
    return r


def player_basic_info(*, cfg: ScrapeConfig = ScrapeConfig(), min_active_from: int = 1990) -> pd.DataFrame:
    players: list[dict] = []
    for letter in string.ascii_lowercase:
        r = _get(f"{BASE_URL}/players/{letter}/", cfg)
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table")
        if not table:
            continue

        table_body = table.find("tbody")
        if not table_body:
            continue

        for row in table_body.find_all("tr"):
            link = row.find("a")
            if not link:
                continue

            cells = row.find_all("td")
            if len(cells) < 7:
                continue

            active_from = int(cells[0].text)
            if active_from < min_active_from:
                continue

            players.append(
                {
                    "url": link["href"],
                    "name": link.text,
                    "active_from": active_from,
                    "active_to": int(cells[1].text),
                    "position": cells[2].text,
                    "height": cells[3].text,
                    "weight": cells[4].text,
                    "birth_date": cells[5].text,
                    "college": cells[6].text,
                }
            )

    return pd.DataFrame(players)


def player_info(player_path: str, *, cfg: ScrapeConfig = ScrapeConfig()) -> dict:
    r = _get(f"{BASE_URL}{player_path}", cfg)
    soup = BeautifulSoup(r.text, "lxml")

    rnd = None
    pick = None
    for p in soup.find_all("p"):
        if "Draft:" not in p.text:
            continue
        s = p.text
        m_round = re.search(r"(\d+)(?:st|nd|rd|th)\s+round", s)
        m_pick = re.search(r"(\d+)(?:st|nd|rd|th)\s+pick", s)
        if m_round:
            rnd = int(m_round.group(1))
        if m_pick:
            pick = int(m_pick.group(1))

    table = soup.find("table")
    if not table:
        return {"Draft Round": rnd, "Pick Number": pick}

    career_row = None
    for row in table.find_all("tr"):
        th = row.find("th")
        if th and th.text == "Career":
            career_row = row
            break

    if not career_row:
        return {"Draft Round": rnd, "Pick Number": pick}

    cells = career_row.find_all("td")
    values: list[float | None] = []
    for c in cells:
        t = c.text.strip()
        if t == "" or t == "NBA":
            values.append(None)
        else:
            try:
                values.append(float(t))
            except ValueError:
                values.append(None)

    # Per-game table layout on basketball-reference (career totals line)
    # We keep field names consistent with existing CSV generation.
    mapping = {
        "Games Played": values[4],
        "Games Started": values[5],
        "Minutes Per Game": values[6],
        "Field Goals Per Game": values[7],
        "Field Goals Attempted Per Game": values[8],
        "Field Goal Percentage": values[9],
        "3 Point Field Goals Per Game": values[10],
        "3 Point Field Goals Attempted Per Game": values[11],
        "3 Point Field Goal Percentage": values[12],
        "2 Point Field Goals Per Game": values[13],
        "2 Point Field Goals Attempted Per Game": values[14],
        "2 Point Field Goal Percentage": values[15],
        "Effective Field Goal Percentage": values[16],
        "Free Throws Per Game": values[17],
        "Free Throws Attempted Per Game": values[18],
        "Free Throw Percentage": values[19],
        "Offensive Rebounds Per Game": values[20],
        "Defensive Rebounds Per Game": values[21],
        "Total Rebounds Per Game": values[22],
        "Assists Per Game": values[23],
        "Steals Per Game": values[24],
        "Blocks Per Game": values[25],
        "Turnovers Per Game": values[26],
        "Personal Fouls Per Game": values[27],
        "Points Per Game": values[28],
    }

    return {"Draft Round": rnd, "Pick Number": pick, **mapping}


def scrape_to_csv(out_csv: str = "players.csv", *, cfg: ScrapeConfig = ScrapeConfig()) -> None:
    basics = player_basic_info(cfg=cfg)
    records: list[dict] = []
    for path in basics["url"].tolist():
        records.append(player_info(path, cfg=cfg))
    stats = pd.DataFrame.from_records(records)
    df = pd.concat([basics.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    df.to_csv(out_csv, index=False)

