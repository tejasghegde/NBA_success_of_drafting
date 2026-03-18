# NBA Draft Success (Draft Round Prediction)

This repo predicts **NBA draft round** (Undrafted vs 1st vs 2nd) from career per-game stats using scikit-learn models.

## What’s in here

- **Data**: `players.csv` is committed so the project runs **offline** (no scraping required).
- **Code**: Python package in `src/nba_draft_success/` with a small CLI.
- **Legacy**: `FinalProject.py` is the original monolithic script (kept for reference).

## Quickstart (local, using uv)

Create a virtualenv (optional, recommended) and install deps:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/pip install uv
./.venv/bin/uv sync --frozen
```

Train/evaluate a model:

```bash
./.venv/bin/uv run nba-draft-success train --model rf
```

Try other models:

```bash
./.venv/bin/uv run nba-draft-success train --model knn
./.venv/bin/uv run nba-draft-success train --model dt
./.venv/bin/uv run nba-draft-success train --model mlp
```

## Quickstart (Docker)

Build:

```bash
docker build -t nba-draft-success .
```

Run:

```bash
docker run --rm nba-draft-success train --model rf
```

## CLI reference

```bash
nba-draft-success train \
  --model {knn,dt,rf,mlp} \
  --test-size 0.2 \
  --random-state 42 \
  --na-strategy {drop,median}
```

By default, it reads `players.csv` from the repo root. You can also pass a custom path:

```bash
nba-draft-success train --model rf --csv-path /path/to/players.csv
```

## Notes on reproducibility

- The train/test split uses `--random-state` and stratification (class-balance aware).
- Dependencies are pinned via `uv.lock` (use `uv sync --frozen`).

## (Optional) Regenerate the dataset via scraping

This project is designed to run without scraping, but a scraper module exists at `src/nba_draft_success/scrape.py`.
If you do use it, please be respectful (rate-limited requests, user-agent set).

