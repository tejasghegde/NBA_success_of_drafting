from __future__ import annotations

import pandas as pd


META_COLUMNS = {
    "index",
    "url",
    "name",
    "active_from",
    "active_to",
    "position",
    "college",
    "height",
    "weight",
    "birth_date",
}

TARGET_DRAFT_ROUND = "Draft Round"
PICK_NUMBER = "Pick Number"


def _normalize_draft_round(series: pd.Series) -> pd.Series:
    # In the current dataset, undrafted is encoded as -1. We normalize it to 0.
    # Keep 1 and 2 as-is. Anything else becomes NA and is filtered out by default.
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({-1: 0})
    return s


def prepare_round_classification(
    df: pd.DataFrame,
    *,
    drop_meta: bool = True,
    drop_pick_number: bool = True,
    na_strategy: str = "drop",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X/y for predicting draft round class:
    - 0: Undrafted
    - 1: First round
    - 2: Second round
    """
    if TARGET_DRAFT_ROUND not in df.columns:
        raise ValueError(f"Missing required column {TARGET_DRAFT_ROUND!r}")

    work = df.copy()
    work[TARGET_DRAFT_ROUND] = _normalize_draft_round(work[TARGET_DRAFT_ROUND])

    # Keep only {0,1,2}
    work = work[work[TARGET_DRAFT_ROUND].isin([0, 1, 2])]

    if drop_meta:
        cols_to_drop = [c for c in META_COLUMNS if c in work.columns]
        if cols_to_drop:
            work = work.drop(columns=cols_to_drop)

    if drop_pick_number and PICK_NUMBER in work.columns:
        work = work.drop(columns=[PICK_NUMBER])

    # Convert all remaining feature columns to numeric where possible.
    y = work[TARGET_DRAFT_ROUND].astype(int)
    X = work.drop(columns=[TARGET_DRAFT_ROUND])

    # Keep numeric-only features (the dataset should already be numeric, but this is safer).
    X = X.apply(pd.to_numeric, errors="coerce")

    if na_strategy == "drop":
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
    elif na_strategy == "median":
        X = X.fillna(X.median(numeric_only=True))
    else:
        raise ValueError("na_strategy must be 'drop' or 'median'")

    return X, y

