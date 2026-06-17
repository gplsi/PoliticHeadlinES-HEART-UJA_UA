from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS = [f"t{i}" for i in range(1, 11)]

def load_dataset(path: str | Path, require_gold: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    required = ["id", "article_body", "image_hash", *TITLE_COLS]
    if require_gold:
        required.append("y_true")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def titles_from_row(row: pd.Series) -> list[str]:
    return [str(row[c]) if pd.notna(row[c]) else "" for c in TITLE_COLS]

def parse_ranking(value: object) -> list[str]:
    valid = set(TOKENS)
    seen: set[str] = set()
    out: list[str] = []
    for token in str(value).strip().split():
        if token in valid and token not in seen:
            out.append(token); seen.add(token)
    return out

def complete_ranking(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set(); out: list[str] = []
    for token in tokens:
        if token in TOKENS and token not in seen:
            out.append(token); seen.add(token)
    out.extend(t for t in TOKENS if t not in seen)
    return out
