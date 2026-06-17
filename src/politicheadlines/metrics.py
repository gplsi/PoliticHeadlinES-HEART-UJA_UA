from __future__ import annotations
import math
import pandas as pd
from .data import parse_ranking

def _ndcg(pred: list[str], gold: list[str], k: int) -> float:
    pos = {x: i for i, x in enumerate(gold)}
    def dcg(order: list[str]) -> float:
        total = 0.0
        for i, item in enumerate(order[:k]):
            if item in pos:
                rel = len(gold) - pos[item]
                total += (2.0 ** rel - 1.0) / math.log2(i + 2)
        return total
    ideal = dcg(gold)
    return dcg(pred) / ideal if ideal else 0.0

def pa_ndcg(pred: list[str], gold: list[str], k: int = 10, alpha: float = 0.9) -> float:
    if not pred or not gold or pred[0] != gold[0]:
        return 0.0
    gold_rest = [x for x in gold if x != gold[0]]
    pred_rest = [x for x in pred if x != gold[0]]
    return alpha + (1.0 - alpha) * _ndcg(pred_rest, gold_rest, max(k - 1, 1))

def evaluate_frame(df: pd.DataFrame, pred_col: str, k: int = 10, alpha: float = 0.9) -> float:
    if "y_true" not in df.columns:
        raise ValueError("Dataset has no y_true column")
    scores = [pa_ndcg(parse_ranking(p), parse_ranking(g), k, alpha) for p, g in zip(df[pred_col], df["y_true"])]
    return float(sum(scores) / len(scores)) if scores else 0.0
