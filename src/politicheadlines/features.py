from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .data import TITLE_COLS
from .encoders import TextEncoder, ImageEncoder, align_dimension, find_image

@dataclass
class Embeddings:
    ids: list[str]
    articles: np.ndarray
    titles: np.ndarray
    gold: list[str] | None

def build_embeddings(df: pd.DataFrame, cfg: dict) -> Embeddings:
    text_cfg, rep, fusion = cfg["text_encoder"], cfg["representation"], cfg.get("fusion", {})
    encoder = TextEncoder(text_cfg["model_name"], text_cfg.get("batch_size", 8))
    articles = encoder.encode_articles(df["article_body"].fillna("").astype(str).tolist(), rep["strategy"], rep["max_length"], rep.get("overlap", 0), rep.get("pooling", "mean"), rep.get("decay", 0.85))
    flat_titles = [str(v) if pd.notna(v) else "" for row in df[TITLE_COLS].itertuples(index=False, name=None) for v in row]
    title_flat = encoder.encode(flat_titles, text_cfg.get("title_max_length", 69))
    titles = title_flat.reshape(len(df), 10, -1)
    if fusion.get("enabled", False) and fusion.get("image_weight", 0) > 0:
        img_encoder = ImageEncoder(fusion["image_model_name"])
        paths = [find_image(cfg["data"]["images_dir"], x) for x in df["image_hash"]]
        images = align_dimension(img_encoder.encode(paths), articles.shape[1])
        wt, wi = float(fusion.get("text_weight", 0.9)), float(fusion.get("image_weight", 0.1))
        articles = wt * articles + wi * images
        articles /= np.clip(np.linalg.norm(articles, axis=1, keepdims=True), 1e-12, None)
    gold = df["y_true"].astype(str).tolist() if "y_true" in df.columns else None
    return Embeddings(df["id"].astype(str).tolist(), articles.astype(np.float32), titles.astype(np.float32), gold)
