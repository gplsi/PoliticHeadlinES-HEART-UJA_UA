from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import load_dataset
from .features import build_embeddings
from .metrics import evaluate_frame
from .ranker import train_ranker, predict, load_ranker

def _write_predictions(df, predictions, path, cfg):
    out = pd.DataFrame({"id": df["id"].astype(str), "task_1": predictions, "task_2": predictions})
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False)
    if "y_true" in df.columns:
        scored=df[["id","y_true"]].copy(); scored["prediction"]=predictions
        score=evaluate_frame(scored,"prediction",cfg.get("evaluation",{}).get("k",10),cfg.get("evaluation",{}).get("alpha",0.9))
        print(f"PA-nDCG: {score:.6f}")
    print(f"Saved predictions to {path}")

def run_similarity(cfg):
    df=load_dataset(cfg["data"]["input_csv"]); emb=build_embeddings(df,cfg); preds=[]
    for a,t in zip(emb.articles,emb.titles):
        scores=t@a; preds.append(" ".join(f"t{i+1}" for i in np.argsort(-scores)))
    _write_predictions(df,preds,cfg["data"]["output_csv"],cfg)

def run_train(cfg):
    df=load_dataset(cfg["data"]["train_csv"],require_gold=True); emb=build_embeddings(df,cfg)
    out=Path(cfg["data"]["output_dir"]); out.mkdir(parents=True,exist_ok=True); model=train_ranker(emb,cfg,str(out/"best_model.pt"))
    if cfg["data"].get("input_csv"):
        pred_df=load_dataset(cfg["data"]["input_csv"]); pred_emb=build_embeddings(pred_df,cfg); _write_predictions(pred_df,predict(model,pred_emb),out/"predictions.csv",cfg)

def run_predict(cfg, checkpoint):
    df=load_dataset(cfg["data"]["input_csv"]); emb=build_embeddings(df,cfg); model=load_ranker(checkpoint)
    out=Path(cfg["data"].get("output_dir","outputs/predict"))/"predictions.csv"; _write_predictions(df,predict(model,emb),out,cfg)
