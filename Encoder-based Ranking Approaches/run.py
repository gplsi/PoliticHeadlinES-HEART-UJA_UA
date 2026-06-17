#!/usr/bin/env python3
"""Single entry point for all encoder-based experiments in Section 3.2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from politicheadlines.pipeline import run_predict, run_similarity, run_train


def build_config(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "data": {
            "train_csv": args.train_csv,
            "input_csv": args.input_csv,
            "images_dir": args.images_dir,
            "output_dir": args.output_dir,
            "output_csv": str(Path(args.output_dir) / "predictions.csv"),
        },
        "text_encoder": {
            "model_name": args.text_model,
            "batch_size": args.encoder_batch_size,
            "title_max_length": 69,
        },
        "representation": {
            "strategy": args.strategy,
            "max_length": 512,
            "overlap": args.overlap,
            "pooling": args.pooling,
            "decay": 0.85,
        },
        "fusion": {
            "enabled": args.image_weight > 0,
            "image_model_name": "openai/clip-vit-large-patch14",
            "text_weight": 1.0 - args.image_weight,
            "image_weight": args.image_weight,
        },
        "ranker": {
            "hidden_dim": 256,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": args.epochs,
            "margin": 0.2,
            "batch_size": args.ranker_batch_size,
            "validation_ratio": 0.2,
        },
        "evaluation": {"k": 10, "alpha": 0.9},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PoliticHeadlinES 2026 encoder-based ranking experiments"
    )
    parser.add_argument("mode", choices=["similarity", "train", "predict"])
    parser.add_argument("--input-csv", default="data/dev_public.csv")
    parser.add_argument("--train-csv", default="data/train_public.csv")
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--output-dir", default="outputs/run")
    parser.add_argument("--checkpoint", default="outputs/run/best_model.pt")

    parser.add_argument("--strategy", choices=["first", "token_chunks", "sentence_chunks"], default="first")
    parser.add_argument("--pooling", choices=["mean", "weighted"], default="mean")
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--image-weight", type=float, default=0.0)

    parser.add_argument("--text-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--ranker-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.image_weight <= 1.0:
        parser.error("--image-weight must be between 0 and 1")

    cfg = build_config(args)
    if args.mode == "similarity":
        run_similarity(cfg)
    elif args.mode == "train":
        run_train(cfg)
    else:
        run_predict(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
