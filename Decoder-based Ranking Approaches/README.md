# PoliticHeadlinES 2026 — Decoder-based Ranking Approaches

Practical guide for running the few-shot headline ranking system.

## Quick Start

```bash
pip install -r requirements.txt
python src/task1/generate_rankings.py --config configs/task1/config.yaml
```

Before running:
1. Place the dataset under `data/`
2. Edit your config: set `api.base_url`, `api.api_key`, and `api.model`

## Requirements

- Python 3.13+
- OpenAI-compatible chat API endpoint (vLLM, TGI, OpenRouter, etc.)
- *(Optional)* OpenAI-compatible embedding API if using `semantic` or `hybrid_rrf` retrieval

## Dataset Setup

Place the PoliticHeadlinES 2026 data under `data/`:

```
data/
├── development_phase_dataset/
│   ├── dev_public.csv        # 50 dev articles (with ground truth)
│   └── train_public.csv      # 100 training articles (with ground truth)
├── test_public/
│   ├── test_public.csv       # Blind test set (labels withheld)
│   └── train_public.csv      # Extra training data from test phase
└── images/                   # Article images (required for Task 2)
```

## Running

### Task 1 — Text-only

```bash
python src/task1/generate_rankings.py --config configs/task1/config_Gemma-4-31B-IT-test.yaml
```

### Task 2 — Text + Image

```bash
python src/task2/generate_rankings.py --config configs/task2/config_Gemma-4-31B-IT_test.yaml
```

## Trial vs. Test

| Config         | Input CSV                                       | Training CSVs                                                        | Evaluation               |
| -------------- | ----------------------------------------------- | -------------------------------------------------------------------- | ------------------------ |
| `*-trial.yaml` | `data/development_phase_dataset/dev_public.csv` | `train_public.csv` (single)                                          | PA-nDCG on dev (enabled) |
| `*-test.yaml`  | `data/test_public/test_public.csv`              | `dev_public.csv`, `train_public.csv`, `test_public/train_public.csv` | Disabled (blind)         |

## Configuration Overview

Key sections in the YAML configs:

- **`api`** — endpoint URL, model name, API key, timeout, thinking mode
- **`inference`** — `temperature: 0.0`, `max_tokens: 4096`, `seed: 42`
- **`data`** — input CSV, training CSVs, output paths
- **`fewshot_selection`** — similarity metric: `tfidf`, `semantic`, or `hybrid_rrf`; number of examples; threshold
- **`task_1` / `task_2`** — enable/disable each task and choose per-task models
- **`prompt`** — Spanish system and user prompt templates
- **`guided_json`** — (optional) force JSON output for parse reliability
- **`evaluation`** — PA-nDCG against a reference CSV (trial mode only)

## Project Structure

```
.
├── src/
│   ├── task1/
│   │   ├── generate_rankings.py      # Main pipeline
│   │   └── similarity_utils.py       # TF-IDF / semantic / RRF retrieval
│   └── task2/
│       ├── generate_rankings.py      # Multimodal pipeline
│       └── similarity_utils.py       # Same retrieval utilities
├── configs/
│   ├── task1/
│   │   ├── config.yaml               # Base template
│   │   ├── config_Gemma-4-31B-IT-trial.yaml
│   │   ├── config_Gemma-4-31B-IT-test.yaml
│   │   ├── config_Qwen3.6-35B-A3B-trial.yaml
│   │   └── config_Qwen3.6-35B-A3B-test.yaml
│   └── task2/
│       ├── config.yaml
│       ├── config_Gemma-4-31B-IT_trial.yaml
│       ├── config_Gemma-4-31B-IT_test.yaml
│       ├── config_Qwen3.6-35B-A3B_trial.yaml
│       └── config_Qwen3.6-35B-A3B_test.yaml
├── requirements.txt
├── README.md                         # This file
└── PAPER.md                          # Paper overview
```


See [PAPER.md](PAPER.md) for citations.
