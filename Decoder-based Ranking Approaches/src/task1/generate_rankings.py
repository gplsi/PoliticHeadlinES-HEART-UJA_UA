"""
Smart Few-Shot Ranking Pipeline — Task 1 (text-only).

Pipeline Overview
-----------------
This script generates ranked lists of candidate headlines for news articles using
a few-shot LLM approach.  For every test article it:

1. **Few-shot selection** – picks the most similar training examples via TF-IDF,
   semantic embeddings, or a hybrid RRF fusion.  The selected examples are
   injected into the prompt so the model learns the expected output format
   (space-separated ``t1``..``t10`` tokens).

2. **LLM inference** – calls an OpenAI-compatible chat API to obtain a ranking.
   The call supports:
   * Plain free-text generation (parsed with regex).
   * Guided JSON mode (``response_format={"type":"json_object"}``) for
     higher parse reliability.
   * Optional multimodal input (image + text) for Task 2.

3. **PA-nDCG evaluation** – compares predictions against a reference CSV using
   the official *Position-Aware nDCG* metric.  The metric imposes a hard top-1
   condition: if the predicted best title differs from the ground-truth best
t   title the score is zeroed; otherwise it is ``alpha + (1-alpha)*nDCG@k`` on
   the remaining positions.

Key Design Decisions
--------------------
* **Incremental saving** – every processed row is appended to the output CSV
  immediately.  If the run crashes, progress is preserved and can be resumed
  without re-inferencing already-completed rows.
* **Deterministic seed** – a fixed RNG seed (configurable) makes few-shot
  selection and any stochastic prompt ordering reproducible across runs.
* **Graceful degradation** – if the LLM returns malformed text, empty output, or
  invalid JSON, the pipeline falls back to a deterministic TF-IDF overlap
  heuristic so every article still receives a valid ``t1..t10`` ranking.
* **CoT tag stripping** – some reasoning models wrap thinking content in
  ``<thinking>…</thinking>`` blocks; these are removed before parsing so they
  do not corrupt the ranking extraction.

How to Run
----------
    python generate_rankings.py --config config.yaml

The YAML config controls model endpoints, hyper-parameters, data paths, few-shot
settings, and evaluation flags.  See ``config.yaml`` in the repository for a
annotated example.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import random
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

from similarity_utils import load_training_data, cache_embeddings, select_similar_examples, build_fewshot_examples_for_prompt

# Regex matching t1..t10 tokens (case-insensitive).
RANKING_TOKEN_PATTERN = re.compile(r"\bt(?:10|[1-9])\b", re.IGNORECASE)
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]
N_COLS = 10

DEBUG_MODEL_OUTPUT_ENABLED = False


def debug_print_model_output(raw_text: str, attempt: int) -> None:
    """Print raw model output when debug mode is enabled."""
    if not DEBUG_MODEL_OUTPUT_ENABLED:
        return
    print(f"[MODEL_RAW_OUTPUT][attempt={attempt}] {raw_text.strip()}")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """Resolve a relative path against the directory that contains the config file."""
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_training_data_from_csvs(training_csvs: list[Path]) -> list[dict[str, Any]]:
    """Load and merge multiple training CSVs, deduplicating by ``id``."""
    merged_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    raw_total = 0

    for csv_path in training_csvs:
        print(f"Loading training data from {csv_path}...")
        rows = load_training_data(csv_path)
        raw_total += len(rows)

        for row in rows:
            row_id = str(row.get("id", "")).strip()
            if row_id:
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
            merged_rows.append(row)

    print(
        f"Loaded {len(merged_rows)} unique training examples "
        f"from {len(training_csvs)} files ({raw_total} raw rows)"
    )
    return merged_rows


def normalize_ranking(raw_text: str) -> str:
    """Extract t1..t10 tokens from raw text, deduplicate, and pad to 10.

    The model may output an incomplete list (e.g. only the first few tokens)
    or may repeat tokens.  We therefore:
    1. Scan the raw text for every ``t1``..``t10`` occurrence.
    2. Deduplicate while preserving the order of first appearance.
    3. **Pad missing tokens** at the end in ascending order (t1, t2, …).
       This guarantees a complete, valid ranking even when the model truncates
       its response, preventing downstream evaluation crashes.
    """
    found = [m.group(0).lower() for m in RANKING_TOKEN_PATTERN.finditer(raw_text or "")]
    unique: list[str] = []
    used = set()

    for token in found:
        if token not in used:
            unique.append(token)
            used.add(token)

    # Append any tokens the model omitted so the ranking is always length 10.
    for i in range(1, 11):
        token = f"t{i}"
        if token not in used:
            unique.append(token)

    return " ".join(unique[:10])


def tokenize(text: str) -> set[str]:
    """Tokenize text for TF-IDF fallback."""
    return set(re.findall(r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]{3,}", text.lower()))


def clean_thinking_tags(text: str) -> str:
    """Strip CoT reasoning tags (<thinking>…</thinking>) from model output."""
    if not text:
        return text
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<thinking>", "").replace("</thinking>", "")
    return cleaned.strip()


def fallback_ranking(article: str, titles: list[str]) -> str:
    """Heuristic fallback: rank titles by TF-IDF token overlap with the article."""
    article_tokens = tokenize(article)
    scored: list[tuple[int, int]] = []

    for i, title in enumerate(titles, start=1):
        title_tokens = tokenize(title)
        overlap = len(article_tokens.intersection(title_tokens))
        scored.append((i, overlap))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return " ".join(f"t{i}" for i, _ in scored)


def extract_json_object_text(raw_text: str) -> str | None:
    """Extract the first JSON object from raw text, handling markdown fences."""
    if not raw_text:
        return None

    text = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL)
    text = text.replace("<thinking>", "").replace("</thinking>", "").strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return text[idx : idx + end]

    return None


def extract_ranking_from_json(raw_text: str, ranking_key: str = "ranking") -> str | None:
    """Validate and extract a t1..t10 ranking from a JSON payload.

    Expects the JSON object to contain a ``ranking`` field (or custom key)
    whose value is either a space-separated string or a list of tokens.
    """
    if not raw_text:
        return None

    payload = extract_json_object_text(raw_text)
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    ranking_value = data.get(ranking_key)
    if ranking_value is None:
        return None

    if isinstance(ranking_value, list):
        ranking_text = " ".join(str(x) for x in ranking_value)
    elif isinstance(ranking_value, str):
        ranking_text = ranking_value
    else:
        return None

    normalized = normalize_ranking(ranking_text)
    tokens = normalized.split()

    if len(tokens) != 10 or len(set(tokens)) != 10:
        return None

    expected = {f"t{i}" for i in range(1, 11)}
    if set(tokens) != expected:
        return None

    return normalized


def build_user_prompt(template: str, article: str, titles: list[str]) -> str:
    """Format article and titles into the user prompt template."""
    mapping = {"article": article}
    for i, title in enumerate(titles, start=1):
        mapping[f"title_{i}"] = title
    return template.format(**mapping)


def image_to_data_url(image_path: Path) -> str:
    """Convert image file to data URL for API."""
    suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
    mime_type = "jpeg" if suffix == "jpg" else suffix
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{mime_type};base64,{b64}"


def find_image_path(images_dir: Path, image_hash: Any) -> Path | None:
    """Resolve image path from hash trying common extensions."""
    if image_hash is None:
        return None

    image_hash_str = str(image_hash).strip()
    if not image_hash_str:
        return None

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{image_hash_str}{ext}"
        if candidate.exists():
            return candidate

    candidate = images_dir / image_hash_str
    if candidate.exists():
        return candidate

    return None


def ask_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: int,
    sleep_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    image_data_url: str | None = None,
    response_format: dict[str, Any] | None = None,
    enable_thinking: bool = True,
) -> str:
    """Call the LLM API with retry logic and optional JSON mode.

    Args:
        client: OpenAI-compatible client instance.
        model: Model identifier.
        system_prompt: System-level instructions.
        user_text: User message text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens to generate.
        timeout_seconds: API request timeout.
        sleep_seconds: Delay after a successful call (rate-limiting).
        max_retries: Number of retries on failure.
        retry_backoff_seconds: Delay between retries.
        image_data_url: Optional base64 data URL for multimodal input.
        response_format: Optional dict like ``{"type": "json_object"}``.
        enable_thinking: Whether to allow model reasoning chains.

    Returns:
        Raw model response text, or empty string on total failure.
    """
    content: list[dict[str, Any]] | str

    if image_data_url:
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        content = user_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout_seconds,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            
            # Some reasoning models (e.g. DeepSeek-R1-style endpoints) expose an
            # ``enable_thinking`` toggle via ``extra_body``.  Disabling thinking
            # reduces latency and avoids <thinking> noise when we only need the final answer.
            extra_body = None
            if not enable_thinking:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            text = clean_thinking_tags(text)
            debug_print_model_output(text, attempt)
            time.sleep(sleep_seconds)
            return text.strip()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds)

    if last_error is not None:
        print(f"[WARN] Falling back after LLM error: {last_error}")
    return ""


def ask_llm_guided_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: int,
    sleep_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    parse_retries: int = 3,
    ranking_key: str = "ranking",
    image_data_url: str | None = None,
    enable_thinking: bool = True,
) -> str:
    """Call the LLM forcing JSON output and validate the extracted ranking.

    Retries the API call up to ``parse_retries`` times if the model returns
    malformed JSON or an invalid ranking string.
    """
    response_format = {"type": "json_object"}

    for attempt in range(1, max(parse_retries, 1) + 1):
        raw = ask_llm(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_text=user_text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            response_format=response_format,
            image_data_url=image_data_url,
            enable_thinking=enable_thinking,
        )

        ranking = extract_ranking_from_json(raw, ranking_key)
        if ranking:
            return ranking

        if attempt < parse_retries:
            time.sleep(retry_backoff_seconds)

    return ""


def predict_ranking(
    client: OpenAI,
    model: str,
    prompt_cfg: dict[str, Any],
    inf_cfg: dict[str, Any],
    api_cfg: dict[str, Any],
    guided_cfg: dict[str, Any],
    article: str,
    titles: list[str],
    image_data_url: str | None,
    rng: random.Random,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a ranking for a single article.

    Args:
        client: OpenAI-compatible client.
        model: Model identifier.
        prompt_cfg: Prompt templates (system + user).
        inf_cfg: Inference hyperparameters.
        api_cfg: API settings (timeout, retries, etc.).
        guided_cfg: Guided JSON configuration.
        article: Article body text.
        titles: List of 10 candidate titles.
        image_data_url: Unused in Task 1 (kept for API compatibility).
        rng: Random instance (kept for API compatibility).
        few_shot_examples: Selected similar training rows to inject.

    Returns:
        Normalized ranking string (e.g., "t3 t1 t9 ...").
    """
    use_guided_json = bool(guided_cfg.get("enabled", False))
    parse_retries = int(guided_cfg.get("parse_retries", 3))
    ranking_key = str(guided_cfg.get("ranking_key", "ranking"))

    base_user_prompt = build_user_prompt(prompt_cfg["user_template"], article, titles)

    # Compose the final user prompt.  If few-shot examples were selected we
    # prepend them (with Spanish instructions because the task data is Spanish)
    # so the model sees the desired input/output pattern before tackling the
    # target article ("CASO_OBJETIVO").
    if few_shot_examples:
        fewshot_block = build_fewshot_examples_for_prompt(few_shot_examples)
        if fewshot_block:
            full_user_prompt = (
                "A continuación tienes ejemplos resueltos del mismo formato. "
                "Aprende el patrón de salida y responde solo con tokens t1..t10.\n\n"
                f"{fewshot_block}\n\n"
                "CASO_OBJETIVO\n"
                f"{base_user_prompt}"
            )
        else:
            full_user_prompt = base_user_prompt
    else:
        full_user_prompt = base_user_prompt

    if use_guided_json:
        raw = ask_llm_guided_json(
            client=client,
            model=model,
            system_prompt=prompt_cfg["system"],
            user_text=full_user_prompt,
            temperature=float(inf_cfg["temperature"]),
            top_p=float(inf_cfg["top_p"]),
            max_tokens=int(inf_cfg["max_tokens"]),
            timeout_seconds=int(api_cfg["timeout_seconds"]),
            sleep_seconds=float(inf_cfg["sleep_seconds"]),
            max_retries=int(inf_cfg["max_retries"]),
            retry_backoff_seconds=float(inf_cfg["retry_backoff_seconds"]),
            parse_retries=parse_retries,
            ranking_key=ranking_key,
            image_data_url=image_data_url,
            enable_thinking=bool(api_cfg.get("enable_thinking", True)),
        )
    else:
        raw = ask_llm(
            client=client,
            model=model,
            system_prompt=prompt_cfg["system"],
            user_text=full_user_prompt,
            temperature=float(inf_cfg["temperature"]),
            top_p=float(inf_cfg["top_p"]),
            max_tokens=int(inf_cfg["max_tokens"]),
            timeout_seconds=int(api_cfg["timeout_seconds"]),
            sleep_seconds=float(inf_cfg["sleep_seconds"]),
            max_retries=int(inf_cfg["max_retries"]),
            retry_backoff_seconds=float(inf_cfg["retry_backoff_seconds"]),
            image_data_url=image_data_url,
            enable_thinking=bool(api_cfg.get("enable_thinking", True)),
        )

    if raw:
        return normalize_ranking(raw)

    return fallback_ranking(article, titles)


def save_submission(df: pd.DataFrame, output_csv: Path, output_zip: Path) -> None:
    """Save results to CSV and ZIP files (final consolidated write)."""
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        output_zip.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_csv, arcname="results.csv")


def initialize_incremental_submission(output_csv: Path, output_columns: list[str]) -> None:
    """Create or overwrite CSV with header before incremental row appends.

    Writing the header once at the start lets us append plain data rows later
    without rewriting the entire file each time.
    """
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    pd.DataFrame(columns=output_columns).to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def append_submission_row(output_csv: Path, output_columns: list[str], record: dict[str, Any]) -> None:
    """Append one result row immediately to avoid losing progress on interruption.

    By using ``mode="a"`` (append) we ensure that even if the process is killed
    after this call, the row is already persisted on disk.  This is the core of
    the incremental-saving strategy.
    """
    pd.DataFrame([record], columns=output_columns).to_csv(
        output_csv,
        mode="a",
        header=False,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def _parse_rank_list(value: Any) -> list[str]:
    """Parse ranking tokens from different string formats."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(token).strip() for token in parsed if str(token).strip()]
        except Exception:
            pass

    cleaned = text.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace(";", " ")
    parts = [part.strip() for part in cleaned.split(",")] if "," in cleaned else [part.strip() for part in cleaned.split()]
    return [part for part in parts if part]


def _token_to_col(token: Any) -> int | None:
    """Convert token like t7 into integer 7."""
    if token is None or (isinstance(token, float) and pd.isna(token)):
        return None

    text = str(token).strip()
    if len(text) < 2:
        return None

    if text[0].lower() not in ("t", "d"):
        return None

    try:
        return int(text[1:])
    except Exception:
        return None


def _unique_valid_pred_cols(pred_tokens: list[str], n_cols: int) -> list[int]:
    """Convert rank tokens to unique numeric columns while preserving order."""
    out: list[int] = []
    seen: set[int] = set()

    for token in pred_tokens:
        col = _token_to_col(token)
        if col is None or not (1 <= col <= n_cols):
            continue
        if col in seen:
            continue
        seen.add(col)
        out.append(col)

    return out


def _ndcg_from_ideal(pred_cols: list[int], ideal_cols: list[int], k: int) -> float:
    """Compute nDCG@k when ideal ranking is explicitly given.

    Gain for a column is its *reverse rank* in the ideal ordering: the top
    item gets ``len(ideal_cols)`` gain, the second gets ``len-1``, etc.
    This makes higher positions in the ideal list more valuable, matching the
    PA-nDCG definition used in the shared task.
    """
    if not ideal_cols:
        return 0.0

    # Map each column to its 0-based index in the ideal ranking.
    ideal_rank: dict[int, int] = {col: idx for idx, col in enumerate(ideal_cols)}

    def gain_for_col(col: int) -> float:
        rank = ideal_rank.get(col)
        if rank is None:
            return 0.0
        return float(len(ideal_cols) - rank)

    dcg = 0.0
    for idx, col in enumerate(pred_cols[:k], start=1):
        dcg += gain_for_col(col) / math.log2(idx + 1)

    idcg = 0.0
    for idx, col in enumerate(ideal_cols[:k], start=1):
        idcg += gain_for_col(col) / math.log2(idx + 1)

    if idcg <= 0.0:
        return 0.0

    return max(0.0, min(1.0, dcg / idcg))


def pa_ndcg(pred_tokens: list[str], true_tokens: list[str], k: int = 10, alpha: float = 0.9) -> float:
    """Baseline PA-nDCG with hard top-1 condition.

    PA-nDCG is defined as:
      * 0.0 if the predicted best title (first token) does not equal the
        ground-truth best title.
      * Otherwise ``alpha + (1 - alpha) * nDCG@k`` computed on the remaining
        9 positions.

    The hard top-1 gate reflects the shared-task requirement that identifying
    the single most relevant headline is critical.
    """
    if not pred_tokens or not true_tokens:
        return 0.0

    ideal_cols = _unique_valid_pred_cols(true_tokens, N_COLS)
    pred_cols = _unique_valid_pred_cols(pred_tokens, N_COLS)

    if not ideal_cols or not pred_cols:
        return 0.0

    # Hard top-1 gate: mismatching best titles yields an instant zero score.
    if pred_cols[0] != ideal_cols[0]:
        return 0.0

    primary = ideal_cols[0]
    pred_rest = [col for col in pred_cols if col != primary]
    ideal_rest = [col for col in ideal_cols if col != primary]

    aux = _ndcg_from_ideal(pred_rest, ideal_rest, k=k)
    score = alpha + (1.0 - alpha) * aux
    return max(0.0, min(1.0, score))


def evaluate_submission(
    reference_csv: Path,
    submission_df: pd.DataFrame,
    task_columns: list[str],
    k: int = 10,
    alpha: float = 0.9,
) -> dict[str, float]:
    """Compute PA-nDCG scores against a reference CSV with ground-truth rankings.

    The PA-nDCG metric applies a hard top-1 condition: if the predicted best
    title does not match the reference best title, the score is zero.
    Otherwise it is ``alpha + (1-alpha) * nDCG@k`` on the remaining positions.
    """
    reference_df = pd.read_csv(reference_csv, dtype={"id": str})
    if "y_true" not in reference_df.columns:
        raise ValueError(f"Reference CSV has no y_true column: {reference_csv}")

    reference_df = reference_df[["id", "y_true"]].copy()
    predictions_df = submission_df[["id", *task_columns]].copy()

    reference_df["id"] = reference_df["id"].astype(str).str.strip()
    predictions_df["id"] = predictions_df["id"].astype(str).str.strip()

    reference_df = reference_df.dropna(subset=["id"]).drop_duplicates(subset=["id"], keep="first")
    predictions_df = predictions_df.dropna(subset=["id"]).drop_duplicates(subset=["id"], keep="first")

    merged = reference_df.merge(predictions_df, on="id", how="left")
    n_rows = len(merged)

    coverage = 0.0
    if n_rows and task_columns:
        coverage = float(merged[task_columns].notna().any(axis=1).mean())

    scores: dict[str, float] = {
        "coverage": coverage,
        "k": float(k),
        "alpha": float(alpha),
    }

    per_task_scores: list[float] = []
    for task_col in task_columns:
        task_values: list[float] = []
        for _, row in merged.iterrows():
            true_tokens = _parse_rank_list(row["y_true"])
            pred_tokens = _parse_rank_list(row.get(task_col))
            task_values.append(pa_ndcg(pred_tokens, true_tokens, k=k, alpha=alpha) if pred_tokens else 0.0)

        task_score = float(sum(task_values) / len(task_values)) if task_values else 0.0
        scores[f"{task_col}_pa_ndcg"] = task_score
        per_task_scores.append(task_score)

    if per_task_scores:
        scores["mean_pa_ndcg"] = float(sum(per_task_scores) / len(per_task_scores))

    return scores


def main() -> None:
    """Load config, run inference, save results, and optionally evaluate."""
    parser = argparse.ArgumentParser(description="Smart few-shot with automatic similar example selection.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)

    global DEBUG_MODEL_OUTPUT_ENABLED
    debug_cfg = cfg.get("debug", {})
    DEBUG_MODEL_OUTPUT_ENABLED = bool(debug_cfg.get("print_model_output", False))
    if DEBUG_MODEL_OUTPUT_ENABLED:
        print("[INFO] Model raw-output debug enabled (debug.print_model_output=true)")

    api_cfg = cfg["api"]
    inf_cfg = cfg["inference"]
    guided_cfg = cfg.get("guided_json", {})
    data_cfg = cfg["data"]
    task1_cfg = cfg.get("task_1", {})
    task2_cfg = cfg.get("task_2", {})
    eval_cfg = cfg.get("evaluation", {})
    prompt_cfg = cfg["prompt"]
    fewshot_select_cfg = cfg.get("fewshot_selection", {})

    input_csv_value = data_cfg.get("input_csv") or data_cfg.get("evaluation_csv", "development_phase_dataset/dev_public.csv")
    input_csv = resolve_path(str(input_csv_value), config_dir)

    training_csvs_cfg = data_cfg.get("training_csvs")
    if training_csvs_cfg:
        if isinstance(training_csvs_cfg, str):
            training_csv_values = [training_csvs_cfg]
        else:
            training_csv_values = [str(item) for item in training_csvs_cfg]
    else:
        training_csv_values = [str(data_cfg.get("training_csv", "development_phase_dataset/train_public.csv"))]
    training_csvs = [resolve_path(training_csv_value, config_dir) for training_csv_value in training_csv_values]

    images_dir = resolve_path(str(data_cfg["images_dir"]), config_dir)
    model_for_output = str(api_cfg.get("model", "model"))
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_for_output).strip("-").lower()
    output_csv = resolve_path(str(data_cfg["output_csv"]).format(model=model_for_output, model_slug=model_slug), config_dir)
    output_zip = resolve_path(str(data_cfg["output_zip"]).format(model=model_for_output, model_slug=model_slug), config_dir)
    max_rows = data_cfg.get("max_rows")

    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.head(int(max_rows))

    client = OpenAI(
        base_url=api_cfg["base_url"],
        api_key=api_cfg["api_key"],
    )

    run_task_1 = bool(task1_cfg.get("enabled", True))
    run_task_2 = bool(task2_cfg.get("enabled", True))
    if not run_task_1 and not run_task_2:
        raise ValueError("At least one task must be enabled: task_1.enabled or task_2.enabled")

    task1_model = task1_cfg.get("model") or api_cfg["model"]
    task2_model = task2_cfg.get("model") or api_cfg["model"]
    use_images = bool(task2_cfg.get("use_images", task2_cfg.get("include_image", True)))
    seed = int(inf_cfg.get("seed", 42))
    rng = random.Random(seed)
    semantic_client = client

    # ------------------------------------------------------------------
    # Few-shot example selection setup
    # ------------------------------------------------------------------
    # When enabled, we load all training data once and (optionally) pre-compute
    # or load cached semantic embeddings.  The embeddings are reused for every
    # test row so we do not re-embed the training corpus on each iteration.
    # ------------------------------------------------------------------
    if fewshot_select_cfg.get("enabled", False):
        training_data = load_training_data_from_csvs(training_csvs)

        similarity_metric = fewshot_select_cfg.get("similarity_metric", "tfidf")
        if similarity_metric in {"semantic", "hybrid_rrf"}:
            semantic_cfg = fewshot_select_cfg.get("semantic", {})
            semantic_base_url = semantic_cfg.get("base_url") or api_cfg.get("base_url")
            semantic_api_key = semantic_cfg.get("api_key") or api_cfg.get("api_key")
            semantic_client = OpenAI(base_url=semantic_base_url, api_key=semantic_api_key)
            cache_embeddings_enabled = semantic_cfg.get("cache_embeddings", True)
            
            if cache_embeddings_enabled:
                cache_path = Path(str(semantic_cfg.get("cache_path", ".embeddings_cache.pkl")))
                if not cache_path.is_absolute():
                    cache_path = config_dir / cache_path

                print(f"Loading/computing embeddings cache from {cache_path}...")
                embedding_model = semantic_cfg.get("model", "text-embedding-3-small")
                embeddings_cache = cache_embeddings(
                    training_data=training_data,
                    cache_path=cache_path,
                    client=semantic_client,
                    embedding_model=embedding_model,
                )

    num_examples = int(fewshot_select_cfg.get("num_examples", 3))
    min_similarity = float(fewshot_select_cfg.get("min_similarity", 0.0))

    output_columns = ["id"]
    if run_task_1:
        output_columns.append("task_1")
    if run_task_2:
        output_columns.append("task_2")
    initialize_incremental_submission(output_csv=output_csv, output_columns=output_columns)

    records = []
    for row in tqdm(df.to_dict(orient="records"), total=len(df), desc="Ranking rows"):
        article = str(row["article_body"])
        titles = [str(row[f"title_{i}"]) for i in range(1, 11)]

        # ------------------------------------------------------------------
        # Per-row few-shot selection
        # ------------------------------------------------------------------
        # For each article we dynamically retrieve the most similar training
        # examples.  When semantic/hybrid metrics are used, the pre-computed
        # embeddings_cache avoids redundant API calls and keeps latency low.
        # ------------------------------------------------------------------
        few_shot_examples = None
        if training_data:
            similarity_metric = fewshot_select_cfg.get("similarity_metric", "tfidf")

            semantic_kwargs = {}
            if similarity_metric in {"semantic", "hybrid_rrf"}:
                semantic_cfg = fewshot_select_cfg.get("semantic", {})
                rrf_cfg = fewshot_select_cfg.get("rrf", {})
                semantic_kwargs = {
                    "similarity_metric": similarity_metric,
                    "client": semantic_client,
                    "embedding_model": semantic_cfg.get("model", "text-embedding-3-small"),
                    "embeddings_cache": embeddings_cache,
                    "rrf_k": int(rrf_cfg.get("k", 60)),
                    "rrf_tfidf_weight": float(rrf_cfg.get("tfidf_weight", 1.0)),
                    "rrf_semantic_weight": float(rrf_cfg.get("semantic_weight", 1.0)),
                }
            else:
                semantic_kwargs = {"similarity_metric": "tfidf"}
            
            few_shot_examples = select_similar_examples(
                article_body=article,
                training_data=training_data,
                num_examples=num_examples,
                min_similarity=min_similarity,
                **semantic_kwargs
            )

        record: dict[str, Any] = {
            "id": row["id"],
        }

        if run_task_1:
            ranking_1 = predict_ranking(
                client=client,
                model=task1_model,
                prompt_cfg=prompt_cfg,
                inf_cfg=inf_cfg,
                api_cfg=api_cfg,
                guided_cfg=guided_cfg,
                article=article,
                titles=titles,
                image_data_url=None,
                rng=rng,
                few_shot_examples=few_shot_examples,
            )
            record["task_1"] = ranking_1

        if run_task_2:
            image_data_url = None
            if use_images:
                image_path = find_image_path(images_dir, row.get("image_hash"))
                if image_path is not None:
                    image_data_url = image_to_data_url(image_path)

            ranking_2 = predict_ranking(
                client=client,
                model=task2_model,
                prompt_cfg=prompt_cfg,
                inf_cfg=inf_cfg,
                api_cfg=api_cfg,
                guided_cfg=guided_cfg,
                article=article,
                titles=titles,
                image_data_url=image_data_url,
                rng=rng,
                few_shot_examples=few_shot_examples,
            )
            record["task_2"] = ranking_2

        records.append(record)
        append_submission_row(output_csv=output_csv, output_columns=output_columns, record=record)

    output_df = pd.DataFrame.from_records(records, columns=output_columns)
    save_submission(output_df, output_csv, output_zip)

    print(f"Generated CSV: {output_csv}")
    print(f"Generated ZIP: {output_zip}")

    if bool(eval_cfg.get("enabled", False)):
        reference_csv_value = eval_cfg.get("reference_csv")
        reference_csv = resolve_path(str(reference_csv_value), config_dir) if reference_csv_value else input_csv
        k = int(eval_cfg.get("k", 10))
        alpha = float(eval_cfg.get("alpha", 0.9))

        task_columns = [col for col in ("task_1", "task_2") if col in output_df.columns]
        scores = evaluate_submission(
            reference_csv=reference_csv,
            submission_df=output_df,
            task_columns=task_columns,
            k=k,
            alpha=alpha,
        )
        print("Evaluation metrics:")
        print(json.dumps(scores, indent=2, ensure_ascii=False))

        save_json = bool(eval_cfg.get("save_json", eval_cfg.get("save_metrics_json", True)))
        if save_json:
            metrics_output_template = str(eval_cfg.get("output_json", "results/metrics_{model_slug}.json"))
            metrics_output_path = resolve_path(
                metrics_output_template.format(model=model_for_output, model_slug=model_slug),
                config_dir,
            )
            metrics_payload = {
                **scores,
                "config_used": cfg,
                "config_path_used": str(config_path),
            }
            metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_output_path.open("w", encoding="utf-8") as metrics_file:
                json.dump(metrics_payload, metrics_file, indent=2, ensure_ascii=False)
            print(f"Saved evaluation metrics JSON: {metrics_output_path}")


if __name__ == "__main__":
    main()
