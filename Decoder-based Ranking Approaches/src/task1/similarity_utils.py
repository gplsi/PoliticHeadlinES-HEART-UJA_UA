"""
Similarity utilities for automatic few-shot example selection.

Provides TF-IDF, semantic (embedding-based), and hybrid RRF similarity
computation to match test articles with training examples, enabling
dynamic few-shot learning without hand-crafted prompts.

Supported similarity metrics:
- tfidf: token overlap (fast, deterministic, no external API calls).
- semantic: cosine similarity of text embeddings (requires OpenAI-compatible API).
- hybrid_rrf: reciprocal rank fusion of TF-IDF + semantic scores.
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI


def tokenize(text: str) -> set[str]:
    """Extract meaningful alphanumeric tokens from text for similarity computation.

    Keeps Spanish accented characters and filters short tokens (<3 chars).
    """
    return set(re.findall(r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]{3,}", text.lower()))


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard-style token overlap similarity between two texts.

    Uses the intersection-over-union of token sets.  Higher value = more similar.

    Args:
        text1: First text (typically the test article body).
        text2: Second text (typically a training article body).

    Returns:
        Similarity score in the range [0.0, 1.0].
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union else 0.0


def load_training_data(training_csv: Path) -> list[dict[str, Any]]:
    """Load training examples from a CSV file.

    Expected columns: id, article_body, title_1..title_10, y_true

    Args:
        training_csv: Path to the training CSV file.

    Returns:
        List of row dictionaries with all columns preserved.
    """
    df = pd.read_csv(training_csv)
    return df.to_dict(orient="records")


def get_embedding(
    client: OpenAI,
    text: str,
    model: str = "text-embedding-3-small",
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> list[float] | None:
    """Request an embedding vector from an OpenAI-compatible API.

    Args:
        client: Initialized OpenAI client instance.
        text: Text to embed.
        model: Embedding model identifier.
        max_retries: Number of retries on transient failures.
        retry_backoff_seconds: Sleep time between retries.

    Returns:
        Embedding vector as a list of floats, or None on failure.
    """
    if not text or not text.strip():
        return None

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model=model,
                input=text.strip(),
            )
            return response.data[0].embedding
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds)

    print(f"[WARN] Failed to get embedding after {max_retries} attempts: {last_error}")
    return None


def compute_semantic_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.

    Returns:
        Similarity score in [-1.0, 1.0] (typically [0.0, 1.0] for normalized embeddings).
    """
    if not embedding1 or not embedding2:
        return 0.0

    arr1 = np.array(embedding1, dtype=np.float32)
    arr2 = np.array(embedding2, dtype=np.float32)

    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def cache_embeddings(
    training_data: list[dict[str, Any]],
    cache_path: Path,
    client: OpenAI,
    embedding_model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Load cached embeddings or compute and persist them.

    Args:
        training_data: List of training row dictionaries.
        cache_path: File path for the pickle cache.
        client: OpenAI client instance for computing embeddings.
        embedding_model: Model identifier to use for embeddings.

    Returns:
        List of embedding vectors, one per training example.
    """
    # Try loading an existing cache if it matches the dataset size.
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                if isinstance(cached, list) and len(cached) == len(training_data):
                    print(f"Loaded {len(cached)} embeddings from cache: {cache_path}")
                    return cached
        except Exception as exc:
            print(f"[WARN] Failed to load embedding cache: {exc}")

    # Compute embeddings sequentially (batches can be added if the API supports it).
    print(f"Computing embeddings for {len(training_data)} training examples...")
    embeddings: list[list[float]] = []
    for idx, row in enumerate(training_data):
        article = str(row.get("article_body", "")).strip()
        embedding = get_embedding(client, article, model=embedding_model)
        embeddings.append(embedding if embedding else [0.0] * 1536)

        if (idx + 1) % 10 == 0:
            print(f"  ... {idx + 1}/{len(training_data)} embeddings computed")

    # Persist the cache for future runs.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings cache to: {cache_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save embedding cache: {exc}")

    return embeddings


def select_similar_examples(
    article_body: str,
    training_data: list[dict[str, Any]],
    num_examples: int = 3,
    min_similarity: float = 0.0,
    similarity_metric: str = "tfidf",
    client: OpenAI | None = None,
    embedding_model: str = "text-embedding-3-small",
    embeddings_cache: list[list[float]] | None = None,
    article_embedding: list[float] | None = None,
    rrf_k: int = 60,
    rrf_tfidf_weight: float = 1.0,
    rrf_semantic_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Select the most similar training examples for few-shot prompting.

    Supports three similarity modes:
    - ``tfidf``: Jaccard token overlap (fastest, no API calls).
    - ``semantic``: cosine similarity of text embeddings.
    - ``hybrid_rrf``: reciprocal rank fusion combining TF-IDF + semantic rankings.

    Args:
        article_body: Target article text.
        training_data: List of training row dictionaries.
        num_examples: Maximum number of examples to return.
        min_similarity: Minimum similarity threshold; scores below are discarded.
        similarity_metric: One of ``"tfidf"``, ``"semantic"``, ``"hybrid_rrf"``.
        client: OpenAI client (required for semantic/hybrid modes).
        embedding_model: Embedding model identifier.
        embeddings_cache: Pre-computed training embeddings (optional).
        article_embedding: Pre-computed article embedding (optional).
        rrf_k: RRF hyperparameter controlling rank smoothing.
        rrf_tfidf_weight: Weight for the TF-IDF component in RRF.
        rrf_semantic_weight: Weight for the semantic component in RRF.

    Returns:
        List of selected example dictionaries, most similar first.

    Raises:
        ValueError: If a required client is missing for semantic/hybrid modes.
    """
    # ------------------------------------------------------------------
    # Hybrid RRF: combine TF-IDF and semantic rankings.
    # ------------------------------------------------------------------
    if similarity_metric == "hybrid_rrf":
        if not client and not article_embedding:
            raise ValueError("client or article_embedding must be provided for hybrid_rrf")

        if article_embedding is None:
            article_embedding = get_embedding(client, article_body, model=embedding_model)
            if not article_embedding:
                print("[WARN] Failed to get article embedding, falling back to TF-IDF")
                return select_similar_examples(
                    article_body=article_body,
                    training_data=training_data,
                    num_examples=num_examples,
                    min_similarity=min_similarity,
                    similarity_metric="tfidf",
                )

        if embeddings_cache is None:
            if not client:
                raise ValueError("client required to compute embeddings")
            embeddings_cache = [
                get_embedding(client, str(row.get("article_body", "")), model=embedding_model)
                or [0.0] * 1536
                for row in training_data
            ]

        tfidf_scores: dict[int, float] = {}
        semantic_scores: dict[int, float] = {}
        for idx, row in enumerate(training_data):
            train_article = str(row.get("article_body", ""))
            tfidf_scores[idx] = compute_tfidf_similarity(article_body, train_article)
            train_embedding = embeddings_cache[idx] if idx < len(embeddings_cache) else [0.0] * 1536
            semantic_scores[idx] = compute_semantic_similarity(article_embedding, train_embedding)

        tfidf_ranked = [idx for idx, _ in sorted(tfidf_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        semantic_ranked = [idx for idx, _ in sorted(semantic_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        tfidf_ranks = {idx: rank for rank, idx in enumerate(tfidf_ranked, start=1)}
        semantic_ranks = {idx: rank for rank, idx in enumerate(semantic_ranked, start=1)}

        k = max(1, int(rrf_k))
        similarities: list[tuple[int, float, dict[str, Any]]] = []
        for idx, row in enumerate(training_data):
            max_component_sim = max(tfidf_scores.get(idx, 0.0), semantic_scores.get(idx, 0.0))
            if max_component_sim < min_similarity:
                continue

            rrf_score = 0.0
            if idx in tfidf_ranks:
                rrf_score += float(rrf_tfidf_weight) / float(k + tfidf_ranks[idx])
            if idx in semantic_ranks:
                rrf_score += float(rrf_semantic_weight) / float(k + semantic_ranks[idx])

            similarities.append((idx, rrf_score, row))

    # ------------------------------------------------------------------
    # Semantic only: embedding cosine similarity.
    # ------------------------------------------------------------------
    elif similarity_metric == "semantic":
        if not client and not article_embedding:
            raise ValueError("client or article_embedding must be provided for semantic similarity")

        if article_embedding is None:
            article_embedding = get_embedding(client, article_body, model=embedding_model)
            if not article_embedding:
                print("[WARN] Failed to get article embedding, falling back to TF-IDF")
                return select_similar_examples(
                    article_body=article_body,
                    training_data=training_data,
                    num_examples=num_examples,
                    min_similarity=min_similarity,
                    similarity_metric="tfidf",
                )

        if embeddings_cache is None:
            if not client:
                raise ValueError("client required to compute embeddings")
            embeddings_cache = [
                get_embedding(client, str(row.get("article_body", "")), model=embedding_model)
                or [0.0] * 1536
                for row in training_data
            ]

        similarities: list[tuple[int, float, dict[str, Any]]] = []
        for idx, row in enumerate(training_data):
            train_embedding = embeddings_cache[idx] if idx < len(embeddings_cache) else [0.0] * 1536
            sim = compute_semantic_similarity(article_embedding, train_embedding)
            if sim >= min_similarity:
                similarities.append((idx, sim, row))

    # ------------------------------------------------------------------
    # TF-IDF only: token overlap (default, fastest).
    # ------------------------------------------------------------------
    else:
        similarities: list[tuple[int, float, dict[str, Any]]] = []
        for idx, row in enumerate(training_data):
            train_article = str(row.get("article_body", ""))
            sim = compute_tfidf_similarity(article_body, train_article)
            if sim >= min_similarity:
                similarities.append((idx, sim, row))

    similarities.sort(key=lambda x: (-x[1], x[0]))
    return [row for _, _, row in similarities[:num_examples]]


def build_fewshot_examples_for_prompt(
    selected_examples: list[dict[str, Any]],
) -> str:
    """Format selected training examples as a few-shot prompt block.

    Args:
        selected_examples: List of row dicts with article_body, titles, and y_true.

    Returns:
        Formatted string suitable for injection into the LLM prompt.
    """
    if not selected_examples:
        return ""

    rendered_examples: list[str] = []
    for idx, row in enumerate(selected_examples, start=1):
        article = str(row.get("article_body", "")).strip()
        ranking = str(row.get("y_true", "")).strip()
        titles = [str(row.get(f"title_{i}", "")) for i in range(1, 11)]

        if not article or not ranking or not titles or len(titles) != 10:
            continue

        titles_text = "\n".join(f"t{i}: {title}" for i, title in enumerate(titles, start=1))
        rendered_examples.append(
            (
                f"EJEMPLO {idx}\n"
                f"ARTICULO:\n{article}\n\n"
                f"TITULARES:\n{titles_text}\n\n"
                f"SALIDA_CORRECTA:\n{ranking}"
            )
        )

    return "\n\n".join(rendered_examples) if rendered_examples else ""
