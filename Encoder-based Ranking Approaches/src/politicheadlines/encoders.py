from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * weights).sum(1) / weights.sum(1).clamp_min(1e-9)

class TextEncoder:
    def __init__(self, model_name: str, batch_size: int = 8, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        batches = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding text", leave=False):
            batch = texts[i:i+self.batch_size]
            x = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            y = self.model(**x)
            emb = F.normalize(mean_pool(y.last_hidden_state, x["attention_mask"]), p=2, dim=1)
            batches.append(emb.cpu().numpy())
        return np.vstack(batches)

    def _token_chunks(self, text: str, max_length: int, overlap: int) -> list[str]:
        usable = max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if overlap >= usable:
            raise ValueError("overlap must be smaller than usable chunk length")
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        step = usable - overlap
        return [self.tokenizer.decode(ids[i:i+usable], skip_special_tokens=True) for i in range(0, max(len(ids), 1), step)] or [""]

    def _sentence_chunks(self, text: str, max_length: int) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] or [text]
        chunks, current = [], []
        for sentence in sentences:
            candidate = " ".join([*current, sentence])
            if current and len(self.tokenizer(candidate, add_special_tokens=True)["input_ids"]) > max_length:
                chunks.append(" ".join(current)); current = [sentence]
            else:
                current.append(sentence)
        if current: chunks.append(" ".join(current))
        return chunks or [""]

    def encode_articles(self, texts: list[str], strategy: str, max_length: int, overlap: int = 0, pooling: str = "mean", decay: float = 0.85) -> np.ndarray:
        if strategy == "first":
            return self.encode(texts, max_length)
        output = []
        for text in tqdm(texts, desc="Encoding articles"):
            chunks = self._token_chunks(str(text), max_length, overlap) if strategy == "token_chunks" else self._sentence_chunks(str(text), max_length)
            emb = self.encode(chunks, max_length)
            if pooling == "weighted":
                w = np.array([decay ** i for i in range(len(emb))], dtype=np.float32); w /= w.sum()
                pooled = (emb * w[:, None]).sum(0)
            else:
                pooled = emb.mean(0)
            output.append(pooled / max(np.linalg.norm(pooled), 1e-12))
        return np.vstack(output)

class ImageEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, paths: list[Path | None]) -> np.ndarray:
        dim = int(self.model.config.projection_dim)
        out = []
        for path in tqdm(paths, desc="Encoding images", leave=False):
            if path is None or not path.exists():
                out.append(np.zeros(dim, dtype=np.float32)); continue
            image = Image.open(path).convert("RGB")
            x = self.processor(images=image, return_tensors="pt").to(self.device)
            emb = F.normalize(self.model.get_image_features(**x), p=2, dim=1)[0]
            out.append(emb.cpu().numpy())
        return np.vstack(out)

def find_image(images_dir: str | Path, image_hash: object) -> Path | None:
    base = Path(images_dir); stem = str(image_hash).strip()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ""):
        p = base / f"{stem}{ext}"
        if p.exists(): return p
    return None

def align_dimension(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[1] < dim:
        x = np.pad(x, ((0, 0), (0, dim - x.shape[1])))
    elif x.shape[1] > dim:
        x = x[:, :dim]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)
