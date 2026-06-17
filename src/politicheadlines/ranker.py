from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .data import parse_ranking
from .features import Embeddings

class RankingMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, article: torch.Tensor, title: torch.Tensor) -> torch.Tensor:
        x = torch.cat([article, title, torch.abs(article-title), article*title], dim=-1)
        return self.net(x).squeeze(-1)

class PairwiseDataset(Dataset):
    def __init__(self, emb: Embeddings, indices: list[int]):
        self.samples = []
        if emb.gold is None: raise ValueError("Gold rankings are required")
        for row_idx in indices:
            order = parse_ranking(emb.gold[row_idx]); pos = {token: i for i, token in enumerate(order)}
            for i in range(10):
                for j in range(i + 1, 10):
                    better, worse = (i, j) if pos[f"t{i+1}"] < pos[f"t{j+1}"] else (j, i)
                    self.samples.append((row_idx, better, worse))
        self.articles = torch.from_numpy(emb.articles); self.titles = torch.from_numpy(emb.titles)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        row, better, worse = self.samples[idx]
        return self.articles[row], self.titles[row, better], self.titles[row, worse]

def train_ranker(emb: Embeddings, cfg: dict, output_path: str) -> RankingMLP:
    rcfg = cfg["ranker"]; seed = int(cfg.get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    indices = list(range(len(emb.ids))); random.shuffle(indices)
    n_val = max(1, int(len(indices) * rcfg.get("validation_ratio", 0.2))); train_idx, val_idx = indices[n_val:], indices[:n_val]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RankingMLP(emb.articles.shape[1], rcfg.get("hidden_dim", 256), rcfg.get("dropout", 0.1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rcfg.get("learning_rate", 1e-3), weight_decay=rcfg.get("weight_decay", 1e-4))
    loss_fn = nn.MarginRankingLoss(margin=rcfg.get("margin", 0.2))
    loader = DataLoader(PairwiseDataset(emb, train_idx), batch_size=rcfg.get("batch_size", 128), shuffle=True)
    best, best_state = float("inf"), None
    for epoch in range(1, rcfg.get("epochs", 20)+1):
        model.train(); total = 0.0
        for article, better, worse in loader:
            article, better, worse = article.to(device), better.to(device), worse.to(device)
            opt.zero_grad(); sb, sw = model(article, better), model(article, worse)
            loss = loss_fn(sb, sw, torch.ones_like(sb)); loss.backward(); opt.step(); total += loss.item() * len(article)
        val = pairwise_loss(model, emb, val_idx, rcfg.get("margin", 0.2), device)
        print(f"epoch={epoch:02d} train_loss={total/max(len(loader.dataset),1):.6f} val_loss={val:.6f}")
        if val < best:
            best = val; best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    assert best_state is not None
    torch.save({"state_dict": best_state, "dim": emb.articles.shape[1], "hidden_dim": rcfg.get("hidden_dim",256), "dropout": rcfg.get("dropout",0.1)}, output_path)
    model.load_state_dict(best_state); return model

@torch.inference_mode()
def pairwise_loss(model, emb, indices, margin, device):
    ds = PairwiseDataset(emb, indices); loader = DataLoader(ds, batch_size=256); fn = nn.MarginRankingLoss(margin=margin, reduction="sum"); model.eval(); total=0.0
    for a,b,w in loader:
        a,b,w=a.to(device),b.to(device),w.to(device); total += fn(model(a,b), model(a,w), torch.ones(len(a), device=device)).item()
    return total/max(len(ds),1)

@torch.inference_mode()
def predict(model: RankingMLP, emb: Embeddings) -> list[str]:
    device = next(model.parameters()).device; model.eval(); out=[]
    for a, titles in zip(emb.articles, emb.titles):
        at = torch.from_numpy(np.repeat(a[None,:], 10, axis=0)).to(device); tt = torch.from_numpy(titles).to(device)
        scores = model(at,tt).cpu().numpy(); order=np.argsort(-scores)
        out.append(" ".join(f"t{i+1}" for i in order))
    return out

def load_ranker(path: str) -> RankingMLP:
    ckpt=torch.load(path, map_location="cpu"); model=RankingMLP(ckpt["dim"],ckpt["hidden_dim"],ckpt["dropout"]); model.load_state_dict(ckpt["state_dict"]); return model.to("cuda" if torch.cuda.is_available() else "cpu")
