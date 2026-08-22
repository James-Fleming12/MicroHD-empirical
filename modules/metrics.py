"""Generalization metrics for compressed HDC models."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.view(-1) == y.view(-1).to(pred.device)).float().mean().item()

def hungarian_accuracy(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Best-match cluster<->class assignment accuracy (unsupervised metric)."""
    c_true = true_labels.max() + 1
    c_pred = cluster_labels.max() + 1
    conf = np.zeros((c_pred, c_true), dtype=np.int64)
    for p, t in zip(cluster_labels, true_labels):
        conf[p, t] += 1
    row, col = linear_sum_assignment(-conf)
    return conf[row, col].sum() / len(true_labels)

def clustering_metrics(embeddings: torch.Tensor, k: int, labels: np.ndarray, seed: int = 0) -> dict:
    emb = embeddings.detach().cpu().numpy().astype(np.float64)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(emb)
    return {
        "nmi": float(normalized_mutual_info_score(labels, km.labels_)),
        "ari": float(adjusted_rand_score(labels, km.labels_)),
        "hungarian_acc": float(hungarian_accuracy(km.labels_, labels)),
    }

def ood_detection_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for separating ID from OOD using a confidence score where higher=more ID."""
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))

@torch.no_grad()
def geometry_fidelity(encoder, x: torch.Tensor, *, n_points: int = 500, seed: int = 0) -> float:
    """Spearman correlation between input-space and encoded-space similarity.

    Random projections preserve pairwise geometry (J-L); aggressive dimension
    reduction breaks this. We measure rank correlation of pairwise cosine
    similarities before/after encoding.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:n_points]
    xs = x[idx]
    xin = torch.nn.functional.normalize(xs, dim=1)
    sim_in = (xin @ xin.T)[torch.triu_indices(len(xs), len(xs), offset=1).unbind()]

    h = encoder.encode(xs)
    hin = torch.nn.functional.normalize(h, dim=1)
    sim_out = (hin @ hin.T)[torch.triu_indices(len(xs), len(xs), offset=1).unbind()]

    rho, _ = spearmanr(sim_in.cpu().numpy(), sim_out.cpu().numpy())
    return float(rho)
