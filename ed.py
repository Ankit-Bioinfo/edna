#!/usr/bin/env python3
"""
DeepSea eDNA pipeline (single-file).
- Read FASTA/FASTQ (.gz ok)
- Simple FASTQ quality filter (min phred & min length)
- k-mer embeddings (default k=6) OR optional DNABERT-2 embeddings if transformers+torch installed
- UMAP 2D projection
- HDBSCAN clustering
- Output: embeddings.npy, clusters.csv, metrics.json in outdir

Usage example:
python deepsea_edna_pipeline.py --input tests/sample.fasta --format fasta --outdir runs/demo --minlen 80

Requirements (pip):
biopython numpy pandas umap-learn hdbscan scikit-learn tqdm
Optional for DNABERT-2: transformers torch tokenizers
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try imports that may be optional
HAS_UMAP = True
HAS_HDBSCAN = True
try:
    import umap
except Exception:
    HAS_UMAP = False
try:
    import hdbscan
except Exception:
    HAS_HDBSCAN = False

# BioPython for file parsing
try:
    from Bio import SeqIO
except Exception as e:
    print("ERROR: Biopython is required (pip install biopython).", file=sys.stderr)
    raise e

# ---------- IO Helpers ----------
def _open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")

def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    with _open_text(path) as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            yield rec.id, str(rec.seq).upper()

def read_fastq_with_filter(path: str, min_phred: int = 20, min_len: int = 50) -> Iterator[Tuple[str, str]]:
    with _open_text(path) as handle:
        for rec in SeqIO.parse(handle, "fastq"):
            if len(rec) < min_len:
                continue
            quals = rec.letter_annotations.get("phred_quality", [])
            if quals and min(quals) >= min_phred:
                yield rec.id, str(rec.seq).upper()

def clean_sequence(seq: str, alphabet: str = "ACGT") -> str:
    # Replace anything not in alphabet with 'N'
    return ''.join(ch if ch in alphabet else 'N' for ch in seq)

# ---------- Embedding: k-mer and optional DNABERT ----------
def generate_kmer_index(k: int = 6, alphabet: str = "ACGT") -> Dict[str, int]:
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def kmer_vector_for_seq(seq: str, k: int = 6, kmer_index: Optional[Dict[str,int]] = None, alphabet: str = "ACGT") -> np.ndarray:
    if kmer_index is None:
        kmer_index = generate_kmer_index(k=k, alphabet=alphabet)
    vec = np.zeros(len(kmer_index), dtype=float)
    seq = seq.upper().replace(" ", "")
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if 'N' in kmer:
            continue
        # safety: skip if unknown char
        if all(ch in alphabet for ch in kmer):
            idx = kmer_index.get(kmer)
            if idx is not None:
                vec[idx] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

def batch_kmer_embeddings(records: List[Tuple[str,str]], k: int = 6) -> np.ndarray:
    idx = generate_kmer_index(k=k)
    mats = []
    for _, seq in tqdm(records, desc="k-mer embeddings"):
        mats.append(kmer_vector_for_seq(seq, k=k, kmer_index=idx))
    return np.vstack(mats) if mats else np.empty((0, len(idx)))

def try_dnabert2_embeddings(records: List[Tuple[str,str]], model_name: str = "zhihan1996/DNABERT-2-117M", max_len_tokens: int = 512) -> np.ndarray:
    """
    Try to compute DNABERT-2 embeddings using transformers + torch.
    Falls back to k-mer embeddings on any failure.
    Note: This requires internet the first time to fetch model from HF unless cached locally.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except Exception as e:
        print("DNABERT not available (transformers/torch missing). Falling back to k-mer.", file=sys.stderr)
        return batch_kmer_embeddings(records, k=6)

    print("Loading DNABERT-2 model (this may take time)...")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        mdl.eval()
    except Exception as e:
        print("Failed to load DNABERT model:", e, file=sys.stderr)
        print("Falling back to k-mer embeddings.", file=sys.stderr)
        return batch_kmer_embeddings(records, k=6)

    embs = []
    with torch.no_grad():
        for _, seq in tqdm(records, desc="DNABERT embeddings"):
            # DNABERT tokenizers sometimes expect spaced tokens or special processing.
            # The original recipe in the doc used ' '.join(list(seq)) or BPE variant;
            # use simple char-separated input; adapt if model requires otherwise.
            s = seq.replace("N", "A")  # avoid unknown tokens by replacing N with A (simple heuristic)
            inputs = tok(s, return_tensors="pt", truncation=True, max_length=max_len_tokens)
            out = mdl(**inputs)
            # use mean pooling across tokens of last_hidden_state
            last = out.last_hidden_state  # (1, L, D)
            pooled = last.mean(dim=1).squeeze(0).cpu().numpy()
            embs.append(pooled)
    if not embs:
        return np.empty((0, mdl.config.hidden_size))
    return np.vstack(embs)

# ---------- Dimensionality reduction & Clustering ----------
def reduce_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    if not HAS_UMAP:
        raise RuntimeError("UMAP not installed (pip install umap-learn)")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)

def cluster_hdbscan(embedding_2d: np.ndarray, min_cluster_size: int = 10, min_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_HDBSCAN:
        raise RuntimeError("hdbscan not installed (pip install hdbscan)")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embedding_2d)
    # clusterer.probabilities_ may be available; otherwise fallback to zeros
    probs = getattr(clusterer, "probabilities_", np.zeros(len(labels), dtype=float))
    return labels, probs

# ---------- Diversity metrics ----------
def shannon_entropy(counts: List[int], base: float = math.e) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            H -= p * math.log(p, base)
    return H

def simpson_index(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 1:
        return 0.0
    num = sum(c * (c - 1) for c in counts)
    return 1.0 - num / (total * (total - 1))

def alpha_diversity_from_labels(labels: Iterable[int]) -> Dict[str, float]:
    # ignore noise (-1)
    filtered = [int(l) for l in labels if int(l) >= 0]
    c = Counter(filtered)
    counts = list(c.values())
    return {
        "num_clusters": len(c),
        "total_non_noise": sum(counts),
        "shannon": float(shannon_entropy(counts)),
        "simpson": float(simpson_index(counts))
    }

# ---------- Save outputs ----------
def save_outputs(outdir: str, ids: List[str], embedding: np.ndarray, umap2d: np.ndarray, labels: np.ndarray, probs: np.ndarray, metrics: Dict):
    os.makedirs(outdir, exist_ok=True)
    emb_path = os.path.join(outdir, "embeddings.npy")
    np.save(emb_path, embedding)
    # ids
    with open(os.path.join(outdir, "embeddings.ids.txt"), "w") as fh:
        fh.write("\n".join(ids) + "\n")
    # clusters CSV
    df = pd.DataFrame({
        "id": ids,
        "label": labels.astype(int),
        "probability": probs.astype(float),
        "umap_x": umap2d[:, 0],
        "umap_y": umap2d[:, 1],
    })
    df.to_csv(os.path.join(outdir, "clusters.csv"), index=False)
    # metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Wrote: {emb_path}, clusters.csv, metrics.json in {outdir}")

# ---------- Main pipeline ----------
def load_records(input_path: str, fmt: str, minlen: int, min_phred: int) -> List[Tuple[str,str]]:
    print("Loading records from", input_path)
    if fmt == "fasta":
        it = read_fasta(input_path)
    elif fmt == "fastq":
        it = read_fastq_with_filter(input_path, min_phred=min_phred, min_len=minlen)
    else:
        raise ValueError("fmt must be 'fasta' or 'fastq'")
    # clean and filter by minlen
    recs = []
    for rid, seq in it:
        seq2 = clean_sequence(seq)
        if len(seq2) >= minlen:
            recs.append((rid, seq2))
    print(f"Loaded {len(recs)} records after filtering (minlen={minlen})")
    return recs

def run_pipeline(
    input_path: str,
    fmt: str,
    outdir: str,
    use_dnabert: bool = False,
    k: int = 6,
    minlen: int = 80,
    minphred: int = 20,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    hdbscan_min_cluster_size: int = 10,
    random_state: int = 42
):
    recs = load_records(input_path, fmt, minlen=minlen, min_phred=minphred)
    ids = [rid for rid, _ in recs]

    if use_dnabert:
        print("Attempting DNABERT-2 embeddings...")
        X = try_dnabert2_embeddings(recs)
        if X is None or X.size == 0:
            print("DNABERT returned empty embeddings; falling back to k-mer.")
            X = batch_kmer_embeddings(recs, k=k)
    else:
        print(f"Computing k-mer (k={k}) embeddings...")
        X = batch_kmer_embeddings(recs, k=k)

    if X.size == 0:
        raise RuntimeError("No embeddings generated (empty input?). Aborting.")

    print("Reducing with UMAP...")
    umap2d = reduce_umap(X, n_neighbors=umap_neighbors, min_dist=umap_min_dist, n_components=2, random_state=random_state)

    print("Clustering with HDBSCAN...")
    labels, probs = cluster_hdbscan(umap2d, min_cluster_size=hdbscan_min_cluster_size)

    metrics = alpha_diversity_from_labels(labels)
    metrics.update({
        "num_sequences": len(ids),
        "embedding_dim": int(X.shape[1]),
    })

    save_outputs(outdir, ids, X, umap2d, labels, probs, metrics)
    print("Pipeline finished. Metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(prog="deepsea_edna_pipeline.py", description="End-to-end eDNA pipeline")
    p.add_argument("--input", required=True, help="Path to input FASTA/FASTQ (gz allowed)")
    p.add_argument("--format", choices=["fasta","fastq"], required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--minlen", type=int, default=80)
    p.add_argument("--minphred", type=int, default=20, help="FASTQ min phred (used only for fastq)")
    p.add_argument("--embedding-model", choices=["kmer","dnabert2"], default="kmer")
    p.add_argument("--k", type=int, default=6, help="k for k-mer embedding")
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    use_dnabert = args.embedding_model == "dnabert2"
    run_pipeline(
        input_path=args.input,
        fmt=args.format,
        outdir=args.outdir,
        use_dnabert=use_dnabert,
        k=args.k,
        minlen=args.minlen,
        minphred=args.minphred,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()
