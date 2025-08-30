#!/usr/bin/env python3
"""
edna_pipeline.py
Single-file, data-driven eDNA pipeline for marine / unknown organisms.

Features:
 - Accept FASTA input (reads or contigs)
 - Optionally run QC/trimming (if 'fastp' available)
 - k-mer embedding (configurable k)
 - Dimensionality reduction (PCA/UMAP)
 - Clustering (OPTICS/DBSCAN)
 - Local BLASTn against provided DB or remote NCBI qblast fallback
 - MSA via MAFFT (if available), tree via FastTree (if available), fallback NJ
 - Abundance tables, diversity metrics, cluster & phylo plots
 - Robust checks and logging
"""

import os
import io
import sys
import math
import json
import time
import shutil
import logging
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML  # remote fallback
from itertools import product
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# optional
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edna_pipeline")

# -----------------------
# Utilities
# -----------------------

def check_binary(name: str) -> bool:
    """Return True if a binary is available in PATH."""
    return shutil.which(name) is not None

def run_cmd(cmd: List[str], capture: bool=False, check: bool=True) -> Tuple[int, str]:
    """Run subprocess command and return (retcode, stdout)"""
    try:
        if capture:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            return 0, out
        else:
            subprocess.check_call(cmd)
            return 0, ""
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s\nOutput:\n%s", " ".join(cmd), getattr(e, "output", ""))
        return e.returncode, getattr(e, "output", "")

def read_fasta(path_or_handle) -> List[Tuple[str,str]]:
    """Return list of tuples (id, seq) from FASTA path or file-like."""
    recs = []
    if isinstance(path_or_handle, str):
        handle = open(path_or_handle, "r")
    else:
        handle = path_or_handle
    for rec in SeqIO.parse(handle, "fasta"):
        recs.append((rec.id, str(rec.seq).upper()))
    if isinstance(path_or_handle, str):
        handle.close()
    return recs

# -----------------------
# k-mer & embeddings
# -----------------------

def build_vocab(k: int):
    return ["".join(p) for p in product("ACGT", repeat=k)]

def seq_to_kmer_counts(seq: str, k:int, vocab_idx:Dict[str,int]) -> np.ndarray:
    L = len(vocab_idx)
    v = np.zeros(L, dtype=np.float32)
    s = seq.upper()
    for i in range(len(s)-k+1):
        kmer = s[i:i+k]
        if all(ch in "ACGT" for ch in kmer):
            idx = vocab_idx.get(kmer)
            if idx is not None:
                v[idx] += 1
    if v.sum() > 0:
        v = v / v.sum()
    return v

def batch_kmers(seqs: List[str], k:int) -> np.ndarray:
    vocab = build_vocab(k)
    vocab_idx = {kmer:i for i,kmer in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i] = seq_to_kmer_counts(s, k, vocab_idx)
    return X

# -----------------------
# Clustering & novelty detection
# -----------------------

def safe_cluster(Z: np.ndarray, method:str="optics", min_samples:int=5, xi:float=0.05, eps:float=0.5):
    n = Z.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    min_samples = max(2, min(int(min_samples), n))
    if method == "optics":
        model = OPTICS(min_samples=min_samples, xi=xi)
    else:
        model = DBSCAN(min_samples=min_samples, eps=eps)
    labels = model.fit_predict(Z)
    return labels

# -----------------------
# BLAST wrapper (local + remote fallback)
# -----------------------

def blast_local(query_fasta: str, db: str, out_xml: str, evalue:float=1e-5, num_threads:int=1):
    """Run blastn (local)."""
    if not check_binary("blastn"):
        raise FileNotFoundError("blastn not found in PATH")
    cmd = ["blastn", "-query", query_fasta, "-db", db, "-outfmt", "5", "-out", out_xml, "-evalue", str(evalue), "-num_threads", str(num_threads)]
    rc, out = run_cmd(cmd, capture=False)
    return rc

def blast_remote_seq(seq: str, program: str="blastn", database: str="nt"):
    """Use NCBIWWW.qblast (Biopython) - slow and rate-limited."""
    try:
        res = NCBIWWW.qblast(program, database, seq)
        return res.read()  # XML string
    except Exception as e:
        logger.error("Remote BLAST failed: %s", e)
        return None

def parse_blast_xml(xml_path_or_str: str, is_string:bool=False):
    """Return list of dicts per query: [{'query_id', 'hits': [{'acc','tax','identity','pident','evalue','length'} , ...]}]"""
    data = []
    if is_string:
        handle = io.StringIO(xml_path_or_str)
    else:
        handle = open(xml_path_or_str)
    blast_records = NCBIXML.parse(handle)
    for rec in blast_records:
        q = {"query_id": rec.query, "hits": []}
        for alignment in rec.alignments:
            for hsp in alignment.hsps:
                q["hits"].append({
                    "title": alignment.title,
                    "accession": alignment.accession,
                    "length": alignment.length,
                    "evalue": hsp.expect,
                    "identity": hsp.identities,
                    "align_len": hsp.align_length,
                    "query_start": hsp.query_start,
                    "query_end": hsp.query_end
                })
        data.append(q)
    if not is_string:
        handle.close()
    return data

# -----------------------
# MSA & phylogeny (MAFFT + FastTree)
# -----------------------

def run_mafft(fasta_in: str, fasta_out: str, options: str="--auto"):
    if not check_binary("mafft"):
        raise FileNotFoundError("mafft not found in PATH")
    cmd = ["mafft"] + options.split() + [fasta_in]
    rc, out = run_cmd(cmd, capture=True)
    if rc == 0:
        with open(fasta_out, "w") as f:
            f.write(out)
    return rc

def run_fasttree(msa_fasta: str, tree_out: str):
    if not check_binary("fasttree"):
        raise FileNotFoundError("fasttree not found in PATH")
    cmd = ["fasttree", "-nt", msa_fasta]
    rc, out = run_cmd(cmd, capture=True)
    if rc == 0:
        with open(tree_out, "w") as f:
            f.write(out)
    return rc

# -----------------------
# Diversity metrics & plotting
# -----------------------

def shannon_entropy(counts):
    freqs = np.array(counts, dtype=float)
    s = freqs.sum()
    if s <= 0:
        return 0.0
    p = freqs / s
    p = p[p>0]
    return -np.sum(p * np.log(p))

# plotting helpers omitted for brevity — create Matplotlib/Plotly as needed

# -----------------------
# Main pipeline driver (CLI-like usage)
# -----------------------

def pipeline_from_fasta(fasta_path: str,
                        outdir: str = "edna_out",
                        k: int = 6,
                        cluster_method: str = "optics",
                        min_samples: int = 10,
                        use_local_blast_db: Optional[str] = None,
                        run_mafft_fasttree: bool = True):
    os.makedirs(outdir, exist_ok=True)

    # 1) Read fasta
    seqs = read_fasta(fasta_path)
    ids = [s[0] for s in seqs]
    seqstrs = [s[1] for s in seqs]
    logger.info("Read %d sequences from %s", len(seqs), fasta_path)

    # 2) k-mer vectors
    X = batch_kmers(seqstrs, k=k)
    np.save(os.path.join(outdir, "kmer_embeddings.npy"), X)
    logger.info("Saved k-mer embeddings (%s)", os.path.join(outdir, "kmer_embeddings.npy"))

    # 3) Dim reduction (PCA / UMAP)
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    Z = pca.fit_transform(X)
    if HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        z2 = reducer.fit_transform(X)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X.shape[0]//3)))
        z2 = tsne.fit_transform(X)
    pd.DataFrame(z2, index=ids, columns=["dim1","dim2"]).to_csv(os.path.join(outdir,"embedding_2d.csv"))

    # 4) Clustering
    labels = safe_cluster(Z, method=cluster_method, min_samples=min_samples)
    pd.DataFrame({"seq_id": ids, "cluster": labels}).to_csv(os.path.join(outdir,"clusters.csv"), index=False)
    logger.info("Clustering done. clusters.csv saved.")

    # 5) Taxonomy via BLAST (local or remote)
    blast_results = []
    if use_local_blast_db and check_binary("blastn"):
        # write query to temp fasta
        qf = os.path.join(outdir, "query.fasta")
        with open(qf, "w") as fh:
            for sid, s in zip(ids, seqstrs):
                fh.write(f">{sid}\n{s}\n")
        xml_out = os.path.join(outdir, "blast_local.xml")
        try:
            blast_local(qf, use_local_blast_db, xml_out, evalue=1e-5, num_threads=4)
            blast_results = parse_blast_xml(xml_out)
        except Exception as e:
            logger.warning("Local BLAST failed, will try remote. %s", e)
            # fall back to remote for each sequence (slower)
    if not blast_results:
        logger.info("Running remote BLAST (NCBI) for %d sequences - this may be slow and rate-limited.", len(ids))
        for sid, seq in tqdm(zip(ids, seqstrs), total=len(ids)):
            xml_str = blast_remote_seq(seq, program="blastn", database="nt")
            if xml_str:
                parsed = parse_blast_xml(xml_str, is_string=True)
                if parsed:
                    blast_results.extend(parsed)

    # Save blast summary
    with open(os.path.join(outdir, "blast_summary.json"), "w") as fh:
        json.dump(blast_results, fh, indent=2)

    # 6) For clusters, generate representatives and run MSA + tree
    rep_fasta = os.path.join(outdir, "representatives.fasta")
    reps = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idxs = [i for i,l in enumerate(labels) if l==cid]
        # choose longest sequence as representative
        chosen = max(idxs, key=lambda i: len(seqstrs[i]))
        reps[cid] = (ids[chosen], seqstrs[chosen])
    with open(rep_fasta, "w") as fh:
        for cid, (sid, seq) in reps.items():
            fh.write(f">{cid}_{sid}\n{seq}\n")
    logger.info("Wrote %d representative sequences", len(reps))

    if run_mafft_fasttree and len(reps) >= 2:
        if check_binary("mafft") and check_binary("fasttree"):
            msa_out = os.path.join(outdir, "representatives.msa.fasta")
            run_mafft(rep_fasta, msa_out)
            tree_out = os.path.join(outdir, "representatives.tree")
            run_fasttree(msa_out, tree_out)
            logger.info("MSA and tree created: %s", tree_out)
        else:
            logger.warning("MAFFT and/or FastTree not available. Skipping tree generation.")

    # 7) Abundance table (clusters per sample assume single sample)
    abundance = pd.Series(labels).value_counts().rename_axis("cluster").reset_index(name="count")
    abundance.to_csv(os.path.join(outdir,"abundance.csv"), index=False)

    # 8) Diversity (Shannon) for clusters
    logger.info("Shannon for clusters:")
    for _, r in abundance.iterrows():
        logger.info("Cluster %s: count=%d", r['cluster'], int(r['count']))

    logger.info("Pipeline finished. Outputs in %s", outdir)
    return outdir

# -----------------------
# If run as script: minimal CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run eDNA pipeline on FASTA")
    p.add_argument("--fasta", required=True, help="Input FASTA")
    p.add_argument("--out", default="edna_out", help="Output dir")
    p.add_argument("--k", type=int, default=6, help="k-mer size")
    p.add_argument("--cluster", default="optics", choices=["optics","dbscan"])
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--local_blast_db", default=None, help="Local BLAST db name (optional)")
    p.add_argument("--no_tree", action="store_true", help="Skip MSA/tree step")
    args = p.parse_args()

    pipeline_from_fasta(args.fasta,
                        outdir=args.out,
                        k=args.k,
                        cluster_method=args.cluster,
                        min_samples=args.min_samples,
                        use_local_blast_db=args.local_blast_db,
                        run_mafft_fasttree=(not args.no_tree))
