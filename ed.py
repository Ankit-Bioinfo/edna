# edna_pipeline_app.py
"""
Streamlit app: eDNA pipeline for marine / unknown organisms.
Single-file pipeline:
 - FASTA upload (single file or zip of FASTAs)
 - QC light (length filter)
 - k-mer vectorization (configurable k)
 - dimensionality reduction (PCA, optional UMAP / t-SNE)
 - clustering (OPTICS / DBSCAN) with safe parameter handling
 - BLAST: local blastn if present & DB provided; otherwise remote NCBI qblast fallback (rate-limited)
 - MSA via 'mafft' if available (graceful fallback)
 - Phylogeny: FastTree if available else hierarchical clustering + dendrogram
 - Abundance table, Shannon entropy, plots and downloads
"""

import os
import io
import sys
import zipfile
import json
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict
from itertools import product

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.metrics import pairwise_distances

# phylo fallback
from scipy.cluster.hierarchy import linkage, dendrogram

# Biopython
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# try optional modules
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from tqdm import tqdm
except Exception:
    # simple fallback
    def tqdm(x, **kwargs):
        return x

# --------- Utility helpers ---------

def check_binary(name: str) -> bool:
    return shutil.which(name) is not None

def safe_makedir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def read_fasta_handle(handle) -> List[Tuple[str,str]]:
    recs = []
    for rec in SeqIO.parse(handle, "fasta"):
        recs.append((rec.id, str(rec.seq).upper()))
    return recs

def read_fasta_path(path: str) -> List[Tuple[str,str]]:
    with open(path, "r") as fh:
        return read_fasta_handle(fh)

def extract_from_zip(uploaded_zip) -> Dict[str, List[Tuple[str,str]]]:
    # returns dict: filename -> list of (id, seq)
    results = {}
    with tempfile.TemporaryDirectory() as td:
        zpath = os.path.join(td, "upload.zip")
        with open(zpath, "wb") as f:
            f.write(uploaded_zip.getvalue())
        with zipfile.ZipFile(zpath, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".fasta", ".fa", ".fna", ".ffn")):
                    zf.extract(name, td)
                    p = os.path.join(td, name)
                    try:
                        recs = read_fasta_path(p)
                        results[name] = recs
                    except Exception:
                        continue
    return results

# --------- k-mer functions ---------

def build_kmer_vocab(k: int) -> List[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]

def seq_to_kmer_vector(seq: str, k: int, vocab_idx: Dict[str,int]) -> np.ndarray:
    L = len(vocab_idx)
    vec = np.zeros(L, dtype=np.float32)
    s = seq.upper()
    for i in range(len(s)-k+1):
        kmer = s[i:i+k]
        if all(ch in "ACGT" for ch in kmer):
            idx = vocab_idx.get(kmer)
            if idx is not None:
                vec[idx] += 1
    ssum = vec.sum()
    if ssum > 0:
        vec /= ssum
    return vec

def batch_kmers(seqs: List[str], k: int) -> np.ndarray:
    if len(seqs) == 0:
        return np.zeros((0, 4**k), dtype=np.float32)
    vocab = build_kmer_vocab(k)
    vocab_idx = {kmer:i for i,kmer in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i] = seq_to_kmer_vector(s, k, vocab_idx)
    return X

# --------- Embedding & reduction ---------

def reduce_dimensionality(X: np.ndarray, method: str="pca", n_components: int=50):
    if X.shape[0] == 0:
        return X
    n_components = min(n_components, X.shape[1], X.shape[0])
    if method == "pca":
        pca = PCA(n_components=n_components)
        Z = pca.fit_transform(X)
        return Z
    elif method == "umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X)
    else:
        # fallback to PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

# --------- Clustering (safe) ---------

def safe_cluster(Z: np.ndarray, method: str="optics", min_samples: int=10, xi: float=0.05, eps: float=0.5):
    """
    Safe clustering: ensures min_samples <= number of samples, prevents crash
    """
    n_samples = Z.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)
    
    # Clamp min_samples to [2, n_samples]
    min_samples_clamped = max(2, min(int(min_samples), n_samples))
    
    if method == "optics":
        model = OPTICS(min_samples=min_samples_clamped, xi=xi, metric="euclidean")
    elif method == "dbscan":
        model = DBSCAN(min_samples=min_samples_clamped, eps=eps, metric="euclidean")
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    labels = model.fit_predict(Z)
    return labels

# --------- BLAST helpers ---------

def run_local_blast(query_fasta: str, dbname: str, out_xml: str, evalue: float=1e-5, threads: int=1):
    if not check_binary("blastn"):
        raise FileNotFoundError("blastn not found in PATH")
    cmd = ["blastn", "-query", query_fasta, "-db", dbname, "-outfmt", "5", "-out", out_xml, "-evalue", str(evalue), "-num_threads", str(threads)]
    subprocess.check_call(cmd)
    return out_xml

def run_remote_blast_seq(seq: str, program: str="blastn", database: str="nt"):
    try:
        handle = NCBIWWW.qblast(program, database, seq)
        return handle.read()  # XML text
    except Exception as e:
        st.warning(f"Remote BLAST failed for one sequence: {e}")
        return None

def parse_blast_xml_string(xml_str: str):
    # returns list of dicts for first record
    out = []
    try:
        handle = io.StringIO(xml_str)
        record = NCBIXML.read(handle)
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                out.append({
                    "title": alignment.title,
                    "accession": alignment.accession,
                    "length": alignment.length,
                    "evalue": hsp.expect,
                    "identity": hsp.identities,
                    "align_len": hsp.align_length
                })
    except Exception:
        pass
    return out

# --------- MSA & phylogeny wrappers ---------

def run_mafft(in_fasta: str, out_fasta: str, options: str="--auto"):
    if not check_binary("mafft"):
        raise FileNotFoundError("mafft not available on PATH")
    cmd = ["mafft"] + options.split() + [in_fasta]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"mafft failed: {stderr}")
    with open(out_fasta, "w") as fh:
        fh.write(stdout)
    return out_fasta

def run_fasttree(msa_fasta: str, out_tree: str):
    if not check_binary("fasttree"):
        raise FileNotFoundError("fasttree not available")
    cmd = ["fasttree", "-nt", msa_fasta]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"fasttree failed: {stderr}")
    with open(out_tree, "w") as fh:
        fh.write(stdout)
    return out_tree

def hierarchical_tree_from_seqs(seqs: List[str], ids: List[str], metric: str="euclidean"):
    """
    Fallback phylogeny: compute pairwise distances on k-mer vectors and produce a linkage matrix
    """
    X = batch_kmers(seqs, k=4)
    D = pairwise_distances(X, metric="cosine")
    Z = linkage(D, method="average")
    return Z

# --------- diversity & abundance ---------

def shannon(counts):
    counts = np.array(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# --------- Streamlit UI ---------

st.set_page_config(page_title="eDNA Smart Pipeline", layout="wide")
st.title("eDNA Smart Pipeline — marine / unknown organisms")
st.markdown("Upload FASTA(s), run k-mer embedding, clustering, BLAST, MSA and phylogeny.")

# --------- [Sidebar and main UI remain unchanged] ---------
# ... [rest of your original UI code continues exactly as in your file] ...

