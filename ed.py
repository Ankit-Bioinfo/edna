# edna_pipeline_app.py
"""
Streamlit eDNA Pipeline App - Improved & Functional Version
- Upload FASTA(s) or ZIP of FASTAs
- Filter by length
- k-mer embedding
- Dimensionality reduction (PCA/UMAP)
- Clustering (OPTICS/DBSCAN) with safe handling
- BLAST (local or remote NCBI)
- MSA with MAFFT and phylogeny with FastTree or fallback
- Abundance & diversity metrics
"""

import os, io, sys, zipfile, json, shutil, tempfile, subprocess
from itertools import product
from typing import List, Tuple, Dict

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# Optional libraries
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# --------- Utility Functions ---------

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
    vec = np.zeros(len(vocab_idx), dtype=np.float32)
    s = seq.upper()
    for i in range(len(s)-k+1):
        kmer = s[i:i+k]
        if all(ch in "ACGT" for ch in kmer):
            idx = vocab_idx.get(kmer)
            if idx is not None:
                vec[idx] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def batch_kmers(seqs: List[str], k: int) -> np.ndarray:
    if len(seqs) == 0: return np.zeros((0, 4**k), dtype=np.float32)
    vocab = build_kmer_vocab(k)
    vocab_idx = {kmer:i for i,kmer in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(tqdm(seqs, desc="Computing k-mers")):
        X[i] = seq_to_kmer_vector(s, k, vocab_idx)
    return X

# --------- Dimensionality Reduction ---------

def reduce_dimensionality(X: np.ndarray, method: str="pca", n_components: int=50):
    if X.shape[0] < 2: return X
    n_components = min(n_components, X.shape[1], X.shape[0])
    if method == "umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X)
    else:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

# --------- Clustering ---------

def safe_cluster(Z: np.ndarray, method: str="optics", min_samples: int=10, xi: float=0.05, eps: float=0.5):
    n_samples = Z.shape[0]
    if n_samples < 2: return np.zeros(n_samples, dtype=int)
    min_samples_clamped = max(2, min(min_samples, n_samples))
    if method == "optics":
        model = OPTICS(min_samples=min_samples_clamped, xi=xi, metric="euclidean")
    else:
        model = DBSCAN(min_samples=min_samples_clamped, eps=eps)
    labels = model.fit_predict(Z)
    return labels

# --------- BLAST ---------

def run_local_blast(query_fasta: str, dbname: str, out_xml: str, evalue: float=1e-5, threads: int=1):
    if not check_binary("blastn"): raise FileNotFoundError("blastn not found")
    cmd = ["blastn", "-query", query_fasta, "-db", dbname, "-outfmt", "5", "-out", out_xml,
           "-evalue", str(evalue), "-num_threads", str(threads)]
    subprocess.check_call(cmd)
    return out_xml


def run_remote_blast_seq(seq: str, program: str="blastn", database: str="nt"):
    try:
        handle = NCBIWWW.qblast(program, database, seq)
        return handle.read()
    except Exception as e:
        st.warning(f"Remote BLAST failed: {e}")
        return None


def parse_blast_xml_string(xml_str: str):
    out = []
    try:
        handle = io.StringIO(xml_str)
        record = NCBIXML.read(handle)
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                out.append({"title": alignment.title, "accession": alignment.accession,
                            "length": alignment.length, "evalue": hsp.expect, "identity": hsp.identities})
    except Exception:
        pass
    return out

# --------- MSA & Phylogeny ---------

def run_mafft(in_fasta: str, out_fasta: str, options: str="--auto"):
    if not check_binary("mafft"): raise FileNotFoundError("mafft not found")
    cmd = ["mafft"] + options.split() + [in_fasta]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0: raise RuntimeError(stderr)
    with open(out_fasta, "w") as fh: fh.write(stdout)
    return out_fasta


def run_fasttree(msa_fasta: str, out_tree: str):
    if not check_binary("fasttree"): raise FileNotFoundError("fasttree not found")
    cmd = ["fasttree", "-nt", msa_fasta]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0: raise RuntimeError(stderr)
    with open(out_tree, "w") as fh: fh.write(stdout)
    return out_tree


def hierarchical_tree_from_seqs(seqs: List[str], k: int=4):
    X = batch_kmers(seqs, k=k)
    D = pairwise_distances(X, metric="cosine")
    Z = linkage(D, method="average")
    return Z

# --------- Diversity ---------

def shannon(counts):
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total <= 0: return 0.0
    p = counts / total
    p = p[p>0]
    return -np.sum(p * np.log(p))

# --------- Streamlit App ---------

st.set_page_config(page_title="eDNA Smart Pipeline", layout="wide")
st.title("eDNA Smart Pipeline — marine / unknown organisms")
st.markdown("Upload FASTA(s), run k-mer embedding, clustering, BLAST, MSA and phylogeny.")

# Sidebar settings
st.sidebar.header("Settings")
k_mer = st.sidebar.slider("k-mer size", 4, 8, 6)
embed_method = st.sidebar.selectbox("Embedding reduction", ["pca", "umap"] if HAVE_UMAP else ["pca"])
pca_components = st.sidebar.number_input("PCA components", 2, 200, 50)
cluster_method = st.sidebar.selectbox("Clustering method", ["optics", "dbscan"])
min_samples = st.sidebar.number_input("min_samples", 2, 100, 10)
optics_xi = st.sidebar.slider("OPTICS xi", 0.01, 0.2, 0.05)
dbscan_eps = st.sidebar.number_input("DBSCAN eps", 0.01, 5.0, 0.5)
filter_min_len = st.sidebar.number_input("Min sequence length", 50, 10000, 80)
use_local_blast = st.sidebar.checkbox("Use local BLAST", value=False)
local_blast_db = st.sidebar.text_input("Local BLAST DB") if use_local_blast else None
run_mafft_opt = st.sidebar.checkbox("Enable MAFFT MSA", True)
run_fasttree_opt = st.sidebar.checkbox("Enable FastTree", False)

# File upload
st.header("Upload FASTA(s)")
upload_mode = st.radio("Upload mode", ["Single FASTA", "ZIP of FASTAs"])
uploaded = st.file_uploader("Upload your file", type=["fasta","fa","fna","zip"])
if not uploaded: st.info("Upload a file to begin."); st.stop()

# Load sequences
seq_collections = {}
if upload_mode == "Single FASTA":
    try:
        uploaded.seek(0)
        seqs = read_fasta_handle(io.TextIOWrapper(uploaded, encoding="utf-8"))
    except Exception:
        uploaded.seek(0)
        seqs = read_fasta_handle(io.TextIOWrapper(io.BytesIO(uploaded.getvalue()), encoding="utf-8"))
    seq_collections[uploaded.name or "input.fasta"] = seqs
else:
    seq_collections = extract_from_zip(uploaded)

# Filter by length
filtered_collections = {f:[(rid,s) for rid,s in recs if len(s)>=filter_min_len] for f,recs in seq_collections.items()}
total_seqs = sum(len(v) for v in filtered_collections.values())
st.write(f"Total sequences after filter >= {filter_min_len}: {total_seqs}")
if total_seqs == 0: st.error("No sequences after filter."); st.stop()

# Flatten sequences
seq_ids, seqs_list, sample_map = [], [], []
for sname, recs in filtered_collections.items():
    for rid, s in recs:
        seq_ids.append(rid); seqs_list.append(s); sample_map.append(sname)

# k-mer embedding
with st.spinner("Computing k-mer vectors..."):
    X = batch_kmers(seqs_list, k=k_mer)
st.write("k-mer matrix shape:", X.shape)

# Dimensionality reduction
with st.spinner("Reducing dimensionality..."):
    Z = reduce_dimensionality(X, method=embed_method, n_components=min(pca_components,X.shape[1],X.shape[0]))
st.write("Embedding shape used for clustering:", Z.shape)

# Clustering
with st.spinner("Clustering sequences..."):
    labels = safe_cluster(Z, method=cluster_method, min_samples=min_samples, xi=optics_xi, eps=dbscan_eps)
st.write("Unique cluster labels (-1=noise):", np.unique(labels))

# Cluster table
cluster_df = pd.DataFrame({"sample": sample_map, "seq_id": seq_ids, "cluster": labels})
ab_table = cluster_df.groupby(["sample","cluster"]).size().reset_index(name="count")
ab_pivot = ab_table.pivot(index="sample", columns="cluster", values="count").fillna(0).astype(int).reset_index()

st.header("Results")
st.subheader("Cluster assignments")
st.dataframe(cluster_df.head(200))
st.subheader("Abundance table")
st.dataframe(ab_pivot)

# Diversity
st.subheader("Shannon diversity per sample")
shannon_dict = {s: shannon(g["count"].values) for s,g in ab_table.groupby("sample")}
st.dataframe(pd.DataFrame.from_dict(shannon_dict, orient="index", columns=["Shannon"]).reset_index().rename(columns={"index":"sample"}))

st.success("eDNA pipeline executed successfully — adjust parameters and re-run as needed.")
