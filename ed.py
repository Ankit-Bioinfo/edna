# edna_pipeline_app.py
"""
Streamlit app: eDNA pipeline for marine / unknown organisms.
Includes:
 - FASTA upload (single or zip)
 - Sequence filtering
 - k-mer vectorization
 - Dimensionality reduction (PCA/UMAP)
 - Clustering (OPTICS/DBSCAN) with safe min_samples
 - BLAST (local or remote)
 - MSA (MAFFT) and phylogeny (FastTree fallback)
 - Abundance, diversity, and download tables
"""

import os
import io
import sys
import zipfile
import shutil
import tempfile
import subprocess
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

# Optional modules
try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# ---------------- Utility ----------------

def check_binary(name: str) -> bool:
    return shutil.which(name) is not None

def safe_makedir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def read_fasta_handle(handle) -> List[Tuple[str,str]]:
    recs = [(rec.id, str(rec.seq).upper()) for rec in SeqIO.parse(handle, "fasta")]
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

# ---------------- K-mer ----------------

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

# ---------------- Dimensionality Reduction ----------------

def reduce_dimensionality(X: np.ndarray, method: str="pca", n_components: int=50):
    if X.shape[0] == 0:
        return X
    n_components = min(n_components, X.shape[1], X.shape[0])
    if method.lower() == "pca":
        return PCA(n_components=n_components).fit_transform(X)
    elif method.lower() == "umap" and HAVE_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X)
    else:
        return PCA(n_components=n_components).fit_transform(X)

# ---------------- Clustering (Safe) ----------------

def safe_cluster(Z: np.ndarray, method: str="optics", min_samples: int=10, xi: float=0.05, eps: float=0.5):
    n_samples = Z.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)
    # clamp min_samples to [2, n_samples]
    min_samples_clamped = max(2, min(min_samples, n_samples))
    if min_samples_clamped != min_samples:
        st.info(f"min_samples adjusted from {min_samples} → {min_samples_clamped} to match n_samples={n_samples}")
    if method.lower() == "optics":
        model = OPTICS(min_samples=min_samples_clamped, xi=xi, metric="euclidean")
    else:
        model = DBSCAN(min_samples=min_samples_clamped, eps=eps)
    labels = model.fit_predict(Z)
    return labels

# ---------------- BLAST ----------------

def run_local_blast(query_fasta: str, dbname: str, out_xml: str, evalue: float=1e-5, threads: int=1):
    if not check_binary("blastn"):
        raise FileNotFoundError("blastn not found in PATH")
    cmd = ["blastn", "-query", query_fasta, "-db", dbname, "-outfmt", "5",
           "-out", out_xml, "-evalue", str(evalue), "-num_threads", str(threads)]
    subprocess.check_call(cmd)
    return out_xml

def run_remote_blast_seq(seq: str, program: str="blastn", database: str="nt"):
    try:
        handle = NCBIWWW.qblast(program, database, seq)
        return handle.read()
    except Exception:
        return None

def parse_blast_xml_string(xml_str: str):
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

# ---------------- MSA & Phylogeny ----------------

def run_mafft(in_fasta: str, out_fasta: str, options: str="--auto"):
    if not check_binary("mafft"):
        raise FileNotFoundError("mafft not available")
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

def hierarchical_tree_from_seqs(seqs: List[str]):
    X = batch_kmers(seqs, k=4)
    D = pairwise_distances(X, metric="cosine")
    Z = linkage(D, method="average")
    return Z

# ---------------- Diversity ----------------

def shannon(counts):
    counts = np.array(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="eDNA Smart Pipeline", layout="wide")
st.title("eDNA Smart Pipeline — marine / unknown organisms")

# Sidebar
st.sidebar.header("Settings")
k_mer = st.sidebar.slider("k-mer size (k)", 4, 8, 6)
embed_method = st.sidebar.selectbox("Embedding reduction", ["pca", "umap"] if HAVE_UMAP else ["pca"])
pca_components = st.sidebar.number_input("PCA components", 2, 200, 50)
cluster_method = st.sidebar.selectbox("Clustering method", ["optics", "dbscan"])
min_samples = st.sidebar.number_input("min_samples (cluster)", 2, 20, 10)
optics_xi = st.sidebar.slider("OPTICS xi", 0.01, 0.2, 0.05)
dbscan_eps = st.sidebar.number_input("DBSCAN eps", 0.01, 10.0, 0.5, format="%.3f")
filter_min_len = st.sidebar.number_input("Minimum sequence length (bp)", 50, 10000, 80)
use_local_blast = st.sidebar.checkbox("Use local BLAST (blastn)", False)
local_blast_db = None
if use_local_blast:
    local_blast_db = st.sidebar.text_input("Local BLAST DB name")
run_mafft_opt = st.sidebar.checkbox("Enable MAFFT", True)
run_fasttree_opt = st.sidebar.checkbox("Enable FastTree", False)

# File upload
st.header("Upload data")
upload_mode = st.radio("Upload mode", ("Single FASTA", "Zip of FASTAs"))
uploaded = None
if upload_mode == "Single FASTA":
    uploaded = st.file_uploader("Upload FASTA", type=["fasta", "fa", "fna"])
else:
    uploaded = st.file_uploader("Upload ZIP of FASTA files", type=["zip"])

if uploaded is None:
    st.info("Upload FASTA or ZIP to start.")
    st.stop()

# Load sequences
seq_collections = {}
if upload_mode == "Single FASTA":
    uploaded.seek(0)
    seqs = read_fasta_handle(io.TextIOWrapper(uploaded, encoding="utf-8"))
    seq_collections[uploaded.name or "input.fasta"] = seqs
else:
    seq_collections = extract_from_zip(uploaded)

# Filter by length
filtered_collections = {}
for fname, recs in seq_collections.items():
    flt = [(rid, seq) for rid, seq in recs if len(seq) >= filter_min_len]
    filtered_collections[fname] = flt

total_seqs = sum(len(v) for v in filtered_collections.values())
st.write(f"Total sequences (after length filter): {total_seqs}")
if total_seqs < 2:
    st.warning("Not enough sequences for clustering or tree. Upload more sequences or reduce min length.")
    st.stop()

# Flatten sequences
sample_map, seq_ids, seqs_list = [], [], []
for sample, recs in filtered_collections.items():
    for rid, seq in recs:
        sample_map.append(sample)
        seq_ids.append(rid)
        seqs_list.append(seq)

st.info(f"Processing {len(seqs_list)} sequences from {len(filtered_collections)} samples.")

# K-mer embeddings
with st.spinner("Computing k-mer vectors..."):
    X = batch_kmers(seqs_list, k=k_mer)
st.write("k-mer matrix shape:", X.shape)

# Dimensionality reduction
with st.spinner("Reducing dimensionality..."):
    Z = reduce_dimensionality(X, method=embed_method, n_components=min(pca_components, X.shape[1], X.shape[0]))
st.write("Embedding shape:", Z.shape)

# Clustering
with st.spinner("Clustering..."):
    labels = safe_cluster(Z, method=cluster_method,
                          min_samples=min_samples, xi=optics_xi, eps=dbscan_eps)
st.write("Unique cluster labels (-1 = noise):", np.unique(labels))

# Cluster table
df_clusters = pd.DataFrame({
    "sample": sample_map,
    "seq_id": seq_ids,
    "cluster": labels
})
ab_table = df_clusters.groupby(["sample","cluster"])["seq_id"].count().reset_index(name="count")
ab_pivot = ab_table.pivot(index="sample", columns="cluster", values="count").fillna(0).astype(int).reset_index()

st.header("Results")
st.subheader("Cluster assignments (first 200 rows)")
st.dataframe(df_clusters.head(200))
st.subheader("Abundance (sample x cluster)")
st.dataframe(ab_pivot)

# Diversity
st.subheader("Diversity (Shannon index)")
shannon_per_sample = {s: shannon(g["count"].values) for s, g in ab_table.groupby("sample")}
st.dataframe(pd.DataFrame.from_dict(shannon_per_sample, orient="index", columns=["Shannon"]).reset_index().rename(columns={"index":"sample"}))

# Representatives, MSA, Tree
st.subheader("Representative sequences & Phylogeny")
if st.button("Select representatives & run MSA/Tree"):
    outdir = tempfile.mkdtemp(prefix="edna_out_")
    reps = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idxs = [i for i, lab in enumerate(labels) if lab == c]
        chosen_i = max(idxs, key=lambda ii: len(seqs_list[ii]))
        reps[c] = (seq_ids[chosen_i], seqs_list[chosen_i])
    st.write(f"Selected {len(reps)} representative sequences.")
    rep_fasta = os.path.join(outdir, "representatives.fasta")
    with open(rep_fasta, "w") as fh:
        for cid, (rid, seq) in reps.items():
            fh.write(f">{cid}_{rid}\n{seq}\n")
    st.success(f"Wrote {rep_fasta}")

    # MAFFT
    msa_file = os.path.join(outdir, "representatives.msa.fasta")
    if run_mafft_opt and check_binary("mafft") and len(reps)>=2:
        try:
            run_mafft(rep_fasta, msa_file)
            st.success("MAFFT MSA created.")
        except Exception as e:
            st.error(f"MAFFT failed: {e}")
            msa_file = None
    else:
        st.warning("MAFFT not run.")

    # FastTree
    tree_file = os.path.join(outdir, "representatives.tree")
    if run_fasttree_opt and check_binary("fasttree") and os.path.exists(msa_file or ""):
        try:
            run_fasttree(msa_file, tree_file)
            st.success("FastTree created.")
        except Exception as e:
            st.error(f"FastTree failed: {e}")
            tree_file = None
    else:
        st.info("FastTree not run. Using hierarchical tree fallback.")
        rep_seqs = [s for _, s in reps.values()]
        if len(rep_seqs) >= 2:
            Zlink = hierarchical_tree_from_seqs(rep_seqs)
            fig, ax = plt.subplots(figsize=(8, max(4, len(rep_seqs)*0.3)))
            dendrogram(Zlink, labels=[f"{cid}_{rid}" for cid,(rid,_) in reps.items()], orientation="right")
            st.pyplot(fig)
        else:
            st.warning("Not enough representatives for tree.")

st.success("Pipeline complete — adjust settings or re-run steps as needed.")
