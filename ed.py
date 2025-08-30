# edna_ai_streamlit_app.py
"""
Enhanced Streamlit eDNA pipeline app with AI assistant.
Features:
- FASTA/ZIP upload
- QC (length filter)
- k-mer vectorization
- Dimensionality reduction (PCA/UMAP)
- Clustering (OPTICS/DBSCAN)
- BLAST (local/remote)
- MSA via MAFFT
- Phylogenetic tree (FastTree/fallback)
- Diversity metrics
- Interactive Plotly visualizations
- AI assistant (ChatGPT) for insights
"""

import os
import io
import sys
import zipfile
import json
import shutil
import tempfile
import subprocess
from itertools import product
from typing import List, Tuple, Dict

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# -------------------- Utility --------------------

def check_binary(name: str) -> bool:
    return shutil.which(name) is not None

def safe_makedir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def read_fasta_handle(handle) -> List[Tuple[str,str]]:
    return [(rec.id, str(rec.seq).upper()) for rec in SeqIO.parse(handle, "fasta")]

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
                        results[name] = read_fasta_path(p)
                    except Exception:
                        continue
    return results

# -------------------- k-mer --------------------

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

# -------------------- Dimensionality reduction --------------------

def reduce_dimensionality(X: np.ndarray, method: str="pca", n_components: int=50):
    if X.shape[0] == 0:
        return X
    n_components = min(n_components, X.shape[1], X.shape[0])
    if method == "pca":
        return PCA(n_components=n_components).fit_transform(X)
    elif method == "umap" and HAVE_UMAP:
        return umap.UMAP(n_components=n_components, random_state=42).fit_transform(X)
    else:
        return PCA(n_components=n_components).fit_transform(X)

# -------------------- Clustering --------------------

def safe_cluster(Z: np.ndarray, method: str="optics", min_samples: int=10, xi: float=0.05, eps: float=0.5):
    n_samples = Z.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)
    min_samples_clamped = max(2, min(int(min_samples), n_samples))
    if method == "optics":
        model = OPTICS(min_samples=min_samples_clamped, xi=xi, metric="euclidean")
    else:
        model = DBSCAN(min_samples=min_samples_clamped, eps=eps)
    return model.fit_predict(Z)

# -------------------- BLAST --------------------

def run_local_blast(query_fasta: str, dbname: str, out_xml: str, evalue: float=1e-5, threads: int=1):
    if not check_binary("blastn"):
        raise FileNotFoundError("blastn not found in PATH")
    cmd = ["blastn", "-query", query_fasta, "-db", dbname, "-outfmt", "5", "-out", out_xml, "-evalue", str(evalue), "-num_threads", str(threads)]
    subprocess.check_call(cmd)
    return out_xml

def run_remote_blast_seq(seq: str, program: str="blastn", database: str="nt"):
    try:
        return NCBIWWW.qblast(program, database, seq).read()
    except Exception:
        return None

def parse_blast_xml_string(xml_str: str):
    out = []
    try:
        handle = io.StringIO(xml_str)
        record = NCBIXML.read(handle)
        for aln in record.alignments:
            for hsp in aln.hsps:
                out.append({"title": aln.title, "accession": aln.accession, "length": aln.length, "evalue": hsp.expect, "identity": hsp.identities, "align_len": hsp.align_length})
    except Exception:
        pass
    return out

# -------------------- MSA & phylogeny --------------------

def run_mafft(in_fasta: str, out_fasta: str, options: str="--auto"):
    if not check_binary("mafft"):
        raise FileNotFoundError("mafft not found")
    cmd = ["mafft"] + options.split() + [in_fasta]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {stderr}")
    with open(out_fasta, "w") as fh:
        fh.write(stdout)
    return out_fasta

def run_fasttree(msa_fasta: str, out_tree: str):
    if not check_binary("fasttree"):
        raise FileNotFoundError("FastTree not found")
    cmd = ["fasttree", "-nt", msa_fasta]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FastTree failed: {stderr}")
    with open(out_tree, "w") as fh:
        fh.write(stdout)
    return out_tree

# -------------------- Diversity --------------------

def shannon(counts):
    counts = np.array(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p>0]
    return -np.sum(p*np.log(p))

# -------------------- Hierarchical fallback --------------------

def hierarchical_tree_from_seqs(seqs: List[str], ids: List[str], metric: str="euclidean"):
    X = batch_kmers(seqs, k=4)
    D = pairwise_distances(X, metric="cosine")
    Z = linkage(D, method="average")
    return Z

# -------------------- Streamlit --------------------

st.set_page_config(page_title="eDNA AI Pipeline", layout="wide")
st.title("Enhanced eDNA Pipeline with AI Assistant")
st.markdown("Upload sequences, run analysis, and get AI insights.")

# Sidebar settings
st.sidebar.header("Settings")
k_mer = st.sidebar.slider("k-mer size", 4, 8, 6)
embed_method = st.sidebar.selectbox("Dimensionality reduction", ["pca", "umap"] if HAVE_UMAP else ["pca"])
pca_components = st.sidebar.number_input("PCA components", 2, 200, 50)
cluster_method = st.sidebar.selectbox("Clustering", ["optics", "dbscan"])
min_samples = st.sidebar.number_input("min_samples", 2, 50, 10)
optics_xi = st.sidebar.slider("OPTICS xi", 0.01, 0.2, 0.05)
dbscan_eps = st.sidebar.number_input("DBSCAN eps", 0.01, 5.0, 0.5)
filter_min_len = st.sidebar.number_input("Min sequence length", 50, 10000, 80)
use_local_blast = st.sidebar.checkbox("Use local BLAST", False)
local_blast_db = None
if use_local_blast:
    local_blast_db = st.sidebar.text_input("Local BLAST DB")
run_mafft_opt = st.sidebar.checkbox("Run MAFFT", True)
run_fasttree_opt = st.sidebar.checkbox("Run FastTree", True)

# Upload
st.header("Upload sequences")
upload_mode = st.radio("Upload mode", ["Single FASTA", "ZIP of FASTAs"])
uploaded = st.file_uploader("Upload your file", type=["fasta","fa","fna","zip"])

if uploaded is None:
    st.stop()

# Load sequences
seq_collections = {}
if upload_mode=="Single FASTA":
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
filtered_collections = {fname:[(rid,seq) for rid,seq in recs if len(seq)>=filter_min_len] for fname,recs in seq_collections.items()}
total_seqs = sum(len(v) for v in filtered_collections.values())
st.write(f"Total sequences after filter: {total_seqs}")

# Flatten
sample_map, seq_ids, seqs_flat = [], [], []
for sample,recs in filtered_collections.items():
    for rid, seq in recs:
        sample_map.append(sample)
        seq_ids.append(rid)
        seqs_flat.append(seq)

if len(seqs_flat)==0:
    st.error("No sequences remain after filtering")
    st.stop()

# k-mer embedding
with st.spinner("Computing k-mer vectors..."):
    X = batch_kmers(seqs_flat, k=k_mer)
st.write("k-mer matrix shape:", X.shape)

# Dimensionality reduction
with st.spinner("Reducing dimensionality..."):
    Z = reduce_dimensionality(X, method=embed_method, n_components=min(pca_components, X.shape[1], X.shape[0]))
st.write("Embedding shape:", Z.shape)

# Clustering
with st.spinner("Clustering..."):
    labels = safe_cluster(Z, method=cluster_method, min_samples=min_samples, xi=optics_xi, eps=dbscan_eps)
st.write("Unique clusters (−1=noise):", np.unique(labels))

# Cluster table
df_clusters = pd.DataFrame({"sample":sample_map, "seq_id":seq_ids, "cluster":labels})
ab_table = df_clusters.groupby(["sample","cluster"])["seq_id"].count().reset_index(name="count")
ab_pivot = ab_table.pivot(index="sample", columns="cluster", values="count").fillna(0).astype(int).reset_index()

st.subheader("Cluster assignments")
st.dataframe(df_clusters.head(200))
st.subheader("Abundance")
st.dataframe(ab_pivot)

# Diversity
st.subheader("Diversity (Shannon)")
shannon_per_sample = {sample:shannon(group["count"].values) for sample, group in ab_table.groupby("sample")}
st.dataframe(pd.DataFrame.from_dict(shannon_per_sample, orient="index", columns=["Shannon"]).reset_index().rename(columns={"index":"sample"}))

# Interactive Plotly cluster plot
st.subheader("Cluster embedding")
fig = px.scatter(x=Z[:,0], y=Z[:,1], color=[str(l) for l in labels], hover_data=[seq_ids, sample_map])
st.plotly_chart(fig)

# AI Assistant
st.subheader("AI Assistant")
st.markdown("Ask AI about clusters, diversity, or sequences")
ai_input = st.text_area("Enter your question")
if st.button("Ask AI"):
    if ai_input.strip():
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            context = f"Clusters: {df_clusters.head(50).to_dict()}, Abundance: {ab_pivot.head().to_dict()}",
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role":"system","content":"You are a bioinformatics assistant."},
                    {"role":"user","content":ai_input+str(context)}
                ]
            )
            st.markdown(response['choices'][0]['message']['content'])
        except Exception as e:
            st.error(f"AI Assistant error: {e}")
