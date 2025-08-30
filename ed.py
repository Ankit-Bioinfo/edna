# edna_app.py
# Single-file Streamlit app implementing the AI-driven eDNA pipeline
# Run with: streamlit run edna_app.py

import io
import os
from typing import List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
import streamlit as st
from itertools import product

# ML imports
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.semi_supervised import LabelPropagation

# plotting
import matplotlib.pyplot as plt

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(page_title="eDNA AI Pipeline", layout="wide")
st.title("eDNA AI Pipeline — k-mer + clustering + semi-supervised annotation")

# -------------------------
# Utility functions
# -------------------------

def detect_format(file_name: str) -> str:
    fn = file_name.lower()
    if fn.endswith((".fastq", ".fq")):
        return "fastq"
    return "fasta"

def read_sequences_from_uploaded(uploaded_file) -> List[Tuple[str, str]]:
    """Return list of (seq_id, sequence) from a single uploaded file-like object."""
    fmt = detect_format(uploaded_file.name)
    uploaded_file.seek(0)
    records = list(SeqIO.parse(io.TextIOWrapper(uploaded_file, encoding="utf-8"), fmt))
    return [(rec.id, str(rec.seq).upper()) for rec in records]

def read_sequences_from_path(path: str) -> Iterator[Tuple[str, str]]:
    fmt = "fastq" if path.lower().endswith(("fastq","fq")) else "fasta"
    for rec in SeqIO.parse(path, fmt):
        yield rec.id, str(rec.seq).upper()

def filter_seq_keep_acgt(seq: str) -> str:
    return "".join([c for c in seq if c in "ACGT"])

def build_kmer_vocab(k: int):
    keys = ["".join(p) for p in product("ACGT", repeat=k)]
    return {kmer:i for i,kmer in enumerate(keys)}

def seq_to_kmer_vector(seq: str, k: int, vocab: dict) -> np.ndarray:
    L = len(vocab)
    vec = np.zeros(L, dtype=np.float32)
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if kmer in vocab:
            vec[vocab[kmer]] += 1
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec

def batch_kmers(seqs: List[str], k: int) -> np.ndarray:
    vocab = build_kmer_vocab(k)
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s2 = filter_seq_keep_acgt(s)
        if len(s2) >= k:
            X[i] = seq_to_kmer_vector(s2, k, vocab)
    return X

def encode_with_autoencoder_or_pca(X: np.ndarray, latent_dim: int=64) -> np.ndarray:
    if X.shape[0] == 0:
        return X
    n_components = min(latent_dim, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, svd_solver="randomized")
    return pca.fit_transform(X)

def cluster_embeddings(Z: np.ndarray, method: str="optics",
                       min_samples: int=10, xi: float=0.05, eps: float=0.5):
    n_samples = Z.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)

    # ensure min_samples is valid
    min_samples = max(2, min(int(min_samples), n_samples))

    if method == "optics":
        model = OPTICS(min_samples=min_samples, xi=xi, metric="euclidean")
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return model.fit_predict(Z)

def summarize_abundance(sample_ids: List[str], seq_ids: List[str], clusters: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame({"sample_id": sample_ids, "seq_id": seq_ids, "cluster": clusters})
    tab = (rows.groupby(["sample_id", "cluster"])["seq_id"]
           .count().reset_index(name="count"))
    pivot = tab.pivot(index="sample_id", columns="cluster", values="count").fillna(0).astype(int)
    pivot = pivot.rename_axis(None, axis=1).reset_index()
    return rows, pivot

def semi_supervised_annotate(seq_ids: List[str], embeddings: np.ndarray,
                             reference_df: Optional[pd.DataFrame]):
    out = pd.DataFrame({"seq_id": seq_ids})
    if reference_df is None or reference_df.shape[0] == 0:
        out["taxon"] = None
        out["confidence"] = 0.0
        return out

    id2index = {sid: i for i, sid in enumerate(seq_ids)}
    y = np.full(len(seq_ids), -1, dtype=int)

    taxa = sorted(reference_df["taxon"].dropna().unique().tolist())
    tax2id = {t: i for i, t in enumerate(taxa)}

    for _, r in reference_df.iterrows():
        sid = str(r["seq_id"])
        tid = r.get("taxon", None)
        if sid in id2index and pd.notna(tid):
            y[id2index[sid]] = tax2id[tid]

    if (y >= 0).sum() == 0:
        out["taxon"] = None
        out["confidence"] = 0.0
        return out

    lp = LabelPropagation(kernel="rbf", gamma=0.5, max_iter=1000)
    lp.fit(embeddings, y)

    pred = lp.transduction_
    conf = lp.label_distributions_.max(axis=1)
    inv = {v: k for k, v in tax2id.items()}

    out["taxon"] = [inv.get(int(i), None) if i >= 0 else None for i in pred]
    out["confidence"] = conf
    return out

# -------------------------
# Streamlit UI
# -------------------------

with st.sidebar:
    st.header("Settings")
    k = st.number_input("k-mer size (k)", value=6, min_value=3, max_value=10, step=1)
    latent_dim = st.number_input("latent dim (PCA components)", value=64, min_value=2, max_value=512, step=1)
    cluster_method = st.selectbox("Clustering method", options=["optics", "dbscan"])
    min_samples = st.number_input("min_samples", value=10, min_value=2, step=1)
    xi = st.slider("OPTICS xi", min_value=0.01, max_value=0.2, value=0.05)
    eps = st.number_input("DBSCAN eps", value=0.5, format="%.3f")
    show_raw = st.checkbox("Show raw sequences table", value=False)

st.subheader("1) Upload data")
uploaded = st.file_uploader("Upload FASTA/FASTQ", type=["fasta","fa","fastq","fq"])
ref_uploaded = st.file_uploader("Optional: reference CSV (seq_id,taxon)", type=["csv"])
gold_uploaded = st.file_uploader("Optional: gold labels CSV (seq_id,taxon)", type=["csv"])

if uploaded is None:
    st.info("Upload a FASTA/FASTQ file to start.")
    st.stop()

# Load sequences
try:
    seq_pairs = read_sequences_from_uploaded(uploaded)
except Exception as e:
    st.error(f"Failed to read sequences: {e}")
    st.stop()

if len(seq_pairs) == 0:
    st.warning("No sequences found in file.")
    st.stop()

seq_ids = [sid for sid, seq in seq_pairs]
seqs = [seq for sid, seq in seq_pairs]
st.success(f"Loaded {len(seqs)} sequences")

if show_raw:
    st.dataframe(pd.DataFrame({"seq_id": seq_ids, "sequence": [s[:200] + ("..." if len(s) > 200 else "") for s in seqs]}))

# Embeddings
st.subheader("2) Build k-mer embeddings")
with st.spinner("Computing k-mer vectors..."):
    X = batch_kmers(seqs, k=k)
st.write("Embeddings shape:", X.shape)

with st.spinner("Encoding with PCA..."):
    Z = encode_with_autoencoder_or_pca(X, latent_dim=latent_dim)
st.write("Latent shape:", Z.shape)

# Clustering
st.subheader("3) Clustering")
with st.spinner("Clustering embeddings..."):
    labels = cluster_embeddings(Z, method=cluster_method,
                                min_samples=min_samples, xi=xi, eps=eps)
st.write("Unique cluster labels (−1 = noise):", np.unique(labels))

rows_df, abundance_df = summarize_abundance(["sample"]*len(seq_ids), seq_ids, labels)
st.dataframe(rows_df.head(200))

# Semi-supervised annotation
st.subheader("4) Optional: semi-supervised annotation")
reference_df = None
if ref_uploaded:
    try:
        reference_df = pd.read_csv(ref_uploaded)
        st.write("Reference rows:", len(reference_df))
        st.dataframe(reference_df.head(10))
    except Exception as e:
        st.error("Couldn't read reference CSV: " + str(e))

annot_df = semi_supervised_annotate(seq_ids, Z, reference_df)
st.dataframe(annot_df.head(10))

# Abundance
st.subheader("5) Abundance table")
st.dataframe(abundance_df.head(50))

# Metrics
st.subheader("6) Metrics (if gold labels provided)")
metrics = {}
if gold_uploaded:
    try:
        gold = pd.read_csv(gold_uploaded)
        merged = annot_df.merge(gold, on="seq_id", how="inner", suffixes=("","_gold"))
        subset = merged[merged["taxon"].notna() & merged["taxon_gold"].notna()]
        if len(subset) > 0:
            metrics["annotation_accuracy"] = float((subset["taxon"] == subset["taxon_gold"]).mean())

        purities = []
        for c, grp in merged.groupby(labels):
            if c == -1:
                continue
            if "taxon_gold" in grp.columns and len(grp) > 0:
                purities.append(grp["taxon_gold"].value_counts(normalize=True).max())
        if purities:
            metrics["cluster_purity_mean"] = float(sum(purities)/len(purities))

        st.json(metrics)
    except Exception as e:
        st.error("Failed to read gold CSV: " + str(e))

# Plots
st.subheader("7) Plots")
try:
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Sequence counts per cluster")
    st.pyplot(fig)
except Exception as e:
    st.write("Plot error:", e)

# Downloads
st.subheader("Download results")
out_dir = "edna_results"
os.makedirs(out_dir, exist_ok=True)

pd.DataFrame({"seq_id": seq_ids, "cluster": labels}).to_csv(os.path.join(out_dir, "clusters.csv"), index=False)
annot_df.to_csv(os.path.join(out_dir, "taxonomy.csv"), index=False)
abundance_df.to_csv(os.path.join(out_dir, "abundance.csv"), index=False)

with open(os.path.join(out_dir, "clusters.csv"), "rb") as f:
    st.download_button("Download clusters.csv", f, file_name="clusters.csv", mime="text/csv")
with open(os.path.join(out_dir, "taxonomy.csv"), "rb") as f:
    st.download_button("Download taxonomy.csv", f, file_name="taxonomy.csv", mime="text/csv")
with open(os.path.join(out_dir, "abundance.csv"), "rb") as f:
    st.download_button("Download abundance.csv", f, file_name="abundance.csv", mime="text/csv")

st.success("Pipeline finished — adjust settings in the sidebar and re-run as needed.")
