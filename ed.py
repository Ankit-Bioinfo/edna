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
    n_samples = Z.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)
    # clamp min_samples
    min_samples_clamped = max(2, min(int(min_samples), n_samples))
    if method == "optics":
        model = OPTICS(min_samples=min_samples_clamped, xi=xi, metric="euclidean")
    else:
        model = DBSCAN(min_samples=min_samples_clamped, eps=eps)
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
    # capture stdout
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
    Fallback phylogeny: compute pairwise distances on k-mer vectors or raw sequences (cheap)
    and produce a linkage matrix for dendrogram plotting.
    """
    # simple distance: pairwise identity fraction based on short k-mer (k=4)
    X = batch_kmers(seqs, k=4)
    # use pairwise distances
    D = pairwise_distances(X, metric="cosine")
    # linkage
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
st.markdown("Upload FASTA(s), run k-mer embedding, clustering, BLAST, MSA and phylogeny. App uses local binaries if available and falls back to safe alternatives.")

# Sidebar settings
st.sidebar.header("Settings")
k_mer = st.sidebar.slider("k-mer size (k)", min_value=4, max_value=8, value=6, step=1)
embed_method = st.sidebar.selectbox("Embedding reduction", ["pca", "umap"] if HAVE_UMAP else ["pca"])
pca_components = st.sidebar.number_input("PCA components (for clustering)", min_value=2, max_value=200, value=50)
cluster_method = st.sidebar.selectbox("Clustering method", ["optics", "dbscan"])
min_samples = st.sidebar.number_input("min_samples (cluster)", min_value=2, value=10)
optics_xi = st.sidebar.slider("OPTICS xi", min_value=0.01, max_value=0.2, value=0.05)
dbscan_eps = st.sidebar.number_input("DBSCAN eps", min_value=0.01, value=0.5, format="%.3f")
filter_min_len = st.sidebar.number_input("Minimum sequence length (bp)", min_value=50, value=80)
use_local_blast = st.sidebar.checkbox("Use local BLAST (blastn)", value=False)
local_blast_db = None
if use_local_blast:
    local_blast_db = st.sidebar.text_input("Local BLAST DB name (makeblastdb formatted)")
run_mafft_opt = st.sidebar.checkbox("Enable MAFFT for MSA (if installed)", value=True)
run_fasttree_opt = st.sidebar.checkbox("Enable FastTree (if installed)", value=False)
reference_csv = st.sidebar.file_uploader("Optional: reference CSV (seq_id,taxon) for semi-supervised propagation", type=["csv"])

# file input
st.header("Upload data")
upload_mode = st.radio("Upload a single FASTA file or a zip of FASTAs?", ("Single FASTA", "Zip of FASTAs"))
uploaded = None
if upload_mode == "Single FASTA":
    uploaded = st.file_uploader("Upload FASTA", type=["fasta", "fa", "fna"])
else:
    uploaded = st.file_uploader("Upload ZIP containing FASTA files", type=["zip"])

if uploaded is None:
    st.info("Upload FASTA or zip to begin. Try the example test data if you don't have one.")
    st.stop()

# load sequences
seq_collections = {}  # name -> list of (id, seq)
if upload_mode == "Single FASTA":
    try:
        uploaded.seek(0)
        seqs = read_fasta_handle(io.TextIOWrapper(uploaded, encoding="utf-8"))
    except Exception:
        # try reading bytes
        uploaded.seek(0)
        seqs = read_fasta_handle(io.TextIOWrapper(io.BytesIO(uploaded.getvalue()), encoding="utf-8"))
    seq_collections[uploaded.name or "input.fasta"] = seqs
else:
    seq_collections = extract_from_zip(uploaded)

# preprocessing: filter by length
filtered_collections = {}
for fname, recs in seq_collections.items():
    flt = [(rid, seq) for rid, seq in recs if len(seq) >= filter_min_len]
    filtered_collections[fname] = flt

total_seqs = sum(len(v) for v in filtered_collections.values())
st.write(f"Total sequences (after length filter >= {filter_min_len}): {total_seqs}")

# flatten sequences for analysis; maintain sample mapping
sample_ids = []
seq_ids = []
seqs = []
sample_map = []
for sample_name, recs in filtered_collections.items():
    for rid, seq in recs:
        sample_ids.append(sample_name)
        seq_ids.append(rid)
        seqs.append(seq)
        sample_map.append(sample_name)

if len(seqs) == 0:
    st.error("No sequences remain after filtering. Increase sequence length threshold or upload different files.")
    st.stop()

st.info(f"Processing {len(seqs)} sequences from {len(filtered_collections)} sample(s).")

# k-mer embeddings
with st.spinner("Computing k-mer vectors..."):
    X = batch_kmers(seqs, k=k_mer)
st.write("k-mer matrix shape:", X.shape)

# dimensionality reduction for clustering
with st.spinner("Reducing dimensionality..."):
    Z = reduce_dimensionality(X, method=embed_method, n_components=min(pca_components, X.shape[1], X.shape[0]))
st.write("Embedding shape used for clustering:", Z.shape)

# clustering
with st.spinner("Clustering (this may take a moment)..."):
    labels = safe_cluster(Z, method=cluster_method,
                          min_samples=min_samples, xi=optics_xi, eps=dbscan_eps)
st.write("Unique cluster labels (−1 = noise):", np.unique(labels))

# build results tables
df_clusters = pd.DataFrame({
    "sample": sample_map,
    "seq_id": seq_ids,
    "cluster": labels
})
# abundance table
ab_table = df_clusters.groupby(["sample", "cluster"])["seq_id"].count().reset_index(name="count")
ab_pivot = ab_table.pivot(index="sample", columns="cluster", values="count").fillna(0).astype(int).reset_index()

st.header("Results")
st.subheader("Cluster assignments (first 200 rows)")
st.dataframe(df_clusters.head(200))

st.subheader("Abundance (sample x cluster)")
st.dataframe(ab_pivot)

# diversity
st.subheader("Diversity metrics")
shannon_per_sample = {}
for sample, group in ab_table.groupby("sample"):
    counts = group["count"].values
    shannon_per_sample[sample] = shannon(counts)
st.dataframe(pd.DataFrame.from_dict(shannon_per_sample, orient="index", columns=["Shannon"]).reset_index().rename(columns={"index":"sample"}))

# semi-supervised annotation (reference)
ref_df = None
if reference_csv is not None:
    try:
        reference_csv.seek(0)
        ref_df = pd.read_csv(reference_csv)
        st.write("Loaded reference CSV (first 10 rows):")
        st.dataframe(ref_df.head(10))
    except Exception as e:
        st.error(f"Failed to read reference CSV: {e}")

# annotate with BLAST (local or remote)
st.subheader("Taxonomic annotation (BLAST)")
annotation_results = []
if st.button("Run taxonomic annotation (BLAST)"):
    tmpdir = tempfile.mkdtemp(prefix="edna_blast_")
    try:
        if use_local_blast and local_blast_db:
            # write query
            qf = os.path.join(tmpdir, "queries.fasta")
            with open(qf, "w") as fh:
                for sid, seq in zip(seq_ids, seqs):
                    fh.write(f">{sid}\n{seq}\n")
            out_xml = os.path.join(tmpdir, "blast_local.xml")
            if not check_binary("blastn"):
                st.error("Local blast requested but 'blastn' not found in PATH.")
            else:
                st.info("Running local blastn...")
                try:
                    run_local = run_local_blast  # from earlier definitions
                except NameError:
                    # define fallback local blast wrapper here
                    def run_local_blast(query_fasta, db, out_xml, evalue=1e-5, threads=1):
                        cmd = ["blastn", "-query", query_fasta, "-db", db, "-outfmt", "5", "-out", out_xml, "-evalue", str(evalue), "-num_threads", str(threads)]
                        subprocess.check_call(cmd)
                        return out_xml
                    run_local = run_local_blast
                try:
                    run_local(qf, local_blast_db, out_xml, evalue=1e-5, threads=2)
                    st.success("Local BLAST finished. Parsing...")
                    # parse XML for each query (simple approach: parse file)
                    parsed = []
                    with open(out_xml) as fh:
                        # parse entire file into records
                        for rec in NCBIXML.parse(fh):
                            hits = []
                            for aln in rec.alignments[:3]:
                                for hsp in aln.hsps:
                                    hits.append({
                                        "query": rec.query,
                                        "title": aln.title,
                                        "accession": aln.accession,
                                        "evalue": hsp.expect,
                                        "identity": hsp.identities
                                    })
                            parsed.append(hits)
                        annotation_results = parsed
                except Exception as e:
                    st.error(f"Local BLAST error: {e}")
        else:
            # remote per-sequence BLAST (very slow for many sequences)
            st.info("Running remote BLAST (NCBI). This is slow and rate-limited — use local BLAST for large batches.")
            parsed_all = {}
            for sid, seq in tqdm(zip(seq_ids, seqs), total=len(seqs)):
                xml_text = run_remote_blast_seq(seq, program="blastn", database="nt")
                if xml_text:
                    hits = parse_blast_xml_string(xml_text)
                    parsed_all[sid] = hits[:3]
            annotation_results = parsed_all
        st.success("Annotation step finished.")
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# show annotation preview
if annotation_results:
    st.subheader("BLAST annotation preview")
    try:
        # if list-like from local parse
        if isinstance(annotation_results, dict):
            rows = []
            for q, hits in annotation_results.items():
                for h in hits:
                    rows.append({"query": q, "title": h.get("title"), "acc": h.get("accession"), "evalue": h.get("evalue")})
            st.dataframe(pd.DataFrame(rows).head(200))
        else:
            # maybe list-of-lists
            rows = []
            for hits in annotation_results:
                for h in hits:
                    rows.append(h)
            st.dataframe(pd.DataFrame(rows).head(200))
    except Exception as e:
        st.write("Cannot display annotations:", e)

# representatives & MSA & Tree
st.subheader("Representatives / MSA / Phylogeny")
if st.button("Select representatives & run MSA + tree (if tools present)"):
    outdir = tempfile.mkdtemp(prefix="edna_out_")
    try:
        # choose one representative per non-noise cluster: longest sequence
        reps = {}
        for c in sorted(set(labels)):
            if c == -1:
                continue
            idxs = [i for i, lab in enumerate(labels) if lab == c]
            chosen_i = max(idxs, key=lambda ii: len(seqs[ii]))
            reps[c] = (seq_ids[chosen_i], seqs[chosen_i])
        st.write(f"Selected {len(reps)} representative sequences.")
        rep_fasta = os.path.join(outdir, "representatives.fasta")
        with open(rep_fasta, "w") as fh:
            for cid, (rid, seq) in reps.items():
                fh.write(f">{cid}_{rid}\n{seq}\n")
        st.success(f"Wrote {rep_fasta}")

        # run MAFFT if available and requested
        msa_file = os.path.join(outdir, "representatives.msa.fasta")
        if run_mafft_opt and check_binary("mafft") and len(reps) >= 2:
            st.info("Running MAFFT...")
            try:
                run_mafft(rep_fasta, msa_file)
                st.success("MAFFT MSA created.")
            except Exception as e:
                st.error(f"MAFFT failed: {e}")
                msa_file = None
        else:
            st.warning("MAFFT not run (either disabled, not available, or not enough representatives).")

        # run FastTree if available
        tree_file = os.path.join(outdir, "representatives.tree")
        if run_fasttree_opt and check_binary("fasttree") and os.path.exists(msa_file or ""):
            st.info("Running FastTree...")
            try:
                run_fasttree(msa_file, tree_file)
                st.success("FastTree created.")
            except Exception as e:
                st.error(f"FastTree failed: {e}")
                tree_file = None
        else:
            st.info("FastTree not used or not available. Producing hierarchical clustering tree (fallback).")
            # fallback: hierarchical tree on kmer distances of representatives
            rep_seqs = [s for _, s in reps.values()]
            rep_ids = [rid for _, rid in reps.values()]
            if len(rep_seqs) >= 2:
                Zlink = hierarchical_tree_from_seqs(rep_seqs, rep_ids)
                # plot dendrogram
                fig, ax = plt.subplots(figsize=(8, max(4, len(rep_seqs)*0.3)))
                dendrogram(Zlink, labels=[f"{cid}_{rid}" for cid, (rid, _) in reps.items()], orientation="right")
                st.pyplot(fig)
            else:
                st.warning("Not enough representatives to build a tree.")

        # if tree_file exists, display simple text
        if os.path.exists(tree_file):
            st.subheader("Newick tree (first 5000 chars):")
            with open(tree_file) as fh:
                st.text(fh.read()[:5000])

    finally:
        # keep outdir so user can download; do not delete immediately
        st.info(f"Results stored in {outdir} — download below.")
        # list files for download
        files = []
        for fname in os.listdir(outdir):
            files.append(os.path.join(outdir, fname))
        for p in files:
            st.download_button(f"Download {os.path.basename(p)}", open(p, "rb"), file_name=os.path.basename(p))

# Provide TSV/CSV downloads for tables
st.subheader("Downloads")
if st.button("Download cluster assignments (CSV)"):
    st.download_button("clusters.csv", df_clusters.to_csv(index=False).encode("utf-8"), file_name="clusters.csv", mime="text/csv")

if st.button("Download abundance pivot (CSV)"):
    st.download_button("abundance.csv", ab_pivot.to_csv(index=False).encode("utf-8"), file_name="abundance.csv", mime="text/csv")

st.success("Pipeline processing complete — adjust settings or re-run steps as needed.")
