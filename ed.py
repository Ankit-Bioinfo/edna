import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from Bio import SeqIO

# -------------------------------
# Function to parse FASTA
# -------------------------------
def parse_fasta(file, k=3):
    """Convert fasta sequences into simple k-mer frequency vectors."""
    sequences = []
    ids = []

    for record in SeqIO.parse(file, "fasta"):
        seq = str(record.seq).upper()
        ids.append(record.id)
        sequences.append(seq)

    # Generate k-mers
    from itertools import product
    kmers = ["".join(p) for p in product("ACGT", repeat=k)]

    # Convert sequences into feature vectors
    features = []
    for seq in sequences:
        vec = []
        for kmer in kmers:
            vec.append(seq.count(kmer))
        features.append(vec)

    df = pd.DataFrame(features, columns=kmers, index=ids)
    return df


# -------------------------------
# Clustering function
# -------------------------------
def cluster_embeddings(Z, method="optics", min_samples=2, xi=0.05, eps=0.5):
    if method == "optics":
        model = OPTICS(min_samples=min_samples, xi=xi, max_eps=eps)
        return model.fit_predict(Z)
    else:
        raise ValueError("Unsupported clustering method")


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧬 eDNA Clustering App with FASTA Support")

uploaded_file = st.file_uploader("Upload a CSV, TSV, or FASTA file", type=["csv", "tsv", "fasta", "fa"])

min_samples = st.slider("Minimum Samples", min_value=2, max_value=100, value=5, step=1)
xi = st.slider("Xi (for OPTICS)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
eps = st.slider("Max Eps", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

if uploaded_file:
    filename = uploaded_file.name.lower()

    if filename.endswith((".csv", ".tsv")):
        sep = "," if filename.endswith(".csv") else "\t"
        df = pd.read_csv(uploaded_file, sep=sep)
        st.write("📄 Data Preview:", df.head())
        Z = df.values

    elif filename.endswith((".fasta", ".fa")):
        df = parse_fasta(uploaded_file, k=3)
        st.write("🧬 Parsed FASTA to k-mer matrix:", df.head())
        Z = df.values

    else:
        st.error("Unsupported file type")
        st.stop()

    # Run clustering
    try:
        labels = cluster_embeddings(Z, method="optics", min_samples=min_samples, xi=xi, eps=eps)
        df["Cluster"] = labels
        st.success("✅ Clustering complete!")
        st.write(df)

    except ValueError as e:
        st.error(f"Error: {e}")
