import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
import matplotlib.pyplot as plt

# -----------------------------
# Helper Functions
# -----------------------------

def load_embeddings(file):
    """
    Load embeddings from a CSV/TSV file.
    Assumes first column is ID and rest are embedding values.
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        df = pd.read_csv(file, sep="\t")

    # Separate IDs and embedding matrix
    ids = df.iloc[:, 0].values
    Z = df.iloc[:, 1:].values
    return ids, Z


def cluster_embeddings(Z, method="optics", min_samples=5, xi=0.05, eps=0.5):
    """
    Cluster embeddings using OPTICS or DBSCAN with safe parameter handling.
    """
    n_samples = Z.shape[0]

    # Fix: Ensure min_samples is not greater than dataset size
    if min_samples > n_samples:
        min_samples = max(2, n_samples // 2)

    if method == "optics":
        model = OPTICS(min_samples=min_samples, xi=xi)
    elif method == "dbscan":
        model = DBSCAN(min_samples=min_samples, eps=eps)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = model.fit_predict(Z)
    return labels


def plot_clusters(Z, labels):
    """
    Plot 2D scatter plot of clusters (first 2 dimensions).
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab20", s=50)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("Clustering of Embeddings")
    st.pyplot(plt)


# -----------------------------
# Streamlit App UI
# -----------------------------

st.title("🧬 eDNA Clustering App")
st.markdown("Upload your embedding dataset (CSV/TSV) and cluster sequences.")

uploaded_file = st.file_uploader("Upload Embeddings File", type=["csv", "tsv"])

if uploaded_file:
    ids, Z = load_embeddings(uploaded_file)
    st.success(f"✅ Loaded {len(ids)} embeddings with {Z.shape[1]} features.")

    st.sidebar.header("Clustering Settings")
    method = st.sidebar.selectbox("Clustering Method", ["optics", "dbscan"])

    if method == "optics":
        min_samples = st.sidebar.slider("min_samples", 2, 50, 5)
        xi = st.sidebar.slider("xi (OPTICS)", 0.01, 0.5, 0.05)
        labels = cluster_embeddings(Z, method="optics", min_samples=min_samples, xi=xi)
    else:
        min_samples = st.sidebar.slider("min_samples", 2, 50, 5)
        eps = st.sidebar.slider("eps (DBSCAN)", 0.1, 5.0, 0.5)
        labels = cluster_embeddings(Z, method="dbscan", min_samples=min_samples, eps=eps)

    # Show results
    st.subheader("📊 Clustering Results")
    result_df = pd.DataFrame({"ID": ids, "Cluster": labels})
    st.dataframe(result_df)

    # Plot
    st.subheader("📈 Cluster Visualization")
    plot_clusters(Z, labels)

    # Option to download results
    st.download_button("Download Cluster Results", result_df.to_csv(index=False).encode("utf-8"),
                       "clusters.csv", "text/csv")
