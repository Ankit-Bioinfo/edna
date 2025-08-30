import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import plotly.express as px

# -------------------------------
# Helper Functions
# -------------------------------

def generate_fake_sequences(n=200, length=100):
    """Generate mock DNA sequences"""
    bases = ["A", "T", "G", "C"]
    return ["".join(np.random.choice(bases, length)) for _ in range(n)]

def kmer_features(seqs, k=5):
    """Convert DNA sequences to k-mer count vectors"""
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k,k))
    X = vectorizer.fit_transform(seqs)
    return X.toarray(), vectorizer.get_feature_names_out()

def compute_diversity(labels):
    """Shannon diversity index & richness"""
    counts = pd.Series(labels).value_counts()
    props = counts / counts.sum()
    shannon = -(props * np.log(props + 1e-9)).sum()  # avoid log(0)
    richness = counts.count()
    return shannon, richness

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌊 eDNA Pipeline Prototype")
st.markdown("""
Simulated eDNA pipeline: from raw sequences → clustering → biodiversity metrics → visualization.
""")

# User parameters
n_seqs = st.sidebar.slider("Number of sequences", 50, 500, 200)
seq_len = st.sidebar.slider("Sequence length", 50, 200, 100)
k = st.sidebar.slider("k-mer size", 3, 6, 5)

# Step 1: Generate sequences
st.subheader("Step 1: Raw Sequences")
sequences = generate_fake_sequences(n_seqs, seq_len)
st.write("Example sequences:", sequences[:5])

# Step 2: k-mer embedding
st.subheader("Step 2: K-mer Embedding")
X, features = kmer_features(sequences, k=k)
st.write(f"Feature matrix shape: {X.shape}")

# Step 3: Dimensionality reduction
st.subheader("Step 3: PCA Dimensionality Reduction")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 4: Clustering (HDBSCAN)
st.subheader("Step 4: Clustering / Novel Taxa Discovery")
clusterer = HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(X)
st.write(f"Number of clusters detected (excluding noise): {len(set(labels)) - (1 if -1 in labels else 0)}")

# Step 5: Diversity metrics
st.subheader("Step 5: Biodiversity Metrics")
shannon, richness = compute_diversity(labels)
st.metric("Shannon Diversity Index", round(shannon, 3))
st.metric("Species Richness", richness)

# Step 6: Visualize clusters
st.subheader("Step 6: Cluster Visualization")
fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
                 title="Sequence Clusters (Potential Taxa)")
st.plotly_chart(fig)

# Step 7: Cluster abundance heatmap
st.subheader("Step 7: Cluster Abundance Heatmap")
cluster_counts = pd.Series(labels).value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Abundance"]
fig2 = px.bar(cluster_counts, x="Cluster", y="Abundance", title="Cluster Abundance")
st.plotly_chart(fig2)

st.success("✅ Pipeline prototype complete! You can now plug in real eDNA FASTQ sequences and extend with DADA2/BLAST/AI embeddings.")
