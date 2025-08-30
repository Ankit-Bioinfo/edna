# app.py
import streamlit as st
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="AI eDNA Analysis", layout="wide")
st.title("Functional AI-based eDNA Analysis Pipeline")

# -------------------------------
# Upload FASTA/FASTQ file
# -------------------------------
uploaded_file = st.file_uploader("Upload FASTA/FASTQ file", type=['fasta', 'fastq'])

# -------------------------------
# k-mer size slider
# -------------------------------
k = st.slider("Select k-mer size", min_value=3, max_value=8, value=4)

# -------------------------------
# Function to generate k-mers
# -------------------------------
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

# -------------------------------
# Main pipeline
# -------------------------------
if uploaded_file:
    st.info("Reading sequences from uploaded file...")
    
    sequences = []
    # Try FASTQ first
    try:
        sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fastq")]
        if len(sequences) == 0:
            uploaded_file.seek(0)  # Reset file pointer
            sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fasta")]
    except Exception as e:
        st.error(f"Error reading sequences: {e}")
        st.stop()
    
    if len(sequences) == 0:
        st.error("No sequences found in the file!")
        st.stop()
    
    st.success(f"Total sequences loaded: {len(sequences)}")
    
    # -------------------------------
    # Convert sequences to k-mers
    # -------------------------------
    kmers_list = [" ".join(generate_kmers(seq, k)) for seq in sequences]
    
    # -------------------------------
    # Vectorize k-mers
    # -------------------------------
    st.info("Vectorizing sequences using CountVectorizer...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers_list)
    
    # -------------------------------
    # Dimensionality reduction using UMAP
    # -------------------------------
    st.info("Reducing dimensionality with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_reduced = reducer.fit_transform(X.toarray())
    
    # -------------------------------
    # Clustering sequences with HDBSCAN
    # -------------------------------
    st.info("Clustering sequences with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X_reduced)
    
    # -------------------------------
    # UMAP scatter plot
    # -------------------------------
    st.subheader("UMAP Clustering of DNA Reads")
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='Spectral', s=20)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    # -------------------------------
    # Cluster summary table
    # -------------------------------
    df_summary = pd.DataFrame({'Sequence': sequences, 'Cluster': labels})
    cluster_counts = df_summary['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Coun]()_
