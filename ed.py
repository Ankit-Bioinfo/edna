# app.py
import streamlit as st
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI eDNA Analysis", layout="wide")
st.title("Functional AI-based eDNA Analysis Pipeline")

# Upload FASTA/FASTQ file
uploaded_file = st.file_uploader("Upload FASTA/FASTQ file", type=['fasta', 'fastq'])

# k-mer size selection
k = st.slider("Select k-mer size", 3, 8, 4)

# Function to generate k-mers from a sequence
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

if uploaded_file:
    st.info("Reading sequences...")
    
    # Read sequences
    try:
        sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fastq")]
        if len(sequences) == 0:
            sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fasta")]
    except Exception as e:
        st.error(f"Error reading sequences: {e}")
        st.stop()
    
    if len(sequences) == 0:
        st.error("No sequences found in the file!")
        st.stop()
    
    st.success(f"Total sequences loaded: {len(sequences)}")
    
    # Convert sequences to k-mers
    kmers_list = [" ".join(generate_kmers(seq, k)) for seq in sequences]
    
    # Vectorize k-mers
    st.info("Vectorizing sequences with CountVectorizer...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers_list)
    
    # Dimensionality reduction with UMAP
    st.info("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_reduced = reducer.fit_transform(X.toarray())
    
    # Clustering with HDBSCAN
    st.info("Clustering sequences with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X_reduced)
    
    # Plot UMAP clusters
    st.subheader("UMAP Clustering of DNA Reads")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='Spectral', s=20)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    # Cluster summary table
    df_summary = pd.DataFrame({'Sequence': sequences, 'Cluster': labels})
    cluster_counts = df_summary['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    st.subheader("Cluster Summary")
    st.dataframe(cluster_counts)
    
    # Download CSV
    csv = df_summary.to_csv(index=False)
    st.download_button("Download Clustered Sequences CSV", data=csv, file_name="clustered_sequences.csv")
