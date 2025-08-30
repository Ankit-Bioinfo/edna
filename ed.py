# app.py
import streamlit as st
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI eDNA Analysis", layout="wide")
st.title("AI-based eDNA Analysis Pipeline")

# Upload FASTA or FASTQ file
uploaded_file = st.file_uploader("Upload FASTA/FASTQ file", type=['fasta', 'fastq'])

# k-mer size
k = st.slider("Select k-mer size", 3, 8, 4)

# Function to generate k-mers from a sequence
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

if uploaded_file:
    st.info("Reading sequences...")
    
    # Read sequences from uploaded file
    sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fastq")]
    st.success(f"Total sequences loaded: {len(sequences)}")
    
    # Convert sequences to k-mers
    kmers_list = [" ".join(generate_kmers(seq, k)) for seq in sequences]
    
    st.info("Vectorizing sequences...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers_list)
    
    st.info("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_reduced = reducer.fit_transform(X.toarray())
    
    st.info("Clustering sequences with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(X_reduced)
    
    # Plotting UMAP clusters
    st.subheader("UMAP Clustering of DNA Reads")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='Spectral', s=10)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    # Cluster summary table
    df_summary = pd.DataFrame({'Cluster': labels})
    cluster_counts = df_summary['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    st.subheader("Cluster Summary")
    st.dataframe(cluster_counts)
    
    # Optionally, download cluster summary
    csv = cluster_counts.to_csv(index=False)
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
