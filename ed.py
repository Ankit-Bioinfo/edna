# app.py
import streamlit as st
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import umap
import hdbscan
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI eDNA Analysis", layout="wide")
st.title("AI-based eDNA Analysis Pipeline")

# Upload FASTA/FASTQ file
uploaded_file = st.file_uploader("Upload FASTA/FASTQ file", type=['fasta', 'fastq'])

# k-mer size selection
k = st.slider("Select k-mer size", 3, 8, 4)

# Function to generate k-mers
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

# Dummy classifier for known species (replace with real model later)
def train_dummy_classifier(k):
    sequences = ["ATGCGT", "CGTATG", "TTGCAA", "AATTGC", "GGCCAA"]
    labels = ["Species_A", "Species_A", "Species_B", "Species_B", "Species_C"]
    kmers_list = [" ".join(generate_kmers(seq, k)) for seq in sequences]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers_list)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, labels)
    return clf, vectorizer

classifier, vectorizer = train_dummy_classifier(k)

if uploaded_file:
    st.info("Reading sequences...")
    
    try:
        sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fastq")]
        if len(sequences) == 0:
            sequences = [str(record.seq) for record in SeqIO.parse(uploaded_file, "fasta")]
    except Exception as e:
        st.error(f"Error reading sequences: {e}")
        st.stop()
    
    st.success(f"Total sequences loaded: {len(sequences)}")
    
    # Convert sequences to k-mers
    kmers_list = [" ".join(generate_kmers(seq, k)) for seq in sequences]
    
    # Vectorize sequences
    X_input = vectorizer.transform(kmers_list)
    
    # Predict known species
    st.info("Predicting known species...")
    predictions = classifier.predict(X_input)
    df_pred = pd.DataFrame({'Sequence': sequences, 'Predicted_Species': predictions})
    st.subheader("Predicted Species")
    st.dataframe(df_pred)
    
    # Dimensionality reduction with UMAP
    st.info("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_reduced = reducer.fit_transform(X_input.toarray())
    
    # Clustering with HDBSCAN
    st.info("Clustering sequences with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X_reduced)
    
    # Plot UMAP clusters
    st.subheader("UMAP Clustering of Sequences")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='Spectral', s=20)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    # Cluster + species summary
    df_summary = pd.DataFrame({'Cluster': labels, 'Predicted_Species': predictions})
    cluster_counts = df_summary.groupby(['Cluster', 'Predicted_Species']).size().reset_index(name='Count')
    st.subheader("Cluster and Species Summary")
    st.dataframe(cluster_counts)
    
    # Download CSV
    csv = cluster_counts.to_csv(index=False)
    st.download_button("Download Cluster + Species Summary CSV", data=csv, file_name="edna_summary.csv")
