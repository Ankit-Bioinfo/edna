# edna_ai_pipeline_app.py
"""
End-to-End eDNA Pipeline with AI Assistant
"""

import os, io, zipfile, tempfile, subprocess, shutil
from itertools import product
from typing import List, Dict, Tuple
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

# optional modules
try:
    import umap
    HAVE_UMAP = True
except:
    HAVE_UMAP = False

try:
    from tqdm import tqdm
except:
    def tqdm(x, **kwargs): return x

# ChatGPT API
try:
    import openai
    HAVE_OPENAI = True
except:
    HAVE_OPENAI = False

# --------- Helpers ---------

def check_binary(name): return shutil.which(name) is not None
def safe_makedir(path): os.makedirs(path, exist_ok=True); return path

def read_fasta_handle(handle) -> List[Tuple[str,str]]:
    recs = [(rec.id, str(rec.seq).upper()) for rec in SeqIO.parse(handle, "fasta")]
    return recs

def read_fasta_path(path): 
    with open(path, "r") as fh: return read_fasta_handle(fh)

def extract_from_zip(uploaded) -> Dict[str, List[Tuple[str,str]]]:
    results = {}
    with tempfile.TemporaryDirectory() as td:
        zpath = os.path.join(td, "upload.zip")
        with open(zpath,"wb") as f: f.write(uploaded.getvalue())
        with zipfile.ZipFile(zpath,"r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".fasta",".fa",".fna",".ffn")):
                    zf.extract(name, td)
                    try: results[name]=read_fasta_path(os.path.join(td,name))
                    except: continue
    return results

# --------- k-mer ---------

def build_kmer_vocab(k): return ["".join(p) for p in product("ACGT", repeat=k)]

def seq_to_kmer_vector(seq:str, k:int, vocab_idx:Dict[str,int])->np.ndarray:
    vec = np.zeros(len(vocab_idx), dtype=np.float32)
    s = seq.upper()
    for i in range(len(s)-k+1):
        kmer = s[i:i+k]
        if all(ch in "ACGT" for ch in kmer):
            idx = vocab_idx.get(kmer)
            if idx is not None: vec[idx]+=1
    ssum = vec.sum()
    if ssum>0: vec/=ssum
    return vec

def batch_kmers(seqs:List[str], k:int)->np.ndarray:
    if not seqs: return np.zeros((0,4**k),dtype=np.float32)
    vocab = build_kmer_vocab(k)
    vocab_idx = {kmer:i for i,kmer in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i,s in enumerate(seqs): X[i] = seq_to_kmer_vector(s,k,vocab_idx)
    return X

# --------- Dimensionality reduction ---------

def reduce_dimensionality(X:np.ndarray, method="pca", n_components=50):
    if X.shape[0]==0: return X
    n_components = min(n_components, X.shape[1], X.shape[0])
    if method=="pca": return PCA(n_components=n_components).fit_transform(X)
    elif method=="umap" and HAVE_UMAP: return umap.UMAP(n_components=n_components, random_state=42).fit_transform(X)
    else: return PCA(n_components=n_components).fit_transform(X)

# --------- Clustering ---------

def safe_cluster(Z:np.ndarray, method="optics", min_samples=10, xi=0.05, eps=0.5):
    n_samples = Z.shape[0]
    if n_samples==0: return np.array([], dtype=int)
    min_samples = max(2, min(int(min_samples), n_samples))
    model = OPTICS(min_samples=min_samples, xi=xi) if method=="optics" else DBSCAN(min_samples=min_samples, eps=eps)
    return model.fit_predict(Z)

# --------- Shannon ---------

def shannon(counts):
    counts = np.array(counts,dtype=float)
    s = counts.sum()
    if s<=0: return 0.0
    p = counts/s
    p=p[p>0]
    return -np.sum(p*np.log(p))

# --------- BLAST ---------

def run_local_blast(query_fasta:str, dbname:str, out_xml:str, evalue:float=1e-5, threads:int=1):
    if not check_binary("blastn"): raise FileNotFoundError("blastn not found")
    cmd=["blastn","-query",query_fasta,"-db",dbname,"-outfmt","5","-out",out_xml,"-evalue",str(evalue),"-num_threads",str(threads)]
    subprocess.check_call(cmd)
    return out_xml

def run_remote_blast_seq(seq:str, program="blastn", database="nt"):
    try: handle = NCBIWWW.qblast(program,database,seq); return handle.read()
    except: return None

def parse_blast_xml_string(xml_str:str):
    out=[]
    try:
        handle = io.StringIO(xml_str)
        rec = NCBIXML.read(handle)
        for aln in rec.alignments:
            for hsp in aln.hsps:
                out.append({"title":aln.title,"accession":aln.accession,"length":aln.length,"evalue":hsp.expect,"identity":hsp.identities})
    except: pass
    return out

# --------- MSA & Phylogeny ---------

def run_mafft(in_fasta,out_fasta,options="--auto"):
    if not check_binary("mafft"): raise FileNotFoundError("mafft not found")
    proc = subprocess.Popen(["mafft"]+options.split()+[in_fasta], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode!=0: raise RuntimeError(stderr)
    with open(out_fasta,"w") as fh: fh.write(stdout)
    return out_fasta

def run_fasttree(msa_fasta,out_tree):
    if not check_binary("fasttree"): raise FileNotFoundError("fasttree not found")
    proc = subprocess.Popen(["fasttree","-nt",msa_fasta], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if proc.returncode!=0: raise RuntimeError(stderr)
    with open(out_tree,"w") as fh: fh.write(stdout)
    return out_tree

def hierarchical_tree_from_seqs(seqs,ids,metric="cosine"):
    X = batch_kmers(seqs, k=4)
    D = pairwise_distances(X, metric=metric)
    return linkage(D, method="average")

# --------- Streamlit UI ---------

st.set_page_config("eDNA AI Pipeline", layout="wide")
st.title("eDNA AI Smart Pipeline")

# Sidebar
st.sidebar.header("Settings")
k_mer = st.sidebar.slider("k-mer size",4,8,6)
embed_method = st.sidebar.selectbox("Dimensionality reduction", ["pca","umap"] if HAVE_UMAP else ["pca"])
pca_components = st.sidebar.number_input("PCA components",2,200,50)
cluster_method = st.sidebar.selectbox("Clustering method",["optics","dbscan"])
min_samples = st.sidebar.number_input("min_samples",2,20,10)
optics_xi = st.sidebar.slider("OPTICS xi",0.01,0.2,0.05)
dbscan_eps = st.sidebar.number_input("DBSCAN eps",0.01,5.0,0.5)
filter_min_len = st.sidebar.number_input("Minimum seq length",50,10000,80)
use_local_blast = st.sidebar.checkbox("Use local BLAST", False)
local_blast_db = st.sidebar.text_input("Local BLAST DB") if use_local_blast else None
run_mafft_opt = st.sidebar.checkbox("Enable MAFFT",True)
run_fasttree_opt = st.sidebar.checkbox("Enable FastTree",False)
openai_key = st.sidebar.text_input("OpenAI API Key (for AI assistant)", type="password")

# Upload
st.header("Upload Sequences")
upload_mode = st.radio("Single or ZIP?",["Single FASTA","ZIP of FASTAs"])
uploaded=None
if upload_mode=="Single FASTA": uploaded=st.file_uploader("Upload FASTA",["fasta","fa","fna"])
else: uploaded=st.file_uploader("Upload ZIP",["zip"])
if not uploaded: st.stop()

# Load sequences
seq_collections={}
if upload_mode=="Single FASTA":
    uploaded.seek(0)
    seq_collections[uploaded.name or "input.fasta"] = read_fasta_handle(io.TextIOWrapper(uploaded,encoding="utf-8"))
else:
    seq_collections = extract_from_zip(uploaded)

# Filter
filtered_collections={fname:[(rid,seq) for rid,seq in recs if len(seq)>=filter_min_len] for fname,recs in seq_collections.items()}

seq_ids=[]
seqs=[]
sample_map=[]
for sname,recs in filtered_collections.items():
    for rid,seq in recs:
        seq_ids.append(rid)
        seqs.append(seq)
        sample_map.append(sname)

if not seqs: st.error("No sequences remain after filter"); st.stop()
st.info(f"Processing {len(seqs)} sequences from {len(filtered_collections)} samples.")

# K-mer embedding
with st.spinner("Computing k-mers..."): X=batch_kmers(seqs,k=k_mer)
Z = reduce_dimensionality(X, embed_method, min(pca_components,X.shape[1],X.shape[0]))

# Clustering
labels = safe_cluster(Z, cluster_method, min_samples, optics_xi, dbscan_eps)
st.write("Cluster labels (−1 = noise):", np.unique(labels))

# Cluster table
df_clusters=pd.DataFrame({"sample":sample_map,"seq_id":seq_ids,"cluster":labels})
ab_table=df_clusters.groupby(["sample","cluster"])["seq_id"].count().reset_index(name="count")
ab_pivot=ab_table.pivot(index="sample",columns="cluster",values="count").fillna(0).astype(int).reset_index()

st.subheader("Cluster Table")
st.dataframe(df_clusters.head(200))
st.subheader("Abundance Table")
st.dataframe(ab_pivot)

# Diversity
shannon_per_sample={sample:shannon(group["count"].values) for sample,group in ab_table.groupby("sample")}
st.subheader("Shannon Diversity")
st.dataframe(pd.DataFrame.from_dict(shannon_per_sample, orient="index", columns=["Shannon"]).reset_index().rename(columns={"index":"sample"}))

# -------- AI Assistant --------
st.subheader("AI Assistant (ChatGPT)")

if openai_key and HAVE_OPENAI:
    openai.api_key = openai_key
    prompt = st.text_area("Ask AI about your sequences, clusters, diversity, BLAST results...")
    if st.button("Ask AI"):
        context = f"You have {len(seqs)} sequences, cluster labels: {np.unique(labels)}, Shannon per sample: {shannon_per_sample}."
        try:
            resp=openai.ChatCompletion.create(model="gpt-4",messages=[{"role":"system","content":"You are a bioinformatics assistant."},
                                                                          {"role":"user","content":context+"\n"+prompt}])
            st.text(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"AI query failed: {e}")
else:
    st.info("Provide OpenAI API key to enable AI assistant.")

# BLAST button (remote only for simplicity)
st.subheader("BLAST Annotation")
blast_results={}
if st.button("Run remote BLAST (slow)"):
    for sid,seq in tqdm(zip(seq_ids,seqs),total=len(seqs)):
        xml=run_remote_blast_seq(seq)
        if xml: blast_results[sid]=parse_blast_xml_string(xml)
    st.success("Remote BLAST complete.")

# -------- MSA & Phylogeny --------
st.subheader("Representatives & Tree")
if st.button("MSA + Phylogeny"):
    outdir=tempfile.mkdtemp()
    reps={}
    for c in sorted(set(labels)):
        if c==-1: continue
        idxs=[i for i,lab in enumerate(labels) if lab==c]
        chosen_i=max(idxs,key=lambda ii: len(seqs[ii]))
        reps[c]=(seq_ids[chosen_i],seqs[chosen_i])
    rep_fasta=os.path.join(outdir,"reps.fasta")
    with open(rep_fasta,"w") as fh:
        for cid,(rid,seq) in reps.items(): fh.write(f">{cid}_{rid}\n{seq}\n")
    msa_file=os.path.join(outdir,"reps.msa.fasta")
    if run_mafft_opt and len(reps)>=2 and check_binary("mafft"):
        run_mafft(rep_fasta, msa_file)
        st.success("MAFFT MSA created.")
    tree_file=os.path.join(outdir,"reps.tree")
    if run_fasttree_opt and check_binary("fasttree") and os.path.exists(msa_file):
        run_fasttree(msa_file, tree_file)
        st.success("FastTree created.")
    else:
        # hierarchical fallback
        rep_seqs=[s for _,s in reps.values()]
        if len(rep_seqs)>=2:
            Zlink=hierarchical_tree_from_seqs(rep_seqs,[rid for _,rid in reps.values()])
            fig,ax=plt.subplots(figsize=(8,max(4,len(rep_seqs)*0.3)))
            dendrogram(Zlink, labels=[f"{cid}_{rid}" for cid,(rid,_) in reps.items()], orientation="right")
            st.pyplot(fig)

# -------- Downloads --------
st.subheader("Downloads")
st.download_button("Download clusters CSV", df_clusters.to_csv(index=False).encode("utf-8"), "clusters.csv","text/csv")
st.download_button("Download abundance CSV", ab_pivot.to_csv(index=False).encode("utf-8"), "abundance.csv","text/csv")
