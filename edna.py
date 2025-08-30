# edna_ai_pipeline - Project code bundle
# The file below contains the main project files concatenated. Copy each file into its path when creating the project.

# === requirements.txt ===
# requirements.txt
import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO


# === edna_pipeline/io_utils.py ===
from typing import Iterator, Tuple
from Bio import SeqIO
import os, glob

def read_sequences(path: str) -> Iterator[Tuple[str, str, str]]:
    """Yield (sample_id, seq_id, sequence) from FASTA/FASTQ.
    If `path` is a folder, iterate over supported files inside.
    `sample_id` is derived from the file name.
    """
    files = []
    if os.path.isdir(path):
        for ext in ("*.fasta", "*.fa", "*.fastq", "*.fq"):
            files.extend(glob.glob(os.path.join(path, ext)))
    else:
        files = [path]

    for f in files:
        sample_id = os.path.splitext(os.path.basename(f))[0]
        fmt = "fastq" if f.lower().endswith(("fastq", "fq")) else "fasta"
        for rec in SeqIO.parse(f, fmt):
            yield sample_id, rec.id, str(rec.seq).upper()

# === edna_pipeline/kmer.py ===
import numpy as np
from typing import List

def build_vocab(k: int):
    from itertools import product
    keys = ["".join(p) for p in product("ACGT", repeat=k)]
    return {kmer:i for i,kmer in enumerate(keys)}

def kmerize(seq: str, k: int) -> np.ndarray:
    table = {}
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if set(kmer) <= set("ACGT"):
            table[kmer] = table.get(kmer, 0) + 1
    if not table:
        return np.zeros(4**k, dtype=np.float32)
    vocab = build_vocab(k)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for kmer, cnt in table.items():
        vec[vocab[kmer]] = cnt
    s = vec.sum()
    return vec / s if s > 0 else vec

def batch_kmers(seqs: List[str], k: int) -> np.ndarray:
    vocab_size = 4**k
    X = np.zeros((len(seqs), vocab_size), dtype=np.float32)
    vocab = build_vocab(k)
    for i, s in enumerate(seqs):
        table = {}
        for j in range(len(s)-k+1):
            kmer = s[j:j+k]
            if set(kmer) <= set("ACGT"):
                table[kmer] = table.get(kmer, 0) + 1
        if table:
            for kmer, cnt in table.items():
                X[i, vocab[kmer]] = cnt
            ssum = X[i].sum()
            if ssum > 0:
                X[i] /= ssum
    return X

# === edna_pipeline/models.py ===
import numpy as np

class AutoencoderWrapper:
    def __init__(self, input_dim: int, latent_dim: int = 64, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.backend = None
        self.model = None
        self._init_backend()

    def _init_backend(self):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.backend = "torch"
            class AE(nn.Module):
                def __init__(self, d, z):
                    super().__init__()
                    self.enc = nn.Sequential(
                        nn.Linear(d, 512), nn.ReLU(),
                        nn.Linear(512, 128), nn.ReLU(),
                        nn.Linear(128, z)
                    )
                    self.dec = nn.Sequential(
                        nn.Linear(z, 128), nn.ReLU(),
                        nn.Linear(128, 512), nn.ReLU(),
                        nn.Linear(512, d), nn.Sigmoid()
                    )
                def forward(self, x):
                    z = self.enc(x)
                    xr = self.dec(z)
                    return xr, z
            self.model = AE(self.input_dim, self.latent_dim)
            self.torch = torch
            self.nn = nn
            self.optim = optim
        except Exception:
            from sklearn.neural_network import MLPRegressor
            from sklearn.decomposition import PCA
            self.backend = "sklearn"
            self.model = MLPRegressor(hidden_layer_sizes=(512,128,512), activation="relu",
                                      solver="adam", learning_rate_init=self.lr, max_iter=self.epochs,
                                      batch_size=self.batch_size, verbose=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            torch = self.torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            opt = self.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = self.nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X.astype(np.float32)))
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()
            for _ in range(self.epochs):
                for (xb,) in loader:
                    xb = xb.to(device)
                    opt.zero_grad()
                    xr, z = self.model(xb)
                    loss = loss_fn(xr, xb)
                    loss.backward()
                    opt.step()

            self.model.eval()
            zs = []
            with torch.no_grad():
                for i in range(0, len(X), self.batch_size):
                    xb = torch.from_numpy(X[i:i+self.batch_size].astype(np.float32)).to(device)
                    _, zi = self.model(xb)
                    zs.append(zi.cpu().numpy())
            return np.vstack(zs)
        else:
            # sklearn fallback: train to reconstruct input then use PCA as proxy for latent
            from sklearn.decomposition import PCA
            self.model.fit(X, X)
            pca = PCA(n_components=min(self.latent_dim, X.shape[1]))
            return pca.fit_transform(X)

# === edna_pipeline/clustering.py ===
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS, DBSCAN

def cluster_embeddings(Z: np.ndarray, method: str = "optics", min_samples: int = 10, xi: float = 0.05, eps: float = 0.5):
    if method == "optics":
        optics = OPTICS(min_samples=min_samples, xi=xi, metric="euclidean")
        labels = optics.fit_predict(Z)
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(Z)
    return labels

def summarize_abundance(df_rows: pd.DataFrame) -> pd.DataFrame:
    tab = (df_rows.groupby(["sample_id", "cluster"])['seq_id']
           .count().reset_index(name='count'))
    pivot = tab.pivot(index='sample_id', columns='cluster', values='count').fillna(0).astype(int)
    pivot = pivot.rename_axis(None, axis=1).reset_index()
    return pivot

# === edna_pipeline
