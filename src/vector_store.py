import faiss, numpy as np

class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, emb: np.ndarray, meta):
        self.index.add(emb.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, query_emb: np.ndarray, top_k=5):
        D, I = self.index.search(query_emb.reshape(1, -1), top_k)
        return [(self.metadata[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
