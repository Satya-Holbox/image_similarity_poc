import os, pickle
from embedder import BedrockEmbedder
from vector_store import VectorStore

CATALOG_DIR = "../catalog"
EMB_FILE = "../embeddings/catalog_embeddings.pkl"
QUERY_IMG = "../catalog/ralph_jacket.jpg"

def build_catalog(embedder, dim=256):
    vs = VectorStore(dim)
    for fname in os.listdir(CATALOG_DIR):
        path = os.path.join(CATALOG_DIR, fname)
        emb = embedder.embed_image(path, dim=dim)
        vs.add(emb, {"file": fname})
    with open(EMB_FILE, "wb") as f:
        pickle.dump((vs, dim), f)
    return vs

def load_catalog():
    with open(EMB_FILE, "rb") as f:
        return pickle.load(f)[0]

def run_query(embedder, vs, query_path):
    q_emb = embedder.embed_image(query_path, dim=vs.dim)
    results = vs.search(q_emb, top_k=5)
    print("Top similar images:")
    for meta, score in results:
        print(meta["file"], "→", f"{score:.4f}")

if __name__ == "__main__":
    embedder = BedrockEmbedder()
    vs = build_catalog(embedder, dim=256)
    vs = load_catalog()   
    run_query(embedder, vs, QUERY_IMG)
