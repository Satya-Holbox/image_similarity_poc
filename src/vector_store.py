import faiss
import json
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = None

    def build_and_save_index(self, embeddings, image_info, index_file, metadata_file, s3_utils, s3_bucket):
        if embeddings.shape[0] == 0:
            print("No embeddings to build index. Exiting.")
            return
        dimension = embeddings.shape[1]
        print(f"Building FAISS index with dimension: {dimension}")
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"FAISS index built. Total vectors: {self.index.ntotal}")
        faiss.write_index(self.index, index_file)
        print(f"FAISS index saved locally to {index_file}")
        image_metadata = {str(i): {'s3_path': info['s3_path'], 'folder_name': info['folder_name']}
                          for i, info in enumerate(image_info)}
        with open(metadata_file, 'w') as f:
            json.dump(image_metadata, f, indent=4)
        print(f"Image metadata saved locally to {metadata_file}")
        print(f"Uploading {index_file} and {metadata_file} to S3 bucket: {s3_bucket}...")
        s3_utils.upload_file(index_file, s3_bucket, index_file)
        s3_utils.upload_file(metadata_file, s3_bucket, metadata_file)
        print("Files uploaded to S3 successfully.")

    def load_index_and_metadata(self, index_file, metadata_file):
        print(f"Loading FAISS index from {index_file}...")
        self.index = faiss.read_index(index_file)
        print(f"FAISS index loaded. Total vectors: {self.index.ntotal}")
        print(f"Loading image metadata from {metadata_file}...")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        print("Image metadata loaded.")

    def search_images(self, query_embedding, k=5, filter_folder=None):
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, self.index.ntotal)
        results = []
        sorted_results = sorted([(distances[0][i], idx) for i, idx in enumerate(indices[0]) if idx != -1])
        count = 0
        for distance, idx in sorted_results:
            if count >= k:
                break
            metadata_entry = self.metadata[str(idx)]
            s3_path = metadata_entry['s3_path']
            folder_name = metadata_entry['folder_name']
            if filter_folder is None or folder_name.lower() == filter_folder.lower():
                results.append((distance, s3_path, folder_name))
                count += 1
        return results

    def get_unique_folder_names(self):
        unique_folders = set()
        for item_id, item_data in self.metadata.items():
            if 'folder_name' in item_data:
                unique_folders.add(item_data['folder_name'].lower())
        return list(unique_folders)
