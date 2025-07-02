import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.s3_utils import S3Utils

S3_BUCKET_NAME = "image2search"
FAISS_INDEX_FILE = "image_faiss_index.bin"
IMAGE_METADATA_FILE = "image_metadata.json"

def main():
    print("Welcome to the AWS Image Search System!")
    print("Choose an option:")
    print("1. Build FAISS Index (Run this first to create/update the database)")
    print("2. Search Images (Run this after building the index)")
    print("3. Exit")

    s3_utils = S3Utils()
    embedder = Embedder()
    vector_store = VectorStore()

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == '1':
            print("\n--- Starting FAISS Index Building Process ---")
            image_s3_info = s3_utils.get_image_keys_with_folders(S3_BUCKET_NAME, top_level_prefix="")
            if not image_s3_info:
                print("No images found in the specified S3 path. Please upload images to S3 first.")
                continue
            embeddings, processed_info = embedder.get_image_embeddings(image_s3_info, s3_utils)
            vector_store.build_and_save_index(embeddings, processed_info, FAISS_INDEX_FILE, IMAGE_METADATA_FILE, s3_utils, S3_BUCKET_NAME)
            print("--- FAISS Index Building Process Completed ---")
        elif choice == '2':
            print("\n--- Starting Image Search Process ---")
            if not os.path.exists(FAISS_INDEX_FILE):
                s3_utils.download_file(S3_BUCKET_NAME, FAISS_INDEX_FILE, FAISS_INDEX_FILE)
            if not os.path.exists(IMAGE_METADATA_FILE):
                s3_utils.download_file(S3_BUCKET_NAME, IMAGE_METADATA_FILE, IMAGE_METADATA_FILE)
            vector_store.load_index_and_metadata(FAISS_INDEX_FILE, IMAGE_METADATA_FILE)
            known_folder_names_lower = vector_store.get_unique_folder_names()
            print(f"Recognized folders: {', '.join(folder.capitalize() for folder in known_folder_names_lower if folder != 'root')}")
            while True:
                text_query = input("\nEnter your search query ('quit' to exit): ").strip()
                if text_query.lower() == 'quit':
                    break
                if not text_query:
                    print("Please enter a valid query.")
                    continue
                detected_folder = s3_utils.detect_folder_from_query(text_query, known_folder_names_lower)
                filter_by_folder = detected_folder
                search_message = f"Searching for: '{text_query}'"
                if filter_by_folder:
                    search_message += f" (filtered to folder '{filter_by_folder.capitalize()}')"
                else:
                    search_message += " (searching all folders)"
                print(search_message + "...")
                query_embedding = embedder.get_text_embedding(text_query)
                search_results = vector_store.search_images(query_embedding, k=5, filter_folder=filter_by_folder)
                if search_results:
                    print("\n--- Top 5 Search Results ---")
                    for i, (distance, image_s3_path, folder_name) in enumerate(search_results):
                        print(f"{i+1}. Distance: {distance:.4f}, Folder: {folder_name}, Image: {image_s3_path}")
                        s3_utils.get_image_url_from_s3_path(image_s3_path)
                        print("-" * 30)
                else:
                    print("No results found for your query with the given filter.")
            print("--- Image Search Process Completed ---")
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
