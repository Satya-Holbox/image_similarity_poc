import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import tempfile

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.s3_utils import S3Utils

import os

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
IMAGE_METADATA_FILE = "image_metadata.json"

# Initialize once at startup
s3_utils = S3Utils()
embedder = Embedder()
vector_store = VectorStore()

# Load index and metadata at startup
if not os.path.exists(IMAGE_METADATA_FILE):
    s3_utils.download_file(S3_BUCKET_NAME, IMAGE_METADATA_FILE, IMAGE_METADATA_FILE)
vector_store.load_index_and_metadata(IMAGE_METADATA_FILE)
known_folder_names_lower = vector_store.get_unique_folder_names()
app = FastAPI(
    title="Image Similarity Search API",
    description="Search for similar images using Amazon Bedrock (Twelve Labs) and OpenSearch",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    folder: Optional[str] = None

class SearchResult(BaseModel):
    distance: float
    folder: str
    s3_path: str
    image_url: str

@app.post("/search", response_model=List[SearchResult])
def search_images(request: SearchRequest):
    # Detect folder if not provided
    filter_by_folder = request.folder
    if not filter_by_folder:
        filter_by_folder = s3_utils.detect_folder_from_query(request.query, known_folder_names_lower)
    query_embedding = embedder.get_text_embedding(request.query)
    search_results = vector_store.search_images(query_embedding, k=request.k, filter_folder=filter_by_folder)
    print("Query:", request.query)
    print("Detected folder:", filter_by_folder)
    print("Search results:", search_results)
    results = []
    for distance, image_s3_path, folder_name in search_results:
        image_url = s3_utils.get_image_url_from_s3_path(image_s3_path)
        results.append(SearchResult(
            distance=float(distance),
            folder=folder_name,
            s3_path=image_s3_path,
            image_url=image_url
        ))
    return results

@app.post("/search_by_image", response_model=List[SearchResult])
async def search_by_image(
    file: UploadFile = File(...),
    k: int = 5
):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        query_embedding = embedder.get_image_embedding_from_file(tmp_path)
        search_results = vector_store.search_images(query_embedding, k=k)
        results = []
        for distance, image_s3_path, folder_name in search_results:
            image_url = s3_utils.get_image_url_from_s3_path(image_s3_path)
            results.append(SearchResult(
                distance=float(distance),
                folder=folder_name,
                s3_path=image_s3_path,
                image_url=image_url
            ))
        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)

@app.get("/")
def root():
    return {"message": "Welcome to the Image Similarity Search API!"}
