import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class Embedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_image_embeddings(self, image_info_list, s3_utils):
        embeddings = []
        processed_image_info = []
        for img_info in image_info_list:
            s3_path = img_info['s3_path']
            try:
                image = s3_utils.load_image_from_s3_path(s3_path)
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                embeddings.append(image_features.cpu().numpy())
                processed_image_info.append(img_info)
            except Exception as e:
                print(f"Error processing image {s3_path}: {e}. Skipping.")
                continue
        if not embeddings:
            return np.array([]), []
        return np.vstack(embeddings), processed_image_info

    def get_text_embedding(self, text_query):
        inputs = self.processor(text=text_query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()
