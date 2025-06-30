import boto3, base64, json
import numpy as np

class BedrockEmbedder:
    def __init__(self, model_id="amazon.titan-embed-image-v1", region="us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def embed_image(self, image_path, dim=256):
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf8")
        body = json.dumps({
            "inputImage": data,
            "embeddingConfig": {"outputEmbeddingLength": dim}
        })
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        emb = json.loads(resp["body"].read())["embedding"]
        return np.array(emb, dtype='float32')
