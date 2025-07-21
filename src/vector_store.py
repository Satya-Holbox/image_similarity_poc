import json
import numpy as np
from dotenv import load_dotenv
import os
import requests
import datetime
import hashlib
import hmac
import time

load_dotenv()

REGION = os.getenv("AWS_REGION")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")  # e.g. https://search-xxx.us-east-1.es.amazonaws.com

# Helper for AWS Signature V4
# Reference: https://docs.aws.amazon.com/general/latest/gr/sigv4-signed-request-examples.html
def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def getSignatureKey(key, dateStamp, regionName, serviceName):
    kDate = sign(('AWS4' + key).encode('utf-8'), dateStamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'aws4_request')
    return kSigning

class VectorStore:
    def __init__(self):
        self.index_name = VECTOR_INDEX_NAME
        self.metadata = {}
        self.region = REGION
        self.access_key = AWS_ACCESS_KEY_ID
        self.secret_key = AWS_SECRET_ACCESS_KEY
        self.session_token = AWS_SESSION_TOKEN
        self.service = 'es'
        self.endpoint = OPENSEARCH_ENDPOINT

    def _aws_headers(self, method, canonical_uri, payload):
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        canonical_querystring = ''
        canonical_headers = f'host:{self.endpoint.replace('https://','').replace('http://','').split('/')[0]}\n' + f'x-amz-date:{amz_date}\n'
        signed_headers = 'host;x-amz-date'
        if self.session_token:
            canonical_headers += f'x-amz-security-token:{self.session_token}\n'
            signed_headers += ';x-amz-security-token'
        payload_hash = hashlib.sha256(payload).hexdigest()
        canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f'{date_stamp}/{self.region}/{self.service}/aws4_request'
        string_to_sign = f'{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}'
        signing_key = getSignatureKey(self.secret_key, date_stamp, self.region, self.service)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        authorization_header = (
            f'{algorithm} Credential={self.access_key}/{credential_scope}, '
            f'SignedHeaders={signed_headers}, Signature={signature}'
        )
        headers = {
            'x-amz-date': amz_date,
            'Authorization': authorization_header,
            'Content-Type': 'application/json',
            'x-amz-content-sha256': payload_hash,
        }
        if self.session_token:
            headers['x-amz-security-token'] = self.session_token
        return headers

    def build_and_save_index(self, embeddings, image_info, metadata_file, s3_utils, s3_bucket):
        if embeddings.shape[0] == 0:
            print("No embeddings to build index. Exiting.")
            return
        print(f"Uploading {embeddings.shape[0]} vectors to OpenSearch index: {self.index_name}")
        for i, (embedding, info) in enumerate(zip(embeddings, image_info)):
            doc = {
                "vector": embedding.tolist(),
                "s3_path": info['s3_path'],
                "folder_name": info['folder_name'],
                "id": str(i)
            }
            try:
                url = f"{self.endpoint}/{self.index_name}/_doc/{i}"
                payload = json.dumps(doc).encode('utf-8')
                headers = self._aws_headers('PUT', f"/{self.index_name}/_doc/{i}", payload)
                response = requests.put(url, headers=headers, data=payload)
                if response.status_code not in [200, 201]:
                    print(f"Error indexing document {i}: {response.text}")
                self.metadata[str(i)] = {
                    "s3_path": info['s3_path'],
                    "folder_name": info['folder_name']
                }
            except Exception as e:
                print(f"Error indexing document {i}: {e}")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        print(f"Image metadata saved locally to {metadata_file}")
        s3_utils.upload_file(metadata_file, s3_bucket, metadata_file)
        print("Metadata uploaded to S3 successfully.")

    def load_index_and_metadata(self, metadata_file):
        print(f"Loading image metadata from {metadata_file}...")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        print("Image metadata loaded.")

    def search_images(self, query_embedding, k=5, filter_folder=None):
        # OpenSearch k-NN search via REST API
        query = {
            "size": k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_embedding.flatten().tolist(),
                        "k": k
                    }
                }
            }
        }
        try:
            url = f"{self.endpoint}/{self.index_name}/_search"
            payload = json.dumps(query).encode('utf-8')
            headers = self._aws_headers('POST', f"/{self.index_name}/_search", payload)
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code != 200:
                print(f"Error during vector search: {response.text}")
                return []
            resp_json = response.json()
            results = []
            for hit in resp_json['hits']['hits']:
                source = hit['_source']
                if filter_folder is None or source['folder_name'].lower() == filter_folder.lower():
                    results.append((hit['_score'], source['s3_path'], source['folder_name']))
            return results
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []

    def get_unique_folder_names(self):
        unique_folders = set()
        for item_id, item_data in self.metadata.items():
            if 'folder_name' in item_data:
                unique_folders.add(item_data['folder_name'].lower())
        return list(unique_folders)