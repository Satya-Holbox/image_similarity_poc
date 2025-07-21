import base64
import numpy as np
from dotenv import load_dotenv
import os
import time
import requests
import json
import datetime
import hashlib
import hmac

load_dotenv()

REGION = os.getenv("AWS_REGION")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

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

class Embedder:
    def __init__(self):
        self.model_id = MODEL_ID
        self.region = REGION
        self.access_key = AWS_ACCESS_KEY_ID
        self.secret_key = AWS_SECRET_ACCESS_KEY
        self.session_token = AWS_SESSION_TOKEN
        self.service = 'bedrock'
        self.api_base = f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def _aws_headers(self, method, canonical_uri, payload):
        # AWS Signature V4 signing
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        canonical_querystring = ''
        canonical_headers = f'host:bedrock-runtime.{self.region}.amazonaws.com\n' + f'x-amz-date:{amz_date}\n'
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
        }
        if self.session_token:
            headers['x-amz-security-token'] = self.session_token
        return headers

    def _start_async_invoke(self, body, content_type):
        # REST API call to start async job
        url = f"{self.api_base}/model-invocations/{self.model_id}/async"
        headers = self._aws_headers('POST', f"/model-invocations/{self.model_id}/async", body)
        headers['Content-Type'] = content_type
        response = requests.post(url, headers=headers, data=body)
        if response.status_code != 200:
            raise Exception(f"Failed to start async job: {response.text}")
        resp_json = response.json()
        return resp_json['jobIdentifier']

    def _get_async_result(self, job_identifier, poll_interval=2, timeout=120):
        # REST API call to poll for result
        url = f"{self.api_base}/model-invocation-jobs/{job_identifier}"
        headers = self._aws_headers('GET', f"/model-invocation-jobs/{job_identifier}", b"")
        start_time = time.time()
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to get async job result: {response.text}")
            result = response.json()
            status = result['status']
            if status == 'COMPLETED':
                output_location = result['outputLocation']
                r = requests.get(output_location)
                output = r.json()
                return output
            elif status == 'FAILED':
                raise Exception(f"Async inference failed: {result.get('failureReason', 'Unknown reason')}")
            elif time.time() - start_time > timeout:
                raise TimeoutError("Async inference timed out.")
            time.sleep(poll_interval)

    def get_image_embeddings(self, image_info_list, s3_utils):
        embeddings = []
        processed_image_info = []
        for img_info in image_info_list:
            s3_path = img_info['s3_path']
            try:
                image = s3_utils.load_image_from_s3_path(s3_path)
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
                # Start async job
                job_identifier = self._start_async_invoke(image_bytes, "image/jpeg")
                # Poll for result
                output = self._get_async_result(job_identifier)
                # Parse embedding (assume base64-encoded float32 array)
                embedding_b64 = output['embedding']
                embedding = np.frombuffer(base64.b64decode(embedding_b64), dtype=np.float32)
                embeddings.append(embedding)
                processed_image_info.append(img_info)
            except Exception as e:
                print(f"Error processing image {s3_path}: {e}. Skipping.")
                continue
        if not embeddings:
            return np.array([]), []
        return np.vstack(embeddings), processed_image_info

    def get_image_embedding_from_file(self, file_path):
        from PIL import Image
        from io import BytesIO
        image = Image.open(file_path).convert("RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        job_identifier = self._start_async_invoke(image_bytes, "image/jpeg")
        output = self._get_async_result(job_identifier)
        embedding_b64 = output['embedding']
        embedding = np.frombuffer(base64.b64decode(embedding_b64), dtype=np.float32)
        return embedding.reshape(1, -1)

    def get_text_embedding(self, text_query):
        try:
            job_identifier = self._start_async_invoke(text_query.encode("utf-8"), "text/plain")
            output = self._get_async_result(job_identifier)
            embedding_b64 = output['embedding']
            embedding = np.frombuffer(base64.b64decode(embedding_b64), dtype=np.float32)
            return embedding.reshape(1, -1)
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros((1, 512), dtype=np.float32)  # fallback