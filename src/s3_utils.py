import requests
from PIL import Image
from dotenv import load_dotenv
import os
import datetime
import hashlib
import hmac
import json
from io import BytesIO
from urllib.parse import urlencode

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")  # Optional: bucket-specific endpoint

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

class S3Utils:
    def __init__(self):
        self.access_key = AWS_ACCESS_KEY_ID
        self.secret_key = AWS_SECRET_ACCESS_KEY
        self.session_token = AWS_SESSION_TOKEN
        self.region = AWS_REGION
        self.service = 's3'
        # Use bucket-specific endpoint if provided, else default to region endpoint
        self.endpoint = S3_ENDPOINT if S3_ENDPOINT else f'https://s3.{self.region}.amazonaws.com'

    def _aws_headers(self, method, bucket, key, payload=b'', querystring=''):
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        canonical_uri = f'/{bucket}/{key}'
        canonical_querystring = querystring
        # Use the host from the endpoint
        host = self.endpoint.replace('https://','').replace('http://','').split('/')[0]
        canonical_headers = f'host:{host}\n' + f'x-amz-date:{amz_date}\n'
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
            'x-amz-content-sha256': payload_hash,
        }
        if self.session_token:
            headers['x-amz-security-token'] = self.session_token
        return headers

    def get_image_keys_with_folders(self, bucket_name, top_level_prefix=""):
        # Use bucket-specific endpoint if set, else default
        querystring = urlencode({'list-type': 2, 'prefix': top_level_prefix})
        url = f'{self.endpoint}/{bucket_name}?{querystring}'
        headers = self._aws_headers('GET', bucket_name, '', b'', querystring)
        response = requests.get(url, headers=headers)
        # Handle PermanentRedirect by switching to the correct endpoint
        if response.status_code == 301 or (b'<Code>PermanentRedirect</Code>' in response.content):
            # Parse the correct endpoint from the XML
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            endpoint_elem = root.find('Endpoint')
            if endpoint_elem is not None:
                self.endpoint = f'https://{endpoint_elem.text}'
                url = f'{self.endpoint}/?{querystring}'
                headers = self._aws_headers('GET', '', '', b'', querystring)
                response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error listing objects: {response.text}")
            return []
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        image_info_list = []
        for contents in root.findall('.//Contents'):
            key = contents.find('Key').text
            if key.endswith('/'):
                continue
            if key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                parts = key.split('/')
                folder_name = parts[0] if len(parts) > 1 else "root"
                image_info_list.append({
                    's3_path': f"s3://{bucket_name}/{key}",
                    'folder_name': folder_name
                })
        print(f"Found {len(image_info_list)} images in S3 across folders.")
        return image_info_list

    def download_file(self, bucket_name, key, local_path):
        url = f'{self.endpoint}/{bucket_name}/{key}'
        headers = self._aws_headers('GET', bucket_name, key, b'')
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"Error downloading {key}: {response.text}")
            return False

    def upload_file(self, local_path, bucket_name, key):
        url = f'{self.endpoint}/{bucket_name}/{key}'
        with open(local_path, 'rb') as f:
            data = f.read()
        headers = self._aws_headers('PUT', bucket_name, key, data)
        response = requests.put(url, headers=headers, data=data)
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"Error uploading {key}: {response.text}")
            return False

    def load_image_from_s3_path(self, s3_path):
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        url = f'{self.endpoint}/{bucket}/{key}'
        headers = self._aws_headers('GET', bucket, key, b'')
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print(f"Error loading image {s3_path}: {response.text}")
            return None

    def get_image_url_from_s3_path(self, s3_path, expires_in=3600, public=True):
        bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
        return f"https://{bucket_name}.s3.amazonaws.com/{key}"

    def detect_folder_from_query(self, query, known_folder_names):
        query_lower = query.lower()
        query_tokens = [token.strip(".,?!;").lower() for token in query.split()]
        for folder_name_lower in sorted(known_folder_names, key=len, reverse=True):
            if folder_name_lower in query_lower:
                return folder_name_lower
            if folder_name_lower.count(' ') == 0 and folder_name_lower in query_tokens:
                return folder_name_lower
        return None