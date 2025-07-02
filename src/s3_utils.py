import boto3
from PIL import Image

# --- Initialize AWS S3 Client ---
s3 = boto3.client('s3')

class S3Utils:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def get_image_keys_with_folders(self, bucket_name, top_level_prefix=""):
        print(f"Listing images in s3://{bucket_name}/{top_level_prefix}...")
        image_info_list = []
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=top_level_prefix)
        for page in pages:
            if "Contents" in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue
                    if key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                        relative_key = key.replace(top_level_prefix, "", 1)
                        parts = relative_key.split('/')
                        folder_name = parts[0] if len(parts) > 1 else "root"
                        image_info_list.append({
                            's3_path': f"s3://{bucket_name}/{key}",
                            'folder_name': folder_name
                        })
        print(f"Found {len(image_info_list)} images in S3 across folders.")
        return image_info_list

    def download_file(self, bucket_name, key, local_path):
        try:
            self.s3.download_file(bucket_name, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            return False

    def upload_file(self, local_path, bucket_name, key):
        try:
            self.s3.upload_file(local_path, bucket_name, key)
            return True
        except Exception as e:
            print(f"Error uploading {key}: {e}")
            return False

    def load_image_from_s3_path(self, s3_path):
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return Image.open(obj['Body']).convert("RGB")

    def get_image_url_from_s3_path(self, s3_path, expires_in=3600, public=True):
        """
        Returns a URL for the image in S3.
        If public=True, returns the public S3 URL.
        If public=False, returns a presigned URL valid for `expires_in` seconds.
        """
        bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
        if public:
            return f"https://{bucket_name}.s3.amazonaws.com/{key}"
        else:
            # Generate a presigned URL (requires correct IAM permissions)
            try:
                url = self.s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': key},
                    ExpiresIn=expires_in
                )
                return url
            except Exception as e:
                print(f"Error generating presigned URL: {e}")
                return None

    def detect_folder_from_query(self, query, known_folder_names):
        query_lower = query.lower()
        query_tokens = [token.strip(".,?!;").lower() for token in query.split()]
        for folder_name_lower in sorted(known_folder_names, key=len, reverse=True):
            if folder_name_lower in query_lower:
                return folder_name_lower
            if folder_name_lower.count(' ') == 0 and folder_name_lower in query_tokens:
                return folder_name_lower
        return None
