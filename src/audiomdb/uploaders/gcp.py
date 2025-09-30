import os
from .base import BaseUploader

try:
    from google.cloud import storage
except Exception:
    print("Google Cloud Storage library not found. GCP uploads will not work.")
    storage = None


class GCPUploader(BaseUploader):
    def __init__(self, bucket: str, prefix: str = "", **kwargs):
        """Create a GCS uploader.

        Args:
            bucket: Target GCS bucket name.
            prefix: Optional object prefix under which files will be uploaded.
        """
        super().__init__(**kwargs)
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        if storage is None:
            raise RuntimeError("google-cloud-storage is required for GCP uploads")
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_dir(self, local_dir: str):
        """Upload a folder to GCS, preserving relative paths under prefix."""
        local_dir = os.path.abspath(local_dir)
        for full_path, rel_path in self.iter_files(local_dir):
            blob_name = f"{self.prefix}/{rel_path}" if self.prefix else rel_path
            blob_name = blob_name.replace("\\", "/")
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(full_path)
