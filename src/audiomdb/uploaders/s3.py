import os
from .base import BaseUploader

try:
    import boto3
except Exception:
    print("boto3 library not found. S3 uploads will not work.")
    boto3 = None


class S3Uploader(BaseUploader):
    def __init__(self, bucket: str, prefix: str = "", region_name: str = None, **kwargs):
        """Create an S3 uploader.

        Args:
            bucket: Target S3 bucket name.
            prefix: Optional key prefix under which files will be uploaded.
            region_name: AWS region for the S3 client.
        """
        super().__init__(**kwargs)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region_name = region_name
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 uploads")
        self.s3 = boto3.client("s3", region_name=self.region_name) if region_name else boto3.client("s3")

    def upload_dir(self, local_dir: str):
        """Upload a folder to S3, preserving relative paths under prefix."""
        local_dir = os.path.abspath(local_dir)
        for full_path, rel_path in self.iter_files(local_dir):
            key = f"{self.prefix}/{rel_path}" if self.prefix else rel_path
            key = key.replace("\\", "/")
            self.s3.upload_file(full_path, self.bucket, key)
