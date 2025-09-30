import os
import shutil
from audiomdb.retrievers.base import BaseRetriever

try:
    import boto3
except Exception:
    boto3 = None


class S3DataRetriever(BaseRetriever):
    def __init__(self, bucket: str, prefix: str = "", cache_dir: str = "cache_dir", prefetch: int = None):
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 retrieval")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3")
        super().__init__(cache_dir=cache_dir, prefetch=prefetch)
        self.file_ids = [os.path.basename(p) for p in self.file_ids]

    def download_metadata(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        key = f"{self.prefix}/metadata.json" if self.prefix else "metadata.json"
        local_path = os.path.join(self.cache_dir, "metadata.json")
        self._s3.download_file(self.bucket, key, local_path)
        return local_path

    def _list_objects(self, prefix):
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def get_file_into_cache(self, file_id) -> str:
        shard_dir = os.path.join(self.cache_dir, os.path.basename(file_id))
        if os.path.isdir(shard_dir):
            return shard_dir
        key_prefix = f"{self.prefix}/{os.path.basename(file_id)}/" if self.prefix else f"{os.path.basename(file_id)}/"
        keys = list(self._list_objects(key_prefix))
        if not keys:
            raise FileNotFoundError(f"No objects found at s3://{self.bucket}/{key_prefix}")
        for key in keys:
            rel = key[len(self.prefix) + 1:] if self.prefix else key
            local_path = os.path.join(self.cache_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if key.endswith("/"):
                continue
            self._s3.download_file(self.bucket, key, local_path)
        return shard_dir

    def delete_file_from_cache(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def prefetch_files(self, n: int):
        for file_id in self.file_ids[:n]:
            self.get_file_into_cache(file_id)
