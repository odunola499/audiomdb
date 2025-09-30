
import os
import shutil
from audiomdb.retrievers.base import BaseRetriever

try:
    from google.cloud import storage
except Exception:
    storage = None


class GCPDataRetriever(BaseRetriever):
    def __init__(self, bucket: str, prefix: str = "", cache_dir: str = "cache_dir", prefetch: int = None):
        if storage is None:
            raise RuntimeError("google-cloud-storage is required for GCP retrieval")
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._client = storage.Client()
        self._bucket = self._client.bucket(self.bucket_name)
        super().__init__(cache_dir=cache_dir, prefetch=prefetch)
        self.file_ids = [os.path.basename(p) for p in self.file_ids]

    def download_metadata(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        blob_name = f"{self.prefix}/metadata.json" if self.prefix else "metadata.json"
        local_path = os.path.join(self.cache_dir, "metadata.json")
        blob = self._bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return local_path

    def get_file_into_cache(self, file_id) -> str:
        shard_dir = os.path.join(self.cache_dir, os.path.basename(file_id))
        if os.path.isdir(shard_dir):
            return shard_dir
        prefix = f"{self.prefix}/{os.path.basename(file_id)}/" if self.prefix else f"{os.path.basename(file_id)}/"
        blobs = list(self._client.list_blobs(self.bucket_name, prefix=prefix))
        if not blobs:
            raise FileNotFoundError(f"No objects found at gs://{self.bucket_name}/{prefix}")
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = blob.name[len(self.prefix) + 1:] if self.prefix else blob.name
            local_path = os.path.join(self.cache_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
        return shard_dir

    def delete_file_from_cache(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def prefetch_files(self, n: int):
        for file_id in self.file_ids[:n]:
            self.get_file_into_cache(file_id)
