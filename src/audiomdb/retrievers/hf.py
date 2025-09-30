from huggingface_hub import snapshot_download, hf_hub_download
from audiomdb.retrievers.base import BaseRetriever
import os
import shutil


class HFDataRetriever(BaseRetriever):
    """Retrieve shards from Hugging Face Hub on demand and cache locally."""
    def __init__(self,
                 repo_id:str,
                 allow_patterns = 'shard_*',
                 cache_dir = 'cache_dir',
                 prefetch = -1):
        self.repo_id = repo_id
        self.allow_patterns = allow_patterns
        super().__init__(
            cache_dir=cache_dir,
            prefetch=prefetch
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.file_ids = [os.path.basename(p) for p in self.file_ids]


    def download_metadata(self):
        """
        Download or fetch metadata from the data source into a local cache directory.
        If the metadata is already in the cache, it should not be downloaded again.
        Return the local path to the cached metadata file.
        """
        file_path = hf_hub_download(
            repo_id = self.repo_id,
            filename = 'metadata.json',
            cache_dir = self.cache_dir
        )
        return file_path

    def get_file_into_cache(self, file_id) -> str:
        name = os.path.basename(file_id)
        shard_dir = os.path.join(self.cache_dir, name)
        if os.path.isdir(shard_dir):
            return shard_dir

        snapshot_download(
            repo_id=self.repo_id,
            allow_patterns=f"{name}/*",
            local_dir=self.cache_dir,
            cache_dir=self.cache_dir,
            local_dir_use_symlinks=False
        )
        return shard_dir

    def delete_file_from_cache(self, path):
        shutil.rmtree(path)

    def prefetch_files(self, n:int):
        patterns = [f"{os.path.basename(p)}/*" for p in self.file_ids[:n]]
        if not patterns:
            return
        snapshot_download(
            repo_id=self.repo_id,
            allow_patterns=patterns,
            local_dir=self.cache_dir,
            cache_dir=self.cache_dir,
            local_dir_use_symlinks=False,
            max_workers=max(os.cpu_count() - 2, 1)
        )

