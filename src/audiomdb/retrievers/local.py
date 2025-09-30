import os
import shutil
from audiomdb.retrievers.base import BaseRetriever


class LocalDataRetriever(BaseRetriever):
    def __init__(self, data_dir: str, cache_dir: str = None, prefetch: int = None, background: bool = False, max_cache_bytes: int = None):
        self.data_dir = os.path.abspath(data_dir)
        if cache_dir is None:
            cache_dir = self.data_dir
        super().__init__(cache_dir=cache_dir, prefetch=prefetch, max_cache_bytes=max_cache_bytes, background=background)
        self.file_ids = [os.path.basename(p) for p in self.file_ids]

    def download_metadata(self):
        path = os.path.join(self.data_dir, 'metadata.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")
        return path

    def get_file_into_cache(self, file_id) -> str:
        name = os.path.basename(file_id)
        shard_dir = os.path.join(self.cache_dir, name)
        if os.path.isdir(shard_dir):
            return shard_dir
        src_dir = os.path.join(self.data_dir, name)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Shard directory not found: {src_dir}")
        if os.path.abspath(self.cache_dir) != os.path.abspath(self.data_dir):
            shutil.copytree(src_dir, shard_dir)
        return shard_dir

    def delete_file_from_cache(self, path):
        apath = os.path.abspath(path)
        if os.path.isdir(apath) and os.path.commonpath([apath, os.path.abspath(self.cache_dir)]) == os.path.abspath(self.cache_dir):
            if os.path.abspath(self.cache_dir) != os.path.abspath(self.data_dir):
                shutil.rmtree(apath, ignore_errors=True)

    def prefetch_files(self, n: int):
        for shard in self.file_ids[:n]:
            self.get_file_into_cache(shard)
