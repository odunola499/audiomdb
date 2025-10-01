from abc import ABC, abstractmethod
import os
import json
from typing import Optional
import lmdb
import pickle
import time

from .cache_manager import CacheManager


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Retrievers fetch data from various sources like hf, s3 ,gcp , local files, etc
    """
    def __init__(self, cache_dir:str = None, prefetch:Optional[int] = None, max_cache_bytes: Optional[int] = None, background: bool = True, workers: int = 2):
        """
        Initialize the retriever.

        Args:
            cache_dir: Local directory where shards are stored or cached.
            prefetch: If set, prefetch this many shards at startup. -1 fetches all.
            max_cache_bytes: Maximum total size of cached shard directories (in bytes). When exceeded, the least recently used shards are evicted. Set to None to disable eviction.
            background: If True, downloads/evictions happen in background threads via CacheManager. If False, downloads are synchronous.
            workers: Number of background downloader threads when background=True.
        """
        self.cache_dir = cache_dir
        metadata_path = self.download_metadata()
        with open(metadata_path, 'r') as fp:
            self.metadata = json.load(fp)

        self.file_ids = [shard["path"] for shard in self.metadata["shards"]]
        self.dataset_size = self.metadata['dataset_size']

        self.manager = CacheManager(self, cache_dir=self.cache_dir, max_cache_bytes=max_cache_bytes, workers=workers) if background else None

        if prefetch:
            if prefetch == -1:
                prefetch = len(self.file_ids)
            if self.manager:
                for fid in self.file_ids[:prefetch]:
                    self.manager.request(fid)
            else:
                self.prefetch_files(prefetch)


    @abstractmethod
    def download_metadata(self):
        """
        Download or fetch metadata from the data source into a local cache directory.
        If the metadata is already in the cache, it should not be downloaded again.
        Return the local path to the cached metadata file.
        """
        pass


    @abstractmethod
    def get_file_into_cache(self, file_id) -> str:
        """
        Download or fetch a file or metadata from the data source into a local cache directory.
        If the file is already in the cache, it should not be downloaded again.
        Return the local path to the cached file.
        """
        pass

    def load_file(self, file_id):
        """Yield decoded samples from a shard.

        Ensures the shard directory is present (background or synchronous), opens
        the LMDB environment in read-only mode, and yields deserialized samples.
        """
        if self.manager:
            self.manager.request(file_id)
            t0 = 0.0
            while not self.manager.has(file_id):
                time.sleep(0.05)
                t0 += 0.05
                if t0 > 60:
                    break
            self.manager.mark_used(file_id)
            file_path = os.path.join(self.cache_dir, os.path.basename(file_id))
        else:
            file_path = self.get_file_into_cache(file_id)
        env = lmdb.open(file_path, readonly = True, lock = False)
        try:
            with env.begin(write = False) as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        sample = pickle.loads(value)
                        yield sample
                    except Exception as e:
                        print(f"Failed to deserialize sample {key}: {e}")
                        continue
        finally:
            env.close()



    @abstractmethod
    def delete_file_from_cache(self, path):
        """
        Delete a file from the local cache directory to free space.
        """
        pass

    @abstractmethod
    def prefetch_files(self, n:int):
        """
        Prefetch n files into the local cache.
        This method should manage the cache size and ensure that it does not exceed the allocated space.
        """
        pass

    def probe(self, verbose:bool = False):
        """
        Probe the data source to get metadata like number of files, total size, shard files id,etc.
        If verbose is True, returns also shard_level information
        Returns a tuple (list_of_file_paths, metadata)
        """

        if not verbose:
            metadata = {k: v for k, v in self.metadata.items() if k != "shards"}
        else:
            metadata = self.metadata
            metadata['shard_paths'] = self.file_ids
        return metadata



