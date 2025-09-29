from abc import ABC, abstractmethod
import os
import json

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Retrievers fetch data from various sources like hf, s3 ,gcp , local files, etc
    """
    def __init__(self, cache_dir:str = None, prefetch_into_cache:int = None):
        self.cache_dir = cache_dir
        metadata_path = self.download_metadata()
        with open(metadata_path, 'r') as fp:
            self.metadata = json.load(fp)

        self.shard_paths = [shard["path"] for shard in self.metadata["shards"]]


    @abstractmethod
    def download_metadata(self):
        """
        Download or fetch metadata from the data source into a local cache directory.
        If the metadata is already in the cache, it should not be downloaded again.
        Return the local path to the cached metadata file.
        """
        pass


    @abstractmethod
    def get_file_into_cache(self, file_id, cache_dir:str = None) -> str:
        """
        Download or fetch a file or metadata from the data source into a local cache directory.
        If the file is already in the cache, it should not be downloaded again.
        Return the local path to the cached file.
        """
        pass


    @abstractmethod
    def delete_file_from_cache(self, file_id, cache_dir:str = None):
        """
        Delete a file from the local cache directory to free space.
        """
        pass

    @abstractmethod
    def prefech_files(self, n:int):
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
            metadata = self.metadata.pop("shards", None)
        else:
            metadata = self.metadata
            metadata['shard_paths'] = self.shard_paths
        return shard_paths, metadata


    def read_shard(self):
        pass

