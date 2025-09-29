from abc import ABC, abstractmethod
import os
import json

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Retrievers fetch data from various sources like hf, s3 ,gcp , local files, etc
    """
    def __init__(self, cache_dir:str = None):
        self.cache_dir = cache_dir
        metadata_path = os.path.join(cache_dir, 'metadata.json') if cache_dir else None
        with open(metadata_path, 'r') as fp:
            metadata = json.load(fp)
        metadata = json.load(fp)
        self.metadata = metadata


    @abstractmethod
    def get_file_into_cache(self, file_id, cache_dir:str = None):
        """
        Download or fetch a file from the data source into a local cache directory.
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
    def probe(self, verbose:bool = False):
        """
        Probe the data source to get metadata like number of files, total size, shard files id,etc.
        If verbose is True, returns also shard_level information
        Returns a tuple (list_of_file_paths, metadata)
        """
        shard_paths = [shard["path"] for shard in self.metadata["shards"]]
        if not verbose:
            metadata = self.metadata.pop("shards", None)
        else:
            metadata = self.metadata
        return shard_paths, metadata

    @abstractmethod
    def _get_candidate_files(self, k:int):
        pass

