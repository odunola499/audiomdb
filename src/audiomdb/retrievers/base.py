from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    Retrievers fetch data from various sources like hf, s3 ,gcp , local files, etc
    """

    @abstractmethod
    def get_file_into_cache(self, file_id, cache_dir:str = None):
        """
        Download or fetch a file from the data source into a local cache directory.
        If the file is already in the cache, it should not be downloaded again.
        Return the local path to the cached file.
        """
        pass

    @abstractmethod
    def probe(self):
        """
        Probe the data source to get metadata like number of files, total size, shard files id,etc.
        """
        pass

    @abstractmethod
    def _get_candidate_files(self, k:int):
        pass

