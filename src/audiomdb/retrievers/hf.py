from huggingface_hub import snapshot_download, hf_hub_download
from audiomdb.retrievers.base import BaseRetriever


class HFDataRetriever(BaseRetriever):
    """
    Retriever for datasets hosted on Hugging Face Hub.
    """
    def __init__(self,
                 repo_id:str,
                 allow_patterns = 'shard_*',
                 cache_dir = 'cache_dir',
                 prefetch_into_cache = None):
        super().__init__(
            cache_dir=cache_dir,
            prefetch_into_cache=None
        )
        self.repo_id = repo_id
        self.allow_patterns = allow_patterns

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