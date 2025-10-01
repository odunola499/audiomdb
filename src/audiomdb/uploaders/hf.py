import os
from .base import BaseUploader

try:
    from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
except Exception:
    raise ImportError("huggingface-hub library not found. HF uploads will not work.")
    HfApi = None


class HFUploader(BaseUploader):
    def __init__(self, repo_id: str, private: bool = False, repo_type: str = "dataset", exist_ok: bool = True, **kwargs):
        """Create a Hugging Face Hub uploader.

        Args:
            repo_id: Target repository ID (e.g., "user/dataset").
            private: Whether to create/use a private repository.
            repo_type: Repository type, typically "dataset".
            exist_ok: Do not error if the repo already exists.
        """
        super().__init__(**kwargs)
        self.repo_id = repo_id
        self.private = private
        self.repo_type = repo_type
        self.exist_ok = exist_ok
        if HfApi is None:
            raise RuntimeError("huggingface-hub is required for HF uploads")
        self.api = HfApi()
        create_repo(repo_id=self.repo_id, private=self.private, repo_type=self.repo_type, exist_ok=self.exist_ok)

    def upload_dir(self, local_dir: str):
        """Upload a folder to the Hub as a dataset repository.

        Preserves folder structure and commits a single snapshot.
        """
        local_dir = os.path.abspath(local_dir)
        upload_folder(
            repo_id=self.repo_id,
            folder_path=local_dir,
            repo_type=self.repo_type,
            commit_message="Upload dataset folder",
            allow_patterns=None,
            ignore_patterns=None
        )
