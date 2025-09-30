import os
from abc import ABC, abstractmethod


class BaseUploader(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def upload_dir(self, local_dir: str):
        pass

    @staticmethod
    def iter_files(local_dir: str):
        for root, _, files in os.walk(local_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, start=local_dir)
                yield full_path, rel_path
