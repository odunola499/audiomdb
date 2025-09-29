from torch.utils.data import IterableDataset #might not need this
from abc import ABC, abstractmethod
import os
import asyncio
import json
from typing import Literal
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import threading
import lmdb
import pickle

class StreamingDataset(ABC):
    def __init__(self, local_dir:str, num_threads = 4, queue_size = 2000):
        super().__init__()
        self.local_dir = local_dir
        manifest_file = os.path.join(self.local_dir, 'metadata.json')
        with open(manifest_file, 'r') as fp:
            metadata = json.load(fp)
        self.metadata = metadata

        self.num_threads = num_threads
        self._finished = threading.Event() #We'd use this to get more files?
        self._active_threads = threading.Semaphore(0)

        self.output_queue = queue.Queue(maxsize = queue_size)

    def prefetch_files(self, n):
        """
        Retriever refetches n files into local cache, depends on cache size allocation if set
        :param n:
        :return:
        """

    def process_file(self, shard_path:str):
        # If shard path not in local dir then download it
        env = lmdb.open(shard_path, readonly=True, lock = False)
        with env.begin(write = False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if self._finished.is_set():
                    #add recursion to get new file and continue
                    break

                sample = pickle.loads(value)
                self.output_queue.put(sample)

    def probe(self):
        #probe has to help tell what features are bytes and need to be unpicked
        return self.metadata

    def begin(self, file_paths):
        #file_paths can just be something from the metadata file
        with ThreadPoolExecutor(max_workers = self.num_threads,
                                thread_name_prefix="FileReader") as executor:
            futures = []
            for filepath in file_paths:
                self._active_threads.acquire(blocking = False)
                future = executor.submit(self.process_file, filepath)
                futures.append(future)

            def monitor_completion():
                for future in futures:
                    future.result()
                self._finished.set()

            threading.Thread(target = monitor_completion, daemon = True).start()

    def __iter__(self):
        return self

    def __next(self):
        while True:
            try:
                return self.output_queue.get(timeout = 0.1)
            except queue.Empty:
                if self._finished.is_set() and self.output_queue.empty():
                    raise StopIteration
