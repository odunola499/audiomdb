import os
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from audiomdb.retrievers import BaseRetriever
from typing import Optional

class StreamingDataset:
    def __init__(self, retriever:BaseRetriever,local_dir:Optional[str] = None,  num_threads = 4, queue_size = 2000):
        super().__init__()
        local_dir = retriever.cache_dir if local_dir is None else local_dir
        self.local_dir = local_dir

        self.metadata = retriever.metadata
        self.retriever = retriever

        self.num_threads = num_threads
        self._finished = threading.Event()
        self._file_semaphore = threading.Semaphore(num_threads)

        self.output_queue = queue.Queue(maxsize = queue_size)

        self.executor = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix="FileReader"
        )
        if getattr(self.retriever, 'manager', None):
            for fid in self.retriever.file_ids[: max(2 * num_threads, 4)]:
                self.retriever.manager.request(fid)
        self.begin(self.retriever.file_ids)

    def __len__(self):
        return self.retriever.dataset_size

    def process_file(self, file_id:str):
        with self._file_semaphore:
            try:
                iterator = self.retriever.load_file(file_id)
                for sample in iterator:
                    if self._finished.is_set():
                        break

                    while not self._finished.is_set():
                        try:
                            self.output_queue.put(sample, timeout=1)
                            break
                        except queue.Full:
                            continue
            except Exception as e:
                self.output_queue.put(e)
                self._finished.set()
            finally:
                file_path = os.path.join(self.retriever.cache_dir, os.path.basename(file_id))
                if os.path.isdir(file_path):
                    self.retriever.delete_file_from_cache(file_path)


    def begin(self, file_ids):
        futures = []
        for file_id in file_ids:
            future = self.executor.submit(self.process_file, file_id)
            futures.append(future)

        def monitor_completion():
            for future in futures:
                future.result()
            self._finished.set()

        threading.Thread(target = monitor_completion, daemon = True).start()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.output_queue.get(timeout = 0.1)
                if isinstance(item, Exception):
                    raise item
                return item
            except queue.Empty:
                if self._finished.is_set() and self.output_queue.empty():
                    raise StopIteration

        if getattr(self.retriever, 'manager', None):
            self.retriever.manager.shutdown()


    def close(self):
        self._finished.set()
        self.executor.shutdown(wait = True)
