import os
import threading
import queue
import time
from typing import Optional, Dict, Tuple


class CacheManager:
    def __init__(self, retriever, cache_dir: str, max_cache_bytes: Optional[int] = None, workers: int = 2):
        self.retriever = retriever
        self.cache_dir = cache_dir
        self.max_cache_bytes = max_cache_bytes
        self.workers = max(1, workers)
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._desired = set()
        self._sizes: Dict[str, int] = {}
        self._lru: Dict[str, float] = {}
        self._threads = []
        for _ in range(self.workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def request(self, file_id: str):
        name = os.path.basename(file_id)
        if name not in self._desired:
            self._desired.add(name)
            self._q.put(name)

    def has(self, file_id: str) -> bool:
        name = os.path.basename(file_id)
        path = os.path.join(self.cache_dir, name)
        return os.path.isdir(path)

    def mark_used(self, file_id: str):
        name = os.path.basename(file_id)
        self._lru[name] = time.time()

    def shutdown(self):
        self._stop.set()
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join(timeout=1)

    def _dir_size(self, path: str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    @property
    def _total_bytes(self) -> int:
        return sum(self._sizes.values())

    def _evict_if_needed(self):
        if not self.max_cache_bytes:
            return
        while self._total_bytes > self.max_cache_bytes and self._sizes:
            if not self._lru:
                break
            victim = min(self._lru.items(), key=lambda kv: kv[1])[0]
            if victim in self._desired:
                self._lru.pop(victim, None)
                continue
            path = os.path.join(self.cache_dir, victim)
            try:
                self.retriever.delete_file_from_cache(path)
            except Exception:
                pass
            self._sizes.pop(victim, None)
            self._lru.pop(victim, None)

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            name = item
            path = os.path.join(self.cache_dir, name)
            if os.path.isdir(path):
                self._sizes.setdefault(name, self._dir_size(path))
                self._lru[name] = time.time()
                continue
            try:
                self.retriever.get_file_into_cache(name)
                if os.path.isdir(path):
                    self._sizes[name] = self._dir_size(path)
                    self._lru[name] = time.time()
                    self._evict_if_needed()
            except Exception:
                continue
