import os
import gc
import lmdb
import pickle
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import soundfile as sf
import librosa
import io
import multiprocessing as mp
from queue import Queue, Empty
import threading
from audiomdb.processors.base import BaseProcessor
from tqdm.auto import tqdm

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import humanize
import json
import hashlib
from datetime import datetime


def compute_checksum(file_path: str) -> str:
    """
    Compute checksum of a file using sha256
    :param file_path:
    :return: checksum
    """
    h = hashlib.new('sha256')
    file_path = os.path.join(file_path, 'data.mdb')
    with open(file_path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_array(data: Union[str, np.ndarray, bytes], sample_rate: int = 16000):
    """Convert input data to a numpy array at the target sample rate.

    Args:
        data: File path, numpy array, or raw bytes.
        sample_rate: Desired sample rate for the returned array.

    Returns:
        A float32 numpy array (mono).
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, str):
        array, sr = sf.read(data, dtype="float32", always_2d=True)
        array = array[0]
        if sr != sample_rate:
            array = librosa.resample(array.T, orig_sr=sr, target_sr=sample_rate).T
        del data
        return array
    elif isinstance(data, bytes):
        array, sr = sf.read(io.BytesIO(data))
        if sr != sample_rate:
            array = librosa.resample(array.T, orig_sr=sr, target_sr=sample_rate).T
        del data
        return array
    else:
        raise ValueError(f"Unsupported data type for load_array, got {type(data)}")


def process_sample(sample: dict, processor: BaseProcessor = None, **kwargs) -> dict:
    """Decode, process, and serialize a sample ready for LMDB."""
    audio_bytes = sample.get('audio')
    if audio_bytes is None:
        raise ValueError("Sample does not contain 'audio' data.")

    audio_array = load_array(data=audio_bytes, sample_rate=sample.get('sample_rate', kwargs.get('sample_rate', 16000)))
    sample['audio'] = audio_array.astype(np.float32)
    sample['shape'] = audio_array.shape
    sample['dtype'] = str(audio_array.dtype)
    sample['duration'] = len(audio_array) / sample.get('sample_rate', kwargs.get('sample_rate', 16000))

    if processor:
        result = processor.process(sample, **kwargs)
        del audio_array
        gc.collect()
        return result
    
    # No processor, just convert audio to bytes
    sample['audio'] = sample['audio'].tobytes()
    del audio_array
    gc.collect()
    return sample


class BaseConverter(ABC):
    """
    Convert datasets into sharded LMDB format.

    Subclasses implement sample_iterator() to yield (key, sample) pairs. This base
    handles buffering into shards, optional multiprocessing, applying processors,
    and writing LMDB environments plus a metadata.json summary.

    Memory considerations:
    - samples_per_shard controls how many samples are buffered in memory at once.
    - If raw audio bytes will be stored (no audio processors with keep_original=False),
      the constructor caps samples_per_shard at 1000 to limit peak memory.
    - map_size sets the LMDB map size per shard (bytes).
    - num_workers sets the number of threads for writing shards.
    - limit_iteration can limit the number of samples converted. (-1 for entire dataset). Has to be explicitly set in the subclass constructor. Check HF converter for inspiration.
    """

    def __init__(
            self,
            output_dir: str,
            samples_per_shard: int = 50_000,
            map_size: int = 500 * 1024 ** 3,
            num_workers: int = 4,
            processor: BaseProcessor = None,
            limit_iteration: int = -1,
            sample_rate: int = 16000,
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.map_size = map_size
        self.sample_rate = sample_rate

        os.makedirs(output_dir, exist_ok=True)

        self.num_workers = min(num_workers, mp.cpu_count())
        self.processor = processor
        will_store_audio_bytes = True
        if processor and not getattr(processor, 'keep_original', True):
            will_store_audio_bytes = False
        if will_store_audio_bytes and self.samples_per_shard > 1000:
            print("audio bytes detected in shards; reducing samples_per_shard to 1000")
            self.samples_per_shard = 1000

        self.limit_iteration = limit_iteration

    @abstractmethod
    def sample_iterator(self):
        """
        Yield (key, sample) pairs.
        `sample` should be serializable (pickle, msgpack, etc.). yielded audio data must be in bytes or numpy array format.
        Example:
            yield "sample_00001", {"audio": audio_array, "text": "hello world"}
        """
        pass

    @property
    @abstractmethod
    def converter_name(self) -> str:
        """
        Return a unique name for the converter.
        """
        pass

    @staticmethod
    def human_bytes(n):
        try:
            return humanize.naturalsize(n)
        except Exception:
            return f"{n} bytes"

    def print_header(self):
        console = Console() if Console else None
        processor_name = self.processor.__class__.__name__ if self.processor else "None"
        map_str = BaseConverter.human_bytes(self.map_size)
        mode = f"multithreading ({self.num_workers} workers)" if self.num_workers > 1 else "single-process"
        if console and Panel and Table and Text:
            table = Table.grid(expand=True)
            table.add_row(Text(f"Dataset: {getattr(self, 'dataset_name', 'unknown')}", style="bold"))
            table.add_row(f"Output: {os.path.abspath(self.output_dir)}")
            table.add_row(f"Mode: {mode}")
            table.add_row(f"Sharding: {self.samples_per_shard} samples/shard, map_size={map_str}")
            table.add_row(f"Processor: {processor_name}")
            console.print(Panel(table, title="AudioMDB Conversion", expand=True))
        else:
            print(f"Dataset: {getattr(self, 'dataset_name', 'unknown')}")
            print(f"Output: {os.path.abspath(self.output_dir)}")
            print(f"Mode: {mode}")
            print(f"Sharding: {self.samples_per_shard} samples/shard, map_size={map_str}")
            print(f"Processor: {processor_name}")

    def print_summary(self, total_samples, total_duration, shard_count, max_shard_size, elapsed):
        console = Console() if Console else None
        dur_str = f"{total_duration / 60:.1f} min" if total_duration < 3600 else f"{total_duration / 3600:.2f} h"
        if console and Table and Panel:
            table = Table(title="Summary", box=None)
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            table.add_row("Total samples", str(total_samples))
            table.add_row("Total duration", dur_str)
            table.add_row("Shards", str(shard_count))
            table.add_row("Max per shard", str(max_shard_size))
            table.add_row("Output dir", os.path.abspath(self.output_dir))
            table.add_row("Metadata", os.path.join(os.path.abspath(self.output_dir), 'metadata.json'))
            table.add_row("Elapsed", f"{elapsed:.2f} s")
            console.print(Panel(table, title="AudioMDB Conversion Complete", expand=True))
        else:
            print("Conversion complete")
            print(f"Total samples: {total_samples}")
            print(f"Total duration: {dur_str}")
            print(f"Shards: {shard_count}")
            print(f"Max per shard: {max_shard_size}")
            print(f"Output dir: {os.path.abspath(self.output_dir)}")
            print(f"Metadata: {os.path.join(os.path.abspath(self.output_dir), 'metadata.json')}")
            print(f"Elapsed: {elapsed:.2f} s")

    @staticmethod
    def _write_shard(shard_id: int, samples: list, output_dir, map_size, processor: BaseProcessor = None, sample_rate: int = 16000) -> tuple:
        shard_path = os.path.join(output_dir, f"shard_{shard_id:05d}")
        print(f"Opening shard {shard_path}")
        env = lmdb.open(shard_path, map_size=map_size)
        print(f"Opened {shard_path}")
        shard_duration = 0.0
        first_sample = None

        try:
            with env.begin(write=True) as txn:
                for key, sample in tqdm(samples, desc='Writing Shard'):
                    processed_sample = process_sample(sample, processor, sample_rate=sample_rate)
                    if first_sample is None:
                        first_sample = processed_sample
                    shard_duration += processed_sample.get('duration', 0.0)
                    txn.put(key.encode("utf-8"), pickle.dumps(processed_sample))

            print(f"Written shard {shard_path}")
            env.sync()
        except Exception as e:
            print(e)
        finally:
            env.close()
            print(f"Closed {shard_path}")
        return shard_path, shard_duration, first_sample

    def convert(self) -> None:
        """
        Convert the dataset to sharded LMDB format.
        This method processes samples sequentially and writes them to shards.
        :return:
        """
        import time as _t
        _start = _t.time()

        self.print_header()

        buffer = []
        buffer_position = 0
        shard_id = 0
        total_samples = 0
        total_duration = 0
        max_shard_size = 0
        test_sample = None
        shard_infos = []

        print("Starting conversion...")
        iterator = self.sample_iterator()
        pbar = tqdm(iterator, desc="Converting samples", unit="sample")

        probe_file = os.path.join(self.output_dir, 'metadata.json')

        print('working...')
        for idx, (key, sample) in enumerate(pbar):
            sample['shard_id'] = shard_id
            sample['shard_position'] = buffer_position

            if test_sample is None:
                probe = {k: v for k, v in sample.items() if k != 'audio'}
                probe['audio'] = '<audio:bytes>'
                test_sample = probe


            buffer_position += 1
            buffer.append((key, sample))
            total_samples += 1

            if len(buffer) >= self.samples_per_shard:
                shard_path, shard_duration, first_sample = BaseConverter._write_shard(shard_id, buffer, self.output_dir,
                                                                                      self.map_size, self.processor, self.sample_rate)
                total_duration += shard_duration
                if test_sample is None:
                    test_sample = first_sample
                size = os.path.getsize(os.path.join(shard_path,'data.mdb'))
                checksum = compute_checksum(shard_path)

                shard_infos.append({
                    'id': shard_id,
                    'path': shard_path,
                    'size_bytes': size,
                    "checksum_sha256": checksum
                })

                if total_duration < 3600:
                    duration_str = f"{total_duration / 60:.1f} min"
                else:
                    duration_str = f"{total_duration / 3600:.2f} h"

                pbar.set_postfix_str(f"dur={duration_str}")
                print(f"[Shard {shard_id:05d}] {len(buffer)} samples, {humanize.naturalsize(size)}")

                max_shard_size = max(max_shard_size, len(buffer))

                del buffer
                gc.collect()
                buffer = []
                shard_id += 1
                buffer_position = 0

        if buffer:
            shard_path, shard_duration, first_sample = BaseConverter._write_shard(shard_id, buffer, self.output_dir,
                                                                                  self.map_size, self.processor, self.sample_rate)
            total_duration += shard_duration
            if test_sample is None:
                test_sample = first_sample
            size = os.path.getsize(os.path.join(shard_path,'data.mdb'))
            checksum = compute_checksum(shard_path)

            del buffer
            gc.collect()

            shard_infos.append({
                "id": shard_id,
                "path": os.path.basename(shard_path),
                "size_bytes": size,
                "checksum_sha256": checksum
            })

            if total_duration < 3600:
                duration_str = f"{total_duration / 60:.1f} min"
            else:
                duration_str = f"{total_duration / 3600:.2f} h"

            pbar.set_postfix_str(f"dur={duration_str}")
            print(
                f"[Shard {shard_id:05d}] {total_samples - (shard_id * self.samples_per_shard)} samples, {humanize.naturalsize(size)}")

        info = {
            "dataset_name": getattr(self, "dataset_name", "unknown"),
            "version": getattr(self, "dataset_version", "unknown"),
            'dataset_size': total_samples,
            'total_duration': total_duration,
            'num_shards': shard_id,
            'max_item_in_shard': max_shard_size,
            'features': test_sample,
            "processors_applied": [self.processor.__class__.__name__] if self.processor else [],
            "shards": shard_infos

        }

        with open(probe_file, 'w') as fp:
            json.dump(info, fp, indent=4)

        _elapsed = _t.time() - _start
        self.print_summary(total_samples, total_duration, shard_id + 1, max_shard_size, _elapsed)

    def mp_convert(self) -> None:
        import time as _t
        _start = _t.time()

        self.print_header()

        task_queue = Queue(maxsize=self.num_workers * 2)
        metadata_lock = threading.Lock()
        metadata = {
            "total_samples": 0,
            "total_duration": 0.0,
            "shard_count": 0,
            "max_shard_size": 0,
            "test_sample": None,
            "shards": []
        }

        probe_file = os.path.join(self.output_dir, 'metadata.json')

        print('working...')

        def producer():
            buffer = []
            shard_id = 0
            buffer_position = 0

            for (key, sample) in self.sample_iterator():
                sample['shard_id'] = shard_id
                sample['shard_position'] = buffer_position

                with metadata_lock:
                    if metadata['test_sample'] is None:
                        probe = {k: v for k, v in sample.copy().items() if k != 'audio'}
                        probe['audio'] = '<audio:bytes>'
                        metadata['test_sample'] = probe

                buffer_position += 1
                buffer.append((key, sample))

                if len(buffer) >= self.samples_per_shard:
                    with metadata_lock:
                        metadata['total_samples'] += len(buffer)
                        metadata['max_shard_size'] = max(metadata['max_shard_size'], len(buffer))

                    task_queue.put((shard_id, buffer.copy()))

                    del buffer
                    gc.collect()
                    buffer = []

                    shard_id += 1
                    buffer_position = 0

            if buffer:
                with metadata_lock:
                    metadata['total_samples'] += len(buffer)
                    metadata['max_shard_size'] = max(metadata['max_shard_size'], len(buffer))

                task_queue.put((shard_id, buffer.copy()))

            for _ in range(self.num_workers):
                task_queue.put(None, timeout=5)

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        with mp.Pool(self.num_workers) as pool, tqdm(desc="Writing shards", unit="shard") as pbar:
            tasks = []
            sentinel_seen = 0
            while True:
                try:
                    task = task_queue.get(timeout=1)
                    if task is None:
                        sentinel_seen += 1
                        if sentinel_seen >= self.num_workers:
                            break
                        else:
                            continue

                    shard_id, samples = task

                    def callback(shard_path, shard_id=shard_id):
                        size = os.path.getsize(os.path.join(shard_path,'data.mdb'))
                        checksum = compute_checksum(shard_path)
                        with metadata_lock:
                            metadata["shards"].append({
                                "id": shard_id,
                                "path": os.path.basename(shard_path),
                                "size_bytes": size,
                                "checksum_sha256": checksum
                            })

                    tasks.append(
                        pool.apply_async(
                            BaseConverter._write_shard,
                            args=(shard_id, samples, self.output_dir, self.map_size, self.processor, self.sample_rate),
                            callback=lambda res, shard_id=shard_id: (callback(res[0], shard_id),
                                                                     metadata.__setitem__('test_sample', metadata.get(
                                                                         'test_sample') or (res[2] if isinstance(
                                                                         res[2].get('audio', None),
                                                                         (bytes, bytearray)) and not metadata.get(
                                                                         'test_sample') else res[2])))
                        )
                    )

                    with metadata_lock:
                        metadata['shard_count'] += 1
                        if metadata['total_duration'] < 3600:
                            dur_str = f"{metadata['total_duration'] / 60:.1f} min"
                        else:
                            dur_str = f"{metadata['total_duration'] / 3600:.2f} h"
                    pbar.set_postfix_str(f"dur={dur_str}, samples={metadata['total_samples']}")
                    pbar.update(1)

                except Empty:
                    continue

            for t in tasks:
                res = t.get()
                with metadata_lock:
                    if metadata['test_sample'] is None:
                        metadata['test_sample'] = res[2]
                    metadata['total_duration'] += res[1]

        producer_thread.join()

        test_sample = metadata['test_sample']
        if 'audio' in test_sample and isinstance(test_sample['audio'], bytes):
            metadata['test_sample']['audio'] = 'audio in bytes'

        info = {
            "dataset_name": getattr(self, "dataset_name", "unknown"),
            "version": getattr(self, "dataset_version", "unknown"),
            "creation_date": datetime.utcnow().isoformat(),
            "dataset_size": metadata['total_samples'],
            "total_duration": metadata['total_duration'],
            "num_shards": metadata['shard_count'],
            "max_item_in_shard": metadata['max_shard_size'],
            "features": metadata['test_sample'],
            "processors_applied": [self.processor.__class__.__name__] if self.processor else [],
            "shards": metadata['shards']
        }

        with open(probe_file, 'w') as fp:
            json.dump(info, fp, indent=4)

        _elapsed = _t.time() - _start
        self.print_summary(metadata['total_samples'], metadata['total_duration'], metadata['shard_count'],
                           metadata['max_shard_size'], _elapsed)

    def run(self, upload: str = None, **kwargs):
        if self.num_workers > 1:
            self.mp_convert()
        else:
            self.convert()
        if upload:
            from audiomdb.uploaders import S3Uploader, GCPUploader, HFUploader
            if upload == 's3':
                uploader = S3Uploader(**kwargs)
            elif upload == 'gcp':
                uploader = GCPUploader(**kwargs)
            elif upload == 'hf':
                uploader = HFUploader(**kwargs)
            else:
                raise ValueError("Unsupported upload type")
            uploader.upload_dir(self.output_dir)

