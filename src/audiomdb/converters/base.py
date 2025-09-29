import os
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
from audiomdb.processors import AudioProcessor
from tqdm.auto import tqdm
import humanize
import json
import hashlib
from datetime import datetime


def compute_checksum(file_path:str) -> str:
    """
    Compute checksum of a file using sha256
    :param file_path:
    :return: checksum
    """
    h =hashlib.new('sha256')
    file_path = os.path.join(file_path,'data.mdb')
    with open(file_path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_array(data: Union[str, np.ndarray, bytes], sample_rate: int = 16000):
    """
    Convert input data to a numpy array.
    Args:
        data: Input data, can be a file path, numpy array, or bytes.
        sample_rate: Desired sample rate for audio data.
    Returns:
        Numpy array representation of the input data.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, str):
        array, sr = sf.read(data, dtype="float32", always_2d=True)
        array = array[0] # TODO: Would remove when stable to support stereo
        if sr != sample_rate:
            array = librosa.resample(array.T, orig_sr=sr, target_sr=sample_rate).T
        return array
    elif isinstance(data, bytes):
        array, sr = sf.read(io.BytesIO(data))
        if sr != sample_rate:
            array = librosa.resample(array.T, orig_sr=sr, target_sr=sample_rate).T
        return array
    else:
        raise ValueError(f"Unsupported data type for load_array, got {type(data)}")

def process_sample(sample: dict, processors: dict = None) -> dict:

    audio_bytes = sample.get('audio',None)
    if audio_bytes is None:
        raise ValueError("Sample does not contain 'audio' data.")

    audio_array = load_array(data=audio_bytes, sample_rate=sample.get('sample_rate', 16000))
    sample['audio'] = audio_array.astype(np.float32)
    sample['shape'] = audio_array.shape
    sample['dtype'] = str(audio_array.dtype)
    sample['duration'] = len(audio_array) / sample.get('sample_rate', 16000)

    final_results = sample.copy()

    if processors:
        for processor_type, processor_list in processors.items():
            if processor_list:
                for column_name, proc in processor_list:
                    data = sample.get(column_name)
                    if not data:
                        raise ValueError(f"Sample does not contain '{column_name}' data. Did you include {column_name} in `store_columns`?")

                    if isinstance(proc, AudioProcessor):
                        result = proc.process(data, sample_rate=sample.get('sample_rate', 16000))
                    else:
                        result = proc.process(data)

                    if proc.keep_original:
                        final_results.update(result)
                    else:
                        if column_name == 'audio':
                            final_results.pop('audio', None)
                            final_results.pop('shape', None)
                            final_results.pop('dtype', None)
                        else:
                            final_results.pop(column_name, None)

                        final_results.update(result)

    if 'audio' in final_results:
        current_audio = final_results['audio']
        if isinstance(current_audio, np.ndarray):
            final_results['audio'] = current_audio.tobytes()
            final_results['shape'] = current_audio.shape
            final_results['dtype'] = current_audio.dtype

    del sample
    return final_results


class BaseConverter(ABC):
    """
        Base class for converting datasets into sharded LMDB format.
        Subclasses should implement `sample_iterator`, which yields (key, value) pairs.
        """
    def __init__(
            self,
            output_dir:str,
            samples_per_shard:int = 50_000,
            map_size:int = 1 << 40,
            num_workers:int = 4,
            processors:dict = None
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.map_size = map_size
        os.makedirs(output_dir, exist_ok = True)
        self.num_workers = min(num_workers, mp.cpu_count())
        self.processors = processors or {}


    @abstractmethod
    def sample_iterator(self):
        """
        Yield (key, sample) pairs.
        `sample` should be serializable (pickle, msgpack, etc.).
        Example:
            yield "sample_00001", {"audio": audio_array, "text": "hello world"}
        """
        pass


    @staticmethod
    def _write_shard(shard_id:int, samples:list, output_dir, map_size, processors:dict = None) -> str:
        """
        Write a shard to LMDB.
        Args:
        :param shard_id:
        :param samples:
        :param output_dir:
        :param map_size:
        :param processors:
        :return:
        """
        shard_path = os.path.join(output_dir, f"shard_{shard_id:05d}")
        env = lmdb.open(
            shard_path, map_size = map_size
        )
        shard_duration = 0.0
        
        try:
            with env.begin(write = True) as txn:
                for key, sample in samples:
                    sample = process_sample(sample, processors or {})
                    shard_duration += sample.get('duration', 0.0)
                    txn.put(key.encode("utf-8"), pickle.dumps(sample))

            env.sync()
        finally:
            env.close()
        return shard_path, shard_duration

    def convert(self) -> None:
        """
        Convert the dataset to sharded LMDB format.
        This method processes samples sequentially and writes them to shards.
        :return:
        """
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
        pbar = tqdm(iterator, desc = "Converting samples", unit = "sample")

        probe_file = os.path.join(self.output_dir, 'metadata.json')

        print('working...')
        for idx, (key, sample) in enumerate(pbar):
            sample['shard_id'] = shard_id
            sample['shard_position'] = buffer_position

            if test_sample is None:
                test_sample = sample

            buffer_position += 1
            buffer.append((key, sample))
            total_samples += 1

            if len(buffer) >= self.samples_per_shard:
                shard_path, shard_duration = BaseConverter._write_shard(shard_id, buffer, self.output_dir, self.map_size, self.processors)
                total_duration += shard_duration
                size = os.path.getsize(shard_path)
                checksum = compute_checksum(shard_path)

                shard_infos.append({
                    'id': shard_id,
                    'path': os.path.abspath(shard_path),
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
                buffer.clear()
                shard_id += 1
                buffer_position = 0

        if buffer:
            shard_path, shard_duration = BaseConverter._write_shard(shard_id, buffer, self.output_dir, self.map_size, self.processors)
            total_duration += shard_duration
            size = os.path.getsize(shard_path)
            checksum = compute_checksum(shard_path)

            shard_infos.append({
                "id": shard_id,
                "path": os.path.abspath(shard_path),
                "size_bytes": size,
                "checksum_sha256": checksum
            })


            if total_duration < 3600:
                duration_str = f"{total_duration / 60:.1f} min"
            else:
                duration_str = f"{total_duration / 3600:.2f} h"

            pbar.set_postfix_str(f"dur={duration_str}")
            print(f"[Shard {shard_id:05d}] {total_samples - (shard_id * self.samples_per_shard)} samples, {humanize.naturalsize(size)}")

        if 'audio' in test_sample:
            if isinstance(test_sample['audio'], bytes):
                test_sample['audio'] = 'ndarray in bytes' #todo: figure out a better way to do this
        info = {
            "dataset_name": getattr(self, "dataset_name", "unknown"),
            "version": getattr(self, "dataset_version", "unknown"),
            'dataset_size': total_samples,
            'total_duration': total_duration,
            'num_shards': shard_id,
            'max_item_in_shard': max_shard_size,
            'features': test_sample,
            "processors_applied": [p.__class__.__name__ for plist in self.processors.values() for _, p in plist],
            "shards": shard_infos

        }

        with open(probe_file, 'w') as fp:
            json.dump(info, fp, indent = 4)

        print(f"Conversion complete. {shard_id + 1} shards created in {self.output_dir}.")

    def mp_convert(self) -> None:
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
                        metadata['test_sample'] = sample.copy()

                buffer_position += 1
                buffer.append((key, sample))

                if len(buffer) >= self.samples_per_shard:
                    with metadata_lock:
                        metadata['total_samples'] += len(buffer)
                        metadata['max_shard_size'] = max(metadata['max_shard_size'], len(buffer))

                    task_queue.put((shard_id, buffer.copy()))
                    buffer.clear()
                    shard_id += 1
                    buffer_position = 0

            if buffer:
                with metadata_lock:
                    metadata['total_samples'] += len(buffer)
                    metadata['max_shard_size'] = max(metadata['max_shard_size'], len(buffer))


                task_queue.put((shard_id, buffer.copy()))

            for _ in range(self.num_workers):
                task_queue.put(None)

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        with mp.Pool(self.num_workers) as pool, tqdm(desc="Writing shards", unit="shard") as pbar:
            tasks = []
            while True:
                try:
                    task = task_queue.get(timeout=1)
                    if task is None:
                        break

                    shard_id, samples = task

                    def callback(shard_path, shard_id=shard_id):
                        size = os.path.getsize(shard_path)
                        checksum = compute_checksum(shard_path)
                        with metadata_lock:
                            metadata["shards"].append({
                                "id": shard_id,
                                "path": os.path.abspath(shard_path),
                                "size_bytes": size,
                                "checksum_sha256": checksum
                            })

                    tasks.append(
                        pool.apply_async(
                            BaseConverter._write_shard,
                            args=(shard_id, samples, self.output_dir, self.map_size, self.processors),
                            callback=lambda res, shard_id=shard_id: callback(res[0], shard_id)
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
                    metadata['total_duration'] += res[1]

        producer_thread.join()

        test_sample = metadata['test_sample']
        if 'audio' in test_sample and isinstance(test_sample['audio'], bytes):
            metadata['test_sample']['audio'] = 'ndarray in bytes'

        info = {
            "dataset_name": getattr(self, "dataset_name", "unknown"),
            "version": getattr(self, "dataset_version", "unknown"),
            "creation_date": datetime.utcnow().isoformat(),
            "dataset_size": metadata['total_samples'],
            "total_duration": metadata['total_duration'],
            "num_shards": metadata['shard_count'],
            "max_item_in_shard": metadata['max_shard_size'],
            "features": metadata['test_sample'],
            "processors_applied": [p.__class__.__name__ for plist in self.processors.values() for _, p in plist],
            "shards": metadata['shards']
        }

        with open(probe_file, 'w') as fp:
            json.dump(info, fp, indent=4)

        print('Done')

    def run(self):
        """
        Run the conversion process, using multiprocessing if specified.
        :return:
        """
        if self.num_workers > 1:
            self.mp_convert()
        else:
            self.convert()


    @property
    @abstractmethod
    def converter_name(self) -> str:
        """
        Return a unique name for the converter.
        """
        pass