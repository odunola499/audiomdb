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

        try:
            with env.begin(write = True) as txn:
                for key, sample in samples:
                    sample = process_sample(sample, processors or {})
                    txn.put(key.encode("utf-8"), pickle.dumps(sample))

            env.sync()
        finally:
            env.close()
        return shard_path

    def convert(self) -> None:
        """
        Convert the dataset to sharded LMDB format.
        This method processes samples sequentially and writes them to shards.
        :return:
        """
        buffer = []
        shard_id = 0
        total_samples = 0
        total_duration = 0

        print("Starting conversion...")
        iterator = self.sample_iterator()
        pbar = tqdm(iterator, desc = "Converting samples", unit = "sample")

        print('working...')
        for idx, (key, sample) in enumerate(pbar):
            buffer.append((key, sample))
            total_samples += 1
            if 'duration' in sample:
                total_duration += sample['duration']

            if len(buffer) >= self.samples_per_shard:
                shard_path = BaseConverter._write_shard(shard_id, buffer, self.output_dir, self.map_size, self.processors)
                size = os.path.getsize(shard_path)
                if total_duration < 3600:
                    duration_str = f"{total_duration / 60:.1f} min"
                else:
                    duration_str = f"{total_duration / 3600:.2f} h"

                pbar.set_postfix_str(f"dur={duration_str}")
                print(f"[Shard {shard_id:05d}] {len(buffer)} samples, {humanize.naturalsize(size)}")
                buffer.clear()
                shard_id += 1

        if buffer:
            shard_path = BaseConverter._write_shard(shard_id, buffer, self.output_dir, self.map_size, self.processors)
            size = os.path.getsize(shard_path)
            if total_duration < 3600:
                duration_str = f"{total_duration / 60:.1f} min"
            else:
                duration_str = f"{total_duration / 3600:.2f} h"

            pbar.set_postfix_str(f"dur={duration_str}")
            print(f"[Shard {shard_id:05d}] {len(buffer)} samples, {humanize.naturalsize(size)}")

        print(f"Conversion complete. {shard_id + 1} shards created in {self.output_dir}.")

    def mp_convert(self) -> None:
        """
        Convert the dataset to sharded LMDB format using multiprocessing.
        This method uses multiple processes to process samples and write them to shards.
        :return:
        """
        task_queue = Queue(maxsize = self.num_workers * 2)
        total_samples = 0
        total_duration = 0.0
        shard_count = 0

        print('working...')
        def producer():
            buffer = []
            shard_id = 0

            for (key, sample) in self.sample_iterator():
                buffer.append((key, sample))

                if len(buffer) >= self.samples_per_shard:
                    task_queue.put((shard_id, buffer.copy()))
                    buffer.clear()
                    shard_id += 1

            if buffer:
                task_queue.put((shard_id, buffer.copy()))

            for _ in range(self.num_workers):
                task_queue.put(None)

        producer_thread = threading.Thread(target = producer)
        producer_thread.start()

        with mp.Pool(self.num_workers) as pool, tqdm(desc="Writing shards", unit="shard") as pbar:
            tasks = []

            while True:
                try:
                    task = task_queue.get(timeout = 1)
                    if task is None:
                        break

                    shard_id, samples = task
                    total_samples += len(samples)
                    for _, s in samples:
                        if "duration" in s:
                            total_duration += s["duration"]

                    tasks.append(
                        pool.apply_async(
                            BaseConverter._write_shard, args = (shard_id, samples, self.output_dir, self.map_size, self.processors)
                        )
                    )

                    shard_count += 1
                    if total_duration < 3600:
                        dur_str = f"{total_duration / 60:.1f} min"
                    else:
                        dur_str = f"{total_duration / 3600:.2f} h"
                    pbar.set_postfix_str(f"dur={dur_str}, samples={total_samples}")
                    pbar.update(1)

                except Empty:
                    continue

            for t in tasks:
                t.get()

        producer_thread.join()

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