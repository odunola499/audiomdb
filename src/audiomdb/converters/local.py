import os
import json
from typing import Optional, List
from audiomdb.converters.base import BaseConverter

class FileConverter(BaseConverter):
    """
    Convert an audio dataset described by a JSONL manifest into sharded LMDB.

    Each line in the manifest is a JSON object, typically including:
    - audio_filepath: path to an audio file
    - text: optional transcription
    - duration, speaker, etc.: optional metadata

    The converter loads file paths, defers decoding to the writer, applies any
    configured processors, and writes shards and a metadata.json.
    """

    def __init__(
        self,
        manifest: str,
        output_dir:str,
        samples_per_shard:int = 50_000,
        map_size:int = 1 << 40,
        num_workers:int = 4,
        processors:dict = None,
        audio_column: str = "audio_filepath",
        text_column: Optional[str] = "text",
        store_columns: Optional[List[str]] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(
            output_dir=output_dir,
            samples_per_shard = samples_per_shard,
            map_size = map_size,
            num_workers = num_workers,
            processors = processors,
            )

        if not os.path.exists(manifest):
            raise FileNotFoundError(f"Manifest file {manifest} not found.")

        self.manifest = manifest
        self.audio_column = audio_column
        self.text_column = text_column
        self.store_columns = store_columns
        self.sample_rate = sample_rate

        self.dataset_name = manifest
        self.version = manifest

        with open(manifest, "r", encoding="utf-8") as f:
            self.entries = [json.loads(line) for line in f]

    def sample_iterator(self):
        for idx, item in enumerate(self.entries):
            key = f"sample_{idx:08d}"
            audio_path = item.get(self.audio_column)
            text = item.get(self.text_column, "")

            sample = {
                "audio": audio_path,
                "sample_rate": self.sample_rate,
                "text": text,
                "converter": self.converter_name,
            }

            if self.store_columns:
                for col in self.store_columns:
                    if col in item:
                        sample[col] = item[col]

            yield key, sample

    @property
    @property
    def converter_name(self) -> str:
        return "manifest_file"
