import os
import json
from typing import Optional, List
from audiomdb.converters.base import BaseConverter

class FileConverter(BaseConverter):
    """
    Convert an audio dataset described by a manifest file into sharded LMDB format.

    The manifest file should be a JSONL (one JSON object per line) with fields:
        {
            "audio_filepath": "/path/to/audio.wav",
            "text": "transcription text",
            "duration": 3.45,        # optional
            "speaker": "spk123"      # optional
        }

    Example:
        converter = FileConverter(
            manifest="data/train_manifest.json",
            output_dir="./lmdb_train",
            samples_per_shard=10000,
            sample_rate=16000
        )
        converter.convert()
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

    def converter_name(self) -> str:
        return "manifest_file"
