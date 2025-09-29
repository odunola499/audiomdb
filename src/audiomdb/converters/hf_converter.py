from audiomdb.converters.base import BaseConverter
from datasets import load_dataset, Audio
from typing import Optional


class HFConverter(BaseConverter):
    """
    Convert a Hugging Face dataset to sharded LMDB format.
    Example:
        converter = HFConverter(
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train"),
            output_dir = "./lmdb_common_voice_en_train",
            samples_per_shard = 10000
        )
        converter.convert()
    """
    def __init__(self, data_id:str,
                 output_dir: str,
                 samples_per_shard: int = 50_000,
                 map_size: int = 1 << 40,
                 num_workers: int = 4,
                 processors: dict = None,
                 audio_column:str = "audio",
                 text_column:Optional[str] = "text",
                 store_columns:Optional[list] = None,
                 data_name:str = None,
                 data_split:str = "train",
                 hf_stream:bool = True,
                 sample_rate = 16000,
                 version = 1234,

                 ):
        super().__init__(
            output_dir=output_dir,
            samples_per_shard=samples_per_shard,
            map_size=map_size,
            num_workers=num_workers,
            processors=processors,
        )
        dataset = load_dataset(data_id,
                                    data_name,
                                    split = data_split,
                                    streaming = hf_stream)

        dataset = dataset.cast_column(audio_column, Audio(decode = False))

        self.dataset = dataset
        self.dataset_name = data_id
        self.version = version

        self.audio_column = audio_column
        self.text_column = text_column
        self.store_columns = store_columns
        self.sample_rate = sample_rate


    def sample_iterator(self):
        for idx, item in enumerate(self.dataset):
            key = f"sample_{idx:08d}"
            audio_bytes = item.get(self.audio_column, '').get('bytes')
            text = item.get(self.text_column, '')

            sample = {
                'audio': audio_bytes,
                'sample_rate': self.sample_rate,
                'text': text,
                'converter': self.converter_name,
            }
            if self.store_columns:
                #todo" some store columns can be a data type that has to be sent to bytes
                for col in self.store_columns:
                    if col in item.keys():
                        sample[col] = item[col]

            yield key, sample

    def converter_name(self) -> str:
        return 'hf'