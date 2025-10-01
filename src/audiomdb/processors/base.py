from abc import ABC, abstractmethod
import numpy as np
import soundfile as sf
import librosa
import io
from typing import Union, Dict, Any


class BaseProcessor(ABC):
    """
    Base class for sample processing tasks.
    Subclasses should implement the `process` method.
    """

    def __init__(self, keep_original: bool = False):
        self.keep_original = keep_original

    @abstractmethod
    def process(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process the input sample.
        Args:
            sample: Input sample dictionary containing dataset row
            **kwargs: Additional processing arguments like sample_rate.
        Returns:
            Processed sample dictionary.
        """
        pass

    def load_array(self, data: Union[str, np.ndarray, bytes], sample_rate: int = 16000) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, str):
            array, sr = sf.read(data, dtype="float32", always_2d=True)
            array = array[0]
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

    def resample(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr != target_sr:
            data = librosa.resample(data.T, orig_sr=orig_sr, target_sr=target_sr).T
        return data

    def get_duration(self, data: np.ndarray, sample_rate: int) -> float:
        return len(data) / sample_rate

    def bytes_to_np_array(self, audio_bytes: bytes, shape: tuple, dtype: str) -> np.ndarray:
        array = np.frombuffer(audio_bytes, dtype=dtype)
        return array.reshape(shape)



