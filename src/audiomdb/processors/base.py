from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List

class AudioProcessor(ABC):
    """
    Base class for audio processing tasks.
    Subclasses should implement the `process` method.
    """

    def __init__(self, keep_original: bool = False):
        self.keep_original = keep_original

    @abstractmethod
    def process(self, data:Union[np.ndarray, List[np.ndarray]], sample_rate, batch_size = 64) -> dict:
        """
        Process the input audio data.
        Args:
            data: Input audio data to be processed.
        Returns:
            Processed audio artifacts as a dictionary.
            :param batch_size:
            :param data:
            :param sample_rate:
        """
        pass

    def resample(self,data:np.ndarray, orig_sr:int, target_sr:int) -> np.ndarray:
        """
        Resample the input audio data to the target sample rate.
        Args:
            data: Input audio data to be resampled.
            orig_sr: Original sample rate of the audio data.
            target_sr: Target sample rate for the audio data.
        Returns:
            Resampled audio data.
        """
        import librosa
        if orig_sr != target_sr:
            data = librosa.resample(data.T, orig_sr=orig_sr, target_sr=target_sr).T
        return data

    def get_duration(self, data:np.ndarray, sample_rate:int) -> float:
        """
        Calculate the duration of the audio data in seconds.
        Args:
            data: Input audio data.
            sample_rate: Sample rate of the audio data.
        Returns:
            Duration of the audio data in seconds.
        """
        return len(data) / sample_rate

    def bytes_to_np_array(self, audio_bytes:bytes, shape:tuple, dtype:str):
        """
            Convert raw bytes back to a numpy array.

            Args:
                audio_bytes: The raw bytes saved earlier (from .tobytes()).
                shape: Original array shape (e.g. (16000,) or (channels, samples)).
                dtype: Original dtype as string (e.g. 'float32').

            Returns:
                Numpy array with correct shape and dtype.
            """
        array = np.frombuffer(audio_bytes, dtype=dtype)
        return array.reshape(shape)


class TextProcessor(ABC):
    """
    Base class for text processing tasks.
    Subclasses should implement the `process` method.
    """
    def __init__(self, keep_original: bool = False):
        self.keep_original = keep_original

    @abstractmethod
    def process(self, text:Union[str, List[str]]) -> dict:
        """
        Process the input text data.
        Args:
            text: Input text data to be processed.
        Returns:
            Processed text artifacts:
        """
        pass



