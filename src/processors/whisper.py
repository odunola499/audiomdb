from transformers import WhisperFeatureExtractor, WhisperTokenizer
from src.processors import AudioProcessor, TextProcessor
import numpy as np


class AudioMDWhisperFeatureExtractor(AudioProcessor):
    """
    Audio processor that extracts features using the Whisper feature extractor from Hugging Face Transformers.
    Example:
        from transformers import WhisperFeatureExtractor
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        processor = AudioMDWhisperFeatureExtractor(feature_extractor=feature_extractor, keep_original=True)
        processed_audio = processor.process(audio_data)
    """

    def __init__(self, feature_extractor: WhisperFeatureExtractor, keep_original: bool = False):
        super().__init__(keep_original)
        self.feature_extractor = feature_extractor

    def process(self, data: np.ndarray, sample_rate):
        """
        Extract features from the input audio data using the Whisper feature extractor.
        Args:
            data: Input audio data to be processed.
        Returns:
            Extracted audio features.
            :param data:
            :param sample_rate:
        """
        if sample_rate != 16000:
            data = self.resample(data, orig_sr=sample_rate, target_sr=16000)
        features = self.feature_extractor(data, sampling_rate=16000, return_tensors="np").input_features
        return {
            'audio_features': features[0]
        }


class TextMDWhisperTokenizer(TextProcessor):
    """
    Text processor that tokenizes text using the Whisper tokenizer from Hugging Face Transformers.
    Example:
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
        processor = TextMDWhisperTokenizer(tokenizer=tokenizer, keep_original=True)
        processed_text = processor.process(text_data)
    """

    def __init__(self, tokenizer: WhisperTokenizer, keep_original: bool = False):
        super().__init__(keep_original)
        self.tokenizer = tokenizer

    def process(self, data: str, sample_rate=None, padding = 'max_length', max_length = 448, truncation = True,  **kwargs):
        """
        Tokenize the input text data using the Whisper tokenizer.
        Args:
            data: Input text data to be processed.
        Returns:
            Tokenized text data.
            :param truncation:
            :param max_length:
            :param padding:
            :param data:
            :param sample_rate:
        """
        tokens = self.tokenizer(data, return_tensors="np", padding=padding, truncation=truncation, max_length=max_length)
        return {
            'input_ids': tokens.input_ids[0],
            'attention_mask': tokens.attention_mask[0],
            'num_tokens': len(tokens.input_ids[0])
        }

    def pad_token(self):
        return self.tokenizer.pad_token

    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def bos_token(self):
        return self.tokenizer.bos_token

    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    def eos_token(self):
        return self.tokenizer.eos_token