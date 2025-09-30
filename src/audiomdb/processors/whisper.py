from transformers import WhisperFeatureExtractor, WhisperTokenizer
from audiomdb.processors import AudioProcessor, TextProcessor
import numpy as np
from typing import List, Union



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

    def process(self, data:Union[np.ndarray, List[np.ndarray]], sample_rate, batch_size = 64):
        """
        Extract features from the input audio data using the Whisper feature extractor.
        Args:
            data: Input audio data to be processed.
        Returns:
            Extracted audio features.
            :param batch_size:
            :param data:
            :param sample_rate:
        """
        if not isinstance(data, list):
            if sample_rate != 16000:
                data = self.resample(data, orig_sr=sample_rate, target_sr=16000)
            features = self.feature_extractor(data, sampling_rate=16000, return_tensors="np").input_features
            return {
                'audio_features': features[0]
            }
        else:
            all_features = []
            for row in range(0, len(data), batch_size):
                features = self.feature_extractor(data[row:row+batch_size], sampling_rate=16000).input_features
                all_features.extend(features)
            return {
                'audio_features': np.array(all_features)
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

    def process(self, text:Union[str, List[str]], padding = 'max_length', max_length = 448, truncation = True, batch_size = 64,  **kwargs):
        """
        Tokenize the input text data using the Whisper tokenizer.
        Args:
            data: Input text data to be processed.
        Returns:
            Tokenized text data.
            :param batch_size:
            :param text:
            :param truncation:
            :param max_length:
            :param padding:
            :param data:
            :param sample_rate:
        """
        if not isinstance(text, list):
            tokens = self.tokenizer(text, return_tensors="np", padding=padding, truncation=truncation, max_length=max_length)
            return {
                'input_ids': tokens.input_ids[0],
                'attention_mask': tokens.attention_mask[0],
                'num_tokens': len(tokens.input_ids[0])
            }
        else:
            all_input_ids = []
            all_attention_masks = []
            all_num_tokens = []
            for row in range(0, len(text), batch_size):
                tokens = self.tokenizer(text[row:row+batch_size], padding=padding, truncation=truncation, max_length=max_length)
                all_input_ids.extend(tokens.input_ids)
                all_attention_masks.extend(tokens.attention_mask)
                all_num_tokens.extend([len(ids) for ids in tokens.input_ids])
            return {
                'input_ids': np.array(all_input_ids),
                'attention_mask': np.array(all_attention_masks),
                'num_tokens': np.array(all_num_tokens)
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

    def eos_token_id(self):
        return self.tokenizer.eos_token_id