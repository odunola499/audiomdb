from transformers import WhisperFeatureExtractor, WhisperTokenizer
from audiomdb.processors.base import BaseProcessor
import numpy as np
from typing import Optional, Dict, Any


class WhisperProcessor(BaseProcessor):
    """
    Processor that handles both Whisper feature extraction and tokenization.
    """
    
    def __init__(self, feature_extractor: Optional[WhisperFeatureExtractor] = None, 
                 tokenizer: Optional[WhisperTokenizer] = None, keep_original: bool = False):
        super().__init__(keep_original)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        
    def process(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:

        sample_rate = kwargs.get('sample_rate', 16000)
        result = sample.copy() if self.keep_original else {}

        if self.feature_extractor and 'audio' in sample:
            audio_data = sample['audio']

            if isinstance(audio_data, bytes):
                audio_array = self.load_array(audio_data, sample_rate)
            else:
                audio_array = audio_data

            current_sr = sample.get('sample_rate', sample_rate)
            if current_sr != 16000:
                audio_array = self.resample(audio_array, current_sr, 16000)

            features = self.feature_extractor(audio_array, sampling_rate=16000, return_tensors="np")
            result['audio_features'] = features.input_features[0]

            if not self.keep_original:
                result['shape'] = audio_array.shape
                result['dtype'] = str(audio_array.dtype)
                result['duration'] = self.get_duration(audio_array, 16000)
                result['sample_rate'] = 16000

        if self.tokenizer and 'text' in sample:
            text = sample['text']
            padding = kwargs.get('padding', 'max_length')
            max_length = kwargs.get('max_length', 448)
            truncation = kwargs.get('truncation', True)
            
            tokens = self.tokenizer(
                text, 
                return_tensors="np", 
                padding=padding, 
                truncation=truncation, 
                max_length=max_length
            )
            
            result['input_ids'] = tokens.input_ids[0]
            result['attention_mask'] = tokens.attention_mask[0]
            result['num_tokens'] = len(tokens.input_ids[0])

        if 'audio' in result and isinstance(result['audio'], np.ndarray):
            result['audio'] = result['audio'].tobytes()
            
        return result


    
    def pad_token(self):
        return self.tokenizer.pad_token if self.tokenizer else None

    def pad_token_id(self):
        return self.tokenizer.pad_token_id if self.tokenizer else None

    def bos_token(self):
        return self.tokenizer.bos_token if self.tokenizer else None

    def bos_token_id(self):
        return self.tokenizer.bos_token_id if self.tokenizer else None

    def eos_token(self):
        return self.tokenizer.eos_token if self.tokenizer else None

    def eos_token_id(self):
        return self.tokenizer.eos_token_id if self.tokenizer else None