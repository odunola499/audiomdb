import numpy as np
import pytest
from unittest.mock import Mock, patch
from src.processors.base import AudioProcessor, TextProcessor
from src.processors.audiomentations import AudiomentationsProcessor


class MockAudioProcessor(AudioProcessor):
    def process(self, data, sample_rate):
        return {"processed_audio": data * 2}


class MockTextProcessor(TextProcessor):
    def process(self, text):
        return {"processed_text": text.upper()}
    
    @property
    def pad_token(self):
        return "<pad>"
    
    @property
    def pad_token_id(self):
        return 0
    
    @property
    def eos_token(self):
        return "</s>"
    
    @property
    def eos_token_id(self):
        return 1
    
    @property
    def bos_token(self):
        return "<s>"
    
    @property
    def bos_token_id(self):
        return 2


class TestAudioProcessor:
    def test_audio_processor_process(self):
        processor = MockAudioProcessor(keep_original=False)
        data = np.array([1, 2, 3, 4])
        result = processor.process(data, 16000)
        assert "processed_audio" in result
        np.testing.assert_array_equal(result["processed_audio"], data * 2)

    def test_audio_processor_resample(self):
        processor = MockAudioProcessor()
        data = np.random.random(1000)
        
        with patch('librosa.resample') as mock_resample:
            mock_resample.return_value = data
            result = processor.resample(data, 22050, 16000)
            mock_resample.assert_called_once()

    def test_audio_processor_get_duration(self):
        processor = MockAudioProcessor()
        data = np.random.random(16000)
        duration = processor.get_duration(data, 16000)
        assert duration == 1.0

    def test_bytes_to_np_array(self):
        processor = MockAudioProcessor()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        audio_bytes = data.tobytes()
        result = processor.bytes_to_np_array(audio_bytes, data.shape, 'float32')
        np.testing.assert_array_equal(result, data)

    def test_keep_original_flag(self):
        processor = MockAudioProcessor(keep_original=True)
        assert processor.keep_original is True


class TestTextProcessor:
    def test_text_processor_process(self):
        processor = MockTextProcessor(keep_original=False)
        result = processor.process("hello world")
        assert result["processed_text"] == "HELLO WORLD"

    def test_text_processor_tokens(self):
        processor = MockTextProcessor()
        assert processor.pad_token == "<pad>"
        assert processor.pad_token_id == 0
        assert processor.eos_token == "</s>"
        assert processor.eos_token_id == 1
        assert processor.bos_token == "<s>"
        assert processor.bos_token_id == 2


class TestAudiomentationsProcessor:
    def test_audiomentations_processor_init(self):
        mock_augmentations = Mock()
        processor = AudiomentationsProcessor(mock_augmentations)
        assert processor.augmentations == mock_augmentations

    def test_audiomentations_processor_process(self):
        mock_augmentations = Mock()
        mock_augmentations.return_value = np.array([1, 2, 3, 4])
        
        processor = AudiomentationsProcessor(mock_augmentations)
        data = np.array([0.1, 0.2, 0.3, 0.4])
        result = processor.process(data, 16000)
        
        mock_augmentations.assert_called_once_with(samples=data, sample_rate=16000)
        assert "audio" in result
        np.testing.assert_array_equal(result["audio"], np.array([1, 2, 3, 4]))


@patch('transformers.WhisperFeatureExtractor')
@patch('transformers.WhisperTokenizer')
class TestWhisperProcessors:
    def test_whisper_feature_extractor_import(self, mock_tokenizer, mock_extractor):
        try:
            from src.processors.whisper import AudioMDWhisperFeatureExtractor, TextMDWhisperTokenizer
            assert AudioMDWhisperFeatureExtractor is not None
            assert TextMDWhisperTokenizer is not None
        except ImportError:
            pytest.skip("Transformers not available")

    def test_whisper_feature_extractor_process(self, mock_tokenizer, mock_extractor):
        try:
            from src.processors.whisper import AudioMDWhisperFeatureExtractor
            
            mock_extractor_instance = Mock()
            mock_extractor_instance.return_value.input_features = np.array([[1, 2, 3]])
            mock_extractor.from_pretrained.return_value = mock_extractor_instance
            
            processor = AudioMDWhisperFeatureExtractor(mock_extractor_instance)
            data = np.random.random(16000)
            result = processor.process(data, 16000)
            
            assert "audio_features" in result
        except ImportError:
            pytest.skip("Transformers not available")

    def test_whisper_tokenizer_process(self, mock_tokenizer, mock_extractor):
        try:
            from src.processors.whisper import TextMDWhisperTokenizer
            
            mock_tokenizer_instance = Mock()
            mock_tokens = Mock()
            mock_tokens.input_ids = np.array([[1, 2, 3, 4]])
            mock_tokens.attention_mask = np.array([[1, 1, 1, 0]])
            mock_tokenizer_instance.return_value = mock_tokens
            
            processor = TextMDWhisperTokenizer(mock_tokenizer_instance)
            result = processor.process("hello world")
            
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "num_tokens" in result
        except ImportError:
            pytest.skip("Transformers not available")