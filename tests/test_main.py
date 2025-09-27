import tempfile
import shutil
import yaml
import pytest
from unittest.mock import patch, Mock
from main import load_config, create_processors, create_converter


class TestLoadConfig:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config_valid(self):
        config_data = {
            'converter': {'type': 'hf'},
            'output': {'directory': './test'},
            'processing': {'num_workers': 1}
        }
        config_path = f"{self.temp_dir}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_config(config_path)
        assert result['converter']['type'] == 'hf'
        assert result['output']['directory'] == './test'


class TestCreateProcessors:
    def test_create_processors_empty(self):
        result = create_processors({})
        assert result == {}

    def test_create_processors_none(self):
        result = create_processors(None)
        assert result == {}

    @patch('transformers.WhisperFeatureExtractor')
    def test_create_processors_whisper_audio(self, mock_extractor):
        try:
            processor_configs = {
                'audio': [{
                    'type': 'whisper_features',
                    'model_name': 'openai/whisper-small',
                    'keep_original': False
                }]
            }
            
            mock_extractor.from_pretrained.return_value = Mock()
            result = create_processors(processor_configs)
            
            assert 'audio' in result
            assert len(result['audio']) == 1
        except ImportError:
            pytest.skip("Transformers not available")

    @patch('audiomentations.AddGaussianNoise')
    @patch('audiomentations.Compose')
    def test_create_processors_audiomentations(self, mock_compose, mock_noise):
        try:
            processor_configs = {
                'audio': [{
                    'type': 'audiomentations',
                    'keep_original': True,
                    'augmentations': [{
                        'name': 'AddGaussianNoise',
                        'params': {'min_amplitude': 0.001, 'max_amplitude': 0.015, 'p': 0.5}
                    }]
                }]
            }
            
            mock_compose.return_value = Mock()
            result = create_processors(processor_configs)
            
            assert 'audio' in result
        except ImportError:
            pytest.skip("Audiomentations not available")

    @patch('transformers.WhisperTokenizer')
    def test_create_processors_whisper_text(self, mock_tokenizer):
        try:
            processor_configs = {
                'text': [{
                    'type': 'whisper_tokenizer',
                    'model_name': 'openai/whisper-small',
                    'keep_original': False
                }]
            }
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            result = create_processors(processor_configs)
            
            assert 'text' in result
            assert len(result['text']) == 1
        except ImportError:
            pytest.skip("Transformers not available")


class TestCreateConverter:
    @patch('src.converters.hf_converter.load_dataset')
    def test_create_converter_hf(self, mock_load_dataset):
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        config = {
            'converter': {
                'type': 'hf',
                'hf': {
                    'dataset_id': 'test/dataset',
                    'dataset_name': 'subset',
                    'split': 'train',
                    'streaming': False,
                    'audio_column': 'audio',
                    'text_column': 'text',
                    'store_columns': ['extra']
                }
            },
            'output': {
                'directory': './test_output',
                'samples_per_shard': 1000,
                'map_size': 1048576
            },
            'processing': {
                'sample_rate': 16000,
                'num_workers': 2
            }
        }
        
        converter = create_converter(config)
        assert converter.converter_name == 'hf'

    def test_create_converter_file(self):
        config = {
            'converter': {
                'type': 'file',
                'file': {
                    'manifest_path': '/fake/path.jsonl',
                    'audio_column': 'audio_filepath',
                    'text_column': 'text',
                    'store_columns': ['duration']
                }
            },
            'output': {
                'directory': './test_output',
                'samples_per_shard': 1000,
                'map_size': 1048576
            },
            'processing': {
                'sample_rate': 16000,
                'num_workers': 2
            }
        }
        
        with pytest.raises(FileNotFoundError):
            create_converter(config)

    def test_create_converter_invalid_type(self):
        config = {
            'converter': {'type': 'invalid'},
            'output': {'directory': './test', 'samples_per_shard': 1000, 'map_size': 1048576},
            'processing': {'sample_rate': 16000, 'num_workers': 1}
        }
        
        with pytest.raises(ValueError, match="Unknown converter type"):
            create_converter(config)

    @patch('src.converters.hf_converter.load_dataset')
    def test_create_converter_with_processors(self, mock_load_dataset):
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        config = {
            'converter': {
                'type': 'hf',
                'hf': {
                    'dataset_id': 'test/dataset',
                    'split': 'train',
                    'streaming': False,
                    'audio_column': 'audio',
                    'text_column': 'text'
                }
            },
            'output': {
                'directory': './test_output',
                'samples_per_shard': 1000,
                'map_size': 1048576
            },
            'processing': {
                'sample_rate': 16000,
                'num_workers': 1
            }
        }
        
        processors = {'audio': [('audio', Mock())]}
        converter = create_converter(config, processors)
        assert converter.processors == processors