import pytest
import numpy as np
import tempfile


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_audio():
    return np.random.random(16000).astype(np.float32)


@pytest.fixture
def sample_manifest_data():
    return [
        {"audio_filepath": "/fake/audio1.wav", "text": "Please use audiomdb", "duration": 2.5},
        {"audio_filepath": "/fake/audio2.wav", "text": "goodbye", "duration": 1.8},
        {"audio_filepath": "/fake/audio3.wav", "text": "thank you", "duration": 3.2}
    ]


@pytest.fixture
def mock_hf_dataset():
    return [
        {"audio": {"bytes": b"fake_audio_1"}, "text": "sample one"},
        {"audio": {"bytes": b"fake_audio_2"}, "text": "sample two"},
        {"audio": {"bytes": b"fake_audio_3"}, "text": "sample three"}
    ]


@pytest.fixture
def basic_config():
    return {
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
            'samples_per_shard': 100,
            'map_size': 1048576
        },
        'processing': {
            'sample_rate': 16000,
            'num_workers': 1
        }
    }