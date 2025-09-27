import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import patch
from src.converters.base import BaseConverter, load_array, process_sample


class MockConverter(BaseConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = [
            ("sample_001", {"audio": np.random.random(1000).astype(np.float32), "text": "hello", "sample_rate": 16000}),
            ("sample_002", {"audio": np.random.random(1000).astype(np.float32), "text": "world", "sample_rate": 16000})
        ]

    def sample_iterator(self):
        for key, sample in self.samples:
            yield key, sample

    @property
    def converter_name(self) -> str:
        return "mock"


class TestLoadArray:
    def test_load_array_numpy(self):
        arr = np.random.random(1000)
        result = load_array(arr)
        assert np.array_equal(result, arr)

    def test_load_array_bytes(self):
        arr = np.random.random(1000).astype(np.float32)
        audio_bytes = arr.tobytes()
        with patch('soundfile.read') as mock_read:
            mock_read.return_value = (arr, 16000)
            result = load_array(audio_bytes, 16000)
            assert isinstance(result, np.ndarray)

    def test_load_array_unsupported_type(self):
        with pytest.raises(ValueError):
            load_array(123)


class TestProcessSample:
    def test_process_sample_basic(self):
        sample = {
            "audio": np.random.random(1000).astype(np.float32),
            "text": "test",
            "sample_rate": 16000
        }
        result = process_sample(sample)
        assert "audio" in result
        assert "text" in result
        assert "shape" in result
        assert "dtype" in result
        assert "duration" in result

    def test_process_sample_no_audio(self):
        sample = {"text": "test"}
        with pytest.raises(ValueError):
            process_sample(sample)


class TestBaseConverter:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_converter_init(self):
        converter = MockConverter(
            output_dir=self.temp_dir,
            samples_per_shard=10
        )
        assert converter.output_dir == self.temp_dir
        assert converter.samples_per_shard == 10

    def test_converter_run_sequential(self):
        converter = MockConverter(
            output_dir=self.temp_dir,
            samples_per_shard=1,
            num_workers=1
        )
        converter.run()
        assert len(converter.samples) == 2

    def test_converter_run_multiprocessing(self):
        converter = MockConverter(
            output_dir=self.temp_dir,
            samples_per_shard=1,
            num_workers=2
        )
        converter.run()
        assert len(converter.samples) == 2