import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.converters.hf_converter import HFConverter


class TestHFConverter:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.converters.hf_converter.load_dataset')
    def test_hf_converter_init(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        converter = HFConverter(
            data_id="test/dataset",
            output_dir=self.temp_dir,
            audio_column="audio",
            text_column="text"
        )

        assert converter.audio_column == "audio"
        assert converter.text_column == "text"
        mock_load_dataset.assert_called_once()

    @patch('src.converters.hf_converter.load_dataset')
    def test_sample_iterator(self, mock_load_dataset):
        mock_data = [
            {"audio": {"bytes": b"fake_audio_data"}, "text": "hello"},
            {"audio": {"bytes": b"more_fake_data"}, "text": "world"}
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(mock_data)
        mock_dataset.cast_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        converter = HFConverter(
            data_id="test/dataset",
            output_dir=self.temp_dir
        )

        samples = list(converter.sample_iterator())
        assert len(samples) == 2
        assert samples[0][0] == "sample_00000000"
        assert samples[1][0] == "sample_00000001"

    @patch('src.converters.hf_converter.load_dataset')
    def test_converter_name(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        converter = HFConverter(
            data_id="test/dataset",
            output_dir=self.temp_dir
        )
        assert converter.converter_name == 'hf'

    @patch('src.converters.hf_converter.load_dataset')
    def test_with_store_columns(self, mock_load_dataset):
        mock_data = [
            {
                "audio": {"bytes": b"fake_audio"},
                "text": "hello",
                "speaker": "person1",
                "duration": 1.5
            }
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(mock_data)
        mock_dataset.cast_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        converter = HFConverter(
            data_id="test/dataset",
            output_dir=self.temp_dir,
            store_columns=["speaker", "duration"]
        )

        samples = list(converter.sample_iterator())
        sample_data = samples[0][1]
        assert "speaker" in sample_data
        assert "duration" in sample_data
        assert sample_data["speaker"] == "person1"
        assert sample_data["duration"] == 1.5