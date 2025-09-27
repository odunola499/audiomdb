import tempfile
import shutil
import json
import os
import pytest
from src.converters.file_converter import FileConverter


class TestFileConverter:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_file = os.path.join(self.temp_dir, "manifest.jsonl")
        
        manifest_data = [
            {"audio_filepath": "/fake/path1.wav", "text": "hello world", "duration": 2.5},
            {"audio_filepath": "/fake/path2.wav", "text": "goodbye", "duration": 1.8}
        ]
        
        with open(self.manifest_file, 'w') as f:
            for item in manifest_data:
                f.write(json.dumps(item) + '\n')

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_converter_init(self):
        converter = FileConverter(
            manifest=self.manifest_file,
            output_dir=self.temp_dir
        )
        assert len(converter.entries) == 2
        assert converter.audio_column == "audio_filepath"
        assert converter.text_column == "text"

    def test_file_converter_missing_manifest(self):
        with pytest.raises(FileNotFoundError):
            FileConverter(
                manifest="/nonexistent/file.jsonl",
                output_dir=self.temp_dir
            )

    def test_sample_iterator(self):
        converter = FileConverter(
            manifest=self.manifest_file,
            output_dir=self.temp_dir
        )
        
        samples = list(converter.sample_iterator())
        assert len(samples) == 2
        
        key1, sample1 = samples[0]
        assert key1 == "sample_00000000"
        assert sample1["audio"] == "/fake/path1.wav"
        assert sample1["text"] == "hello world"
        assert sample1["converter"] == "manifest_file"

    def test_custom_columns(self):
        converter = FileConverter(
            manifest=self.manifest_file,
            output_dir=self.temp_dir,
            audio_column="audio_filepath",
            text_column="text"
        )
        
        samples = list(converter.sample_iterator())
        sample_data = samples[0][1]
        assert sample_data["audio"] == "/fake/path1.wav"
        assert sample_data["text"] == "hello world"

    def test_store_columns(self):
        converter = FileConverter(
            manifest=self.manifest_file,
            output_dir=self.temp_dir,
            store_columns=["duration"]
        )
        
        samples = list(converter.sample_iterator())
        sample_data = samples[0][1]
        assert "duration" in sample_data
        assert sample_data["duration"] == 2.5

    def test_converter_name(self):
        converter = FileConverter(
            manifest=self.manifest_file,
            output_dir=self.temp_dir
        )
        assert converter.converter_name == "manifest_file"

    def test_empty_manifest(self):
        empty_manifest = os.path.join(self.temp_dir, "empty.jsonl")
        with open(empty_manifest, 'w') as f:
            pass
        
        converter = FileConverter(
            manifest=empty_manifest,
            output_dir=self.temp_dir
        )
        assert len(converter.entries) == 0
        samples = list(converter.sample_iterator())
        assert len(samples) == 0