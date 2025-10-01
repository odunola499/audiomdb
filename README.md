# AudioMDB

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AudioMDB** is a Python library for converting and loading/streaming audio datasets in Lightning Memory-Mapped Database (LMDB) format, designed for very fast machine learning training workflows and in constrained environments like inadequate disk space/bandwidth.

This project came as a result of the need for a flexible, efficient way to handle and stream large-scale audio data in ML pipelines without GPUs waiting idly. 

While working on [AudioRL](https://github.com/odunola499/audiorl) and streaming per-samples from HuggingFace datasets, I realised that the existing libraries were not optimized for high-throughput training with large audio datasets. In multi-gpu setups especially, the bottleneck was often data loading and processing, not model training.
AudioMDB addresses this by providing a **high-performance, extensible framework** for audio data conversion, retrieval, and processing, with a focus on **speed, flexibility, and ease of use**

## Philosophy

AudioMDB is designed for **high-throughput machine learning training** with large-scale audio datasets. Here's what makes it fast:

### Lightning Memory-Mapped Database (LMDB)
- **Zero-copy reads**: Data is memory-mapped directly from disk, eliminating buffer copies
- **Concurrent access**: Multiple training processes can read simultaneously without locks
- **Optimized storage**: LMDB's B-tree structure provides O(log n) random access
- **Atomic transactions**: Guarantees data consistency even during concurrent operations

### Intelligent Shard Management
- **Predictive prefetching**: Background threads pre-download shards before they're needed
- **Smart caching**: Configurable disk space limits with LRU eviction of completed shards  
- **Streaming architecture**: Process datasets larger than available RAM or storage
- **Parallel I/O**: Multiple worker threads handle shard downloads while training continues


The more exciting parts of this project are the **base classes** that you can subclass to create your own custom data pipelines. The included implementations (HuggingFace, Whisper, S3, etc.) are just **pre-built examples** to explain what you could do with thi. The real power comes from subclassing the base components for your specific infrastructure.
### Examples
```python
# Set maximum cache size - shards are auto-evicted when exceeded
retriever = S3Retriever(
    bucket="my-dataset", 
    max_cache_bytes=50 * 1024**3,  # 50GB disk budget
    prefetch=50,                    # Download 50 shards ahead
    workers=4                      # 4 concurrent download threads
)

# Training loop never blocks on I/O
for batch in dataloader:
    # Shard already in local cache, zero latency
    train_step(batch)
    # Meanwhile: next shards downloading in background if not already cached by retriever due to prefetch
```

### Performance Features
- **Sharded datasets**: Parallel processing during conversion and loading
- **Compression-aware**: Handles compressed audio efficiently without decompression overhead
- **Memory efficiency**: Samples loaded on-demand, not kept in RAM
- **Network optimization**: Resumable downloads, connection pooling, retry logic
- **Cache locality**: Hot shards stay in cache, cold shards evicted automatically

### Multi-Dataset Training
- **CombinedDataset**: Seamlessly combine multiple datasets with different shuffling strategies
  - `shuffle='pseudo'`: Random sampling across datasets for balanced training
  - `shuffle='ordered'`: Round-robin sampling maintains dataset proportions  
  - `shuffle=None`: Sequential iteration through all datasets
- **Independent caching**: Each dataset manages its own shard cache and prefetching
- **Mixed sources**: Combine local, cloud, and streaming datasets in one training loop

```python
# Train on multiple datasets simultaneously
local_dataset = StreamingDataset(LocalRetriever("./speech_data"))
cloud_dataset = StreamingDataset(S3Retriever("bucket", "music_data"))
hf_dataset = StreamingDataset(HFRetriever("mozilla/common_voice"))

# Pseudo-random sampling across all datasets
combined = CombinedDataset([local_dataset, cloud_dataset, hf_dataset], shuffle='pseudo')
dataloader = DataLoader(combined, batch_size=32)

# Training sees samples from all datasets mixed together
for batch in dataloader:
    train_step(batch)  # Batch may contain samples from any dataset
```

### Extensible Architecture
The included implementations (HuggingFace, Whisper, S3, etc.) are just **pre-built examples**. The real power comes from subclassing the base components for your specific infrastructure:

- **BaseConverter**: Handle proprietary data sources, custom formats, streaming APIs
- **BaseProcessor**: Domain-specific feature extraction, augmentation pipelines  
- **BaseRetriever**: Custom storage backends, caching strategies, networking protocols
- **BaseUploader**: Deploy to internal systems, specialized cloud services

Every component is designed to be extended and customized. There may be a few bugs still in this as the project already solves most problems i encounter for my use case so i may not make any big changes for a while. Please feel free to open an issue if you find any bugs or have suggestions for improvements, or do reach out as well!


## Quick Start

### Installation

```bash
pip install audiomdb
```

### Using Pre-built Components

```python
from audiomdb.converters.hf import HFConverter
from audiomdb.processors.whisper import WhisperProcessor
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# Using pre-built HuggingFace Converter and Whisper feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
processor = WhisperProcessor(feature_extractor, tokenizer)

converter = HFConverter(
    data_id="mozilla-foundation/common_voice_17_0",
    output_dir="./cv_lmdb",
    data_split="train",
    processor=processor,
    samples_per_shard=10000
)

converter.run()
```

## Creating Custom Components

### Custom Processor

Subclass `BaseProcessor` to create your own audio/text processing pipeline:

```python
from audiomdb.processors.base import BaseProcessor
import torch
import torchaudio
import numpy as np

class MelSpectrogramProcessor(BaseProcessor):
    def __init__(self, n_mels=80, keep_original=False):
        super().__init__(keep_original)
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=n_mels
        )
    
    def process(self, sample, **kwargs):
        sample_rate = kwargs.get('sample_rate', 16000)
        result = sample.copy() if self.keep_original else {}
        
        if 'audio' in sample:
            audio_data = sample['audio']
            if isinstance(audio_data, bytes):
                audio_array = self.load_array(audio_data, sample_rate)
            else:
                audio_array = audio_data
            
            # Convert to mel spectrogram
            audio_tensor = torch.from_numpy(audio_array)
            mel_spec = self.mel_transform(audio_tensor)
            
            result['mel_spectrogram'] = mel_spec.numpy()
            result['mel_shape'] = mel_spec.shape
            result['sample_rate'] = sample_rate
        
        return result

class CustomNLPProcessor(BaseProcessor):
    def __init__(self, model_name="bert-base-uncased", keep_original=False):
        super().__init__(keep_original)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def process(self, sample, **kwargs):
        result = sample.copy() if self.keep_original else {}
        
        if 'text' in sample:
            text = sample['text']
            text = text.lower().strip()
            
            tokens = self.tokenizer(
                text, 
                padding='max_length', 
                max_length=256, 
                truncation=True,
                return_tensors="np"
            )
            
            result['input_ids'] = tokens.input_ids[0]
            result['attention_mask'] = tokens.attention_mask[0]
            result['text_length'] = len(text)
        
        return result
```

### Custom Converter

Subclass `BaseConverter` to handle new data sources:

```python
from audiomdb.converters.base import BaseConverter
import pandas as pd
import requests
import os

class APIConverter(BaseConverter):
    """Convert data from API to LMDB AudioRL format"""
    
    def __init__(self, api_endpoint,  **kwargs):
        super().__init__(**kwargs)
        self.api_endpoint = api_endpoint
        self.auth_token = os.environ['OSINT_API_TOKEN']
        self.dataset_name = f"api_data_{api_endpoint.split('/')[-1]}"
    
    def sample_iterator(self):
        """Fetch data from API and yield samples"""
        headers = {'Authorization': f'Bearer {self.auth_token}'}
        response = requests.get(self.api_endpoint, headers=headers)
        data = response.json()
        
        for idx, item in enumerate(data['samples']):
            audio_response = requests.get(item['audio_url'])
            audio_bytes = audio_response.content
            
            yield f"sample_{idx:08d}", {
                'audio': audio_bytes,
                'text': item['transcript'],
                'sample_rate': item.get('sample_rate', 16000),
                'speaker_id': item.get('speaker_id'),
                'duration': item.get('duration')
            }
    
    @property
    def converter_name(self) -> str:
        return 'api'

class DatabaseConverter(BaseConverter):
    """Convert data from a database to LMDB"""
    
    def __init__(self, db_connection_string, query, **kwargs):
        super().__init__(**kwargs)
        self.db_connection_string = db_connection_string
        self.query = query
        self.dataset_name = "database_audio"
    
    def sample_iterator(self):
        import sqlalchemy as sa
        
        engine = sa.create_engine(self.db_connection_string)
        with engine.connect() as conn:
            result = conn.execute(sa.text(self.query))
            
            for idx, row in enumerate(result):
                with open(row.audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                yield f"sample_{idx:08d}", {
                    'audio': audio_bytes,
                    'text': row.transcript,
                    'sample_rate': 16000,
                    'metadata': {
                        'speaker': row.speaker,
                        'emotion': row.emotion,
                        'quality_score': row.quality_score
                    }
                }
    
    @property 
    def converter_name(self) -> str:
        return 'database'
```

### Custom Retriever

Subclass `BaseRetriever` to load data from custom storage:

```python
from audiomdb.retrievers.base import BaseRetriever
import boto3
import json
import os

class CustomS3Retriever(BaseRetriever):
    """Retriever for s3 storage"""
    
    def __init__(self, bucket, prefix, endpoint_url=None, **kwargs):
        self.bucket = bucket  
        self.prefix = prefix
        self.s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        super().__init__(**kwargs)
    
    def download_metadata(self):
        """Download and return path to metadata.json"""
        metadata_key = f"{self.prefix}/metadata.json"
        local_path = os.path.join(self.cache_dir, "metadata.json")
        
        self.s3_client.download_file(self.bucket, metadata_key, local_path)
        return local_path
    
    def download_shard(self, shard_id):
        """Download a specific shard directory"""
        shard_name = f"shard_{shard_id:05d}"
        shard_dir = os.path.join(self.cache_dir, shard_name)
        os.makedirs(shard_dir, exist_ok=True)
        
        # Download LMDB files
        for file_name in ['data.mdb', 'lock.mdb']:
            s3_key = f"{self.prefix}/{shard_name}/{file_name}"
            local_path = os.path.join(shard_dir, file_name)
            self.s3_client.download_file(self.bucket, s3_key, local_path)
        
        return shard_dir

class RedisRetriever(BaseRetriever):
    """Redis Cache Retriever, Redis is already fast, You may want to check if AudioRL may be worth it"""
    
    def __init__(self, redis_host, redis_port=6379, **kwargs):
        import redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        super().__init__(**kwargs)
    
    def download_metadata(self):
        metadata_json = self.redis_client.get("audiomdb:metadata")
        metadata = json.loads(metadata_json)
        
        local_path = os.path.join(self.cache_dir, "metadata.json") 
        with open(local_path, 'w') as f:
            json.dump(metadata, f)
        return local_path
    
    def download_shard(self, shard_id):
        shard_data = self.redis_client.get(f"audiomdb:shard:{shard_id}")
        shard_dir = os.path.join(self.cache_dir, f"shard_{shard_id:05d}")
        return shard_dir
```

### Custom Uploader

Subclass `BaseUploader` to deploy to custom storage:

```python
from audiomdb.uploaders.base import BaseUploader
import paramiko
import ftplib

class SFTPUploader(BaseUploader):
    """Upload to SFTP server"""
    
    def __init__(self, hostname, username, password, remote_path, **kwargs):
        super().__init__(**kwargs)
        self.hostname = hostname
        self.username = username  
        self.password = password
        self.remote_path = remote_path
    
    def upload_dir(self, local_dir):
        transport = paramiko.Transport((self.hostname, 22))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        try:
            sftp.mkdir(self.remote_path)
        except:
            pass  # Directory might exist
        
        for full_path, rel_path in self.iter_files(local_dir):
            remote_file = f"{self.remote_path}/{rel_path}"
            remote_dir = "/".join(remote_file.split("/")[:-1])
            try:
                sftp.mkdir(remote_dir) 
            except:
                pass
            
            sftp.put(full_path, remote_file)
            print(f"Uploaded {rel_path}")
        
        sftp.close()
        transport.close()

class WebDAVUploader(BaseUploader):
    """Upload to WebDAV server"""
    
    def __init__(self, webdav_url, username, password, **kwargs):
        super().__init__(**kwargs)
        self.webdav_url = webdav_url
        self.username = username
        self.password = password
    
    def upload_dir(self, local_dir):
        from webdav3.client import Client
        
        options = {
            'webdav_hostname': self.webdav_url,
            'webdav_login': self.username,
            'webdav_password': self.password
        }
        client = Client(options)
        
        for full_path, rel_path in self.iter_files(local_dir):
            client.upload_sync(remote_path=rel_path, local_path=full_path)
            print(f"Uploaded {rel_path}")
```

## Loading Data for Training

```python
from audiomdb.data.base import StreamingDataset
from torch.utils.data import DataLoader

# Use any retriever (built-in or custom)
retriever = CustomS3Retriever("my-bucket", "datasets/speech")
dataset = StreamingDataset(retriever, num_threads=4)

dataloader = DataLoader(dataset, batch_size=32, num_workers=2)

for batch in dataloader:
    # Your training code
    features = batch['mel_spectrogram']  # From custom processor
    labels = batch['input_ids']
    # loss, back propagation, optimizer and scheduler step, etc
```

## Performance Tips

- **Sharding**: Use 10K-50K samples per shard for optimal I/O performance
- **Memory**: Set appropriate `map_size` for your dataset size 
- **Streaming**: Enable streaming for datasets larger than available RAM
- **Workers**: Use multiple workers for CPU-intensive processing
- **Caching**: Implement smart caching in custom retrievers for remote data, check out the cache manager
- **Prefetching**: Use prefetching to download shards ahead of time depending on how much disk space you have. 

## Potential Use-cases

The library is designed to handle diverse use cases:

- **Research Labs**: Custom converters for proprietary audio formats
- **Production Systems**: Custom retrievers with advanced caching strategies  
- **Edge Deployment**: Lightweight processors optimized for mobile devices
- **Multi-Modal**: Processors that handle audio, text, and visual data together

## Probelsm
- **Slow Writes**: In some environments, LMDB writes can be slower than expected. Consider using SSDs or optimizing write batch sizes.

## Contributing

Please help with new base class implementations, optimizations, and examples if you can. The modular design makes it easy to add support for new data sources and processing pipelines.

You can also reach out to me on [Twitter](https://x.com/Jenrola_odun) or [LinkedIn](https://www.linkedin.com/in/odunola499/) for discussions, suggestions, or contributions.

## License

MIT License - see LICENSE file for details.

---

**AudioMDB** - Extensible, high-performance audio ML data pipeline