# onnx-dl

**ONNX Dynamic Loading** - A transparent proxy wrapper for `onnxruntime.InferenceSession` that provides automatic model lifecycle management based on configurable time-to-live (TTL) policies.

The "dl" stands for **dynamic loading**, referring to the proxy's ability to automatically load and unload ONNX models from memory based on usage patterns, helping optimize memory usage in applications with multiple models or varying workload patterns.

## Features

- **Transparent API**: Drop-in replacement for `onnxruntime.InferenceSession` - no code changes needed in your model classes
- **Flexible TTL Policies**: Configure model lifecycle with three modes:
  - `ttl=-1`: Persistent (model stays loaded indefinitely)
  - `ttl=0`: Immediate unload (minimal memory footprint)
  - `ttl>0`: Timed unload (auto-cleanup after N seconds of inactivity)
- **Smart TTL Reset**: Automatically resets timer on each access, preventing unloading of actively used models
- **Context Manager Support**: Explicit lifecycle control for batch processing with `proxy.session()`
- **Reference Counting**: Properly handles nested context managers
- **Thread-Safe**: Safe concurrent access from multiple threads
- **Lazy Loading**: Models load on first use, not at construction time
- **Shared Proxies**: Multiple instances can share a single proxy for memory efficiency
- **Lifecycle Monitoring**: Built-in logging for tracking load/unload events

## Installation

### Using uv

```bash
uv add onnx-dl
```

### Using pip

```bash
pip install onnx-dl
```

## Usage

### Quick Start

```python
from onnx_dl import InferenceSessionProxy
import numpy as np

# Create a proxy with 60-second TTL
proxy = InferenceSessionProxy("model.onnx", ttl=60)

# Use exactly like onnxruntime.InferenceSession
inputs = proxy.get_inputs()
outputs = proxy.get_outputs()

# Run inference
result = proxy.run(
    [outputs[0].name],
    {inputs[0].name: your_data}
)

# Model automatically unloads after 60 seconds of inactivity
```

### Different TTL Strategies

```python
# Frequently used model - keep in memory
vad_proxy = InferenceSessionProxy("vad.onnx", ttl=-1)

# Occasionally used model - auto-unload when idle
seg_proxy = InferenceSessionProxy("segmentation.onnx", ttl=60)

# Rarely used model - unload immediately
emb_proxy = InferenceSessionProxy("embedding.onnx", ttl=0)
```

### Context Manager for Batch Processing

```python
proxy = InferenceSessionProxy("model.onnx", ttl=30)

# Keep model loaded for entire batch
with proxy.session() as sess:
    for data in batch:
        result = sess.run([output_name], {input_name: data})
        process(result)
# TTL timer starts after exiting context
```

### Drop-in Replacement

```python
class EmbeddingModel:
    def __init__(self, session):
        """Works with both InferenceSession and InferenceSessionProxy"""
        self.session = session

    def extract(self, segments):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: segments})
        return result[0]

# Use with proxy for automatic lifecycle management
proxy = InferenceSessionProxy("model.onnx", ttl=60)
model = EmbeddingModel(proxy)
```

### Shared Proxy Pattern

```python
# Share a single proxy across multiple instances for memory efficiency
shared_proxy = InferenceSessionProxy("embedding.onnx", ttl=300)

processor1 = AudioProcessor(shared_proxy)
processor2 = AudioProcessor(shared_proxy)
processor3 = AudioProcessor(shared_proxy)

# All processors use the same model instance in memory
```

### API Reference

#### Constructor

```python
InferenceSessionProxy(
    model_path: str | Path,
    sess_options: ort.SessionOptions | None = None,
    providers: list[str] | None = None,
    ttl: int = -1
)
```

**Parameters:**

- `model_path`: Path to the ONNX model file
- `sess_options`: ONNX Runtime session options (optional)
- `providers`: List of execution providers (optional)
- `ttl`: Time-to-live in seconds (-1 = persistent, 0 = immediate, >0 = seconds)

#### Methods

- `run(output_names, input_feed, run_options=None)` - Run inference
- `get_inputs()` - Get input metadata
- `get_outputs()` - Get output metadata
- `get_providers()` - Get execution providers
- `session()` - Context manager for explicit lifecycle control (use with `with` statement)
- `unload()` - Manually unload the model immediately
- `is_loaded` - Property to check if model is currently loaded

All other `onnxruntime.InferenceSession` methods are automatically proxied.

### More Examples

See the [`examples/`](./examples) directory for comprehensive examples:

- **01_basic_usage.py** - TTL modes and basic operations
- **02_context_manager.py** - Context manager patterns
- **03_drop_in_replacement.py** - Using as transparent replacement
- **04_shared_proxy.py** - Sharing proxies across instances

## Contributing

Bug reports, feature requests, and contributions are welcome! Please open an issue or pull request on the project repository.

## License

[MIT](./LICENSE)
