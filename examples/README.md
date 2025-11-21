# InferenceSessionProxy Examples

This directory contains examples demonstrating various usage patterns of `InferenceSessionProxy`.

## Prerequisites

Make sure you have the test model downloaded:

```bash
# The examples use this model:
~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/snapshots/f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx
```

## Running Examples

```bash
# From the package root
uv run python examples/01_basic_usage.py
uv run python examples/02_context_manager.py
uv run python examples/03_drop_in_replacement.py
uv run python examples/04_shared_proxy.py
```

## Example Descriptions

### 01_basic_usage.py

Demonstrates the three TTL (time-to-live) modes:

- **TTL = -1**: Persistent mode - model stays loaded indefinitely
- **TTL = 0**: Immediate unload - model unloads after each use
- **TTL > 0**: Timed unload - model unloads after N seconds of inactivity

Also shows:
- Lazy loading behavior
- Manual unloading with `unload()`
- Running actual inference

**Key takeaway**: Choose TTL based on your usage pattern to optimize memory usage.

### 02_context_manager.py

Demonstrates the context manager API for explicit lifecycle control:

- Batch processing - keep model loaded during batch
- Nested context managers with reference counting
- Mixed usage (context manager + direct calls)

**Key takeaway**: Use context managers when you want to ensure the model stays loaded for a specific code block, regardless of TTL.

### 03_drop_in_replacement.py

Shows how `InferenceSessionProxy` can transparently replace `onnxruntime.InferenceSession`:

- Model classes work identically with both
- Same interface, enhanced lifecycle management
- Multiple models with different policies

**Key takeaway**: `InferenceSessionProxy` is a drop-in replacement - no code changes needed in your model classes.

### 04_shared_proxy.py

Demonstrates sharing a single proxy across multiple instances:

- Memory efficiency - one model in memory, multiple users
- Coordinated lifecycle management
- Independent vs shared comparison

**Key takeaway**: Share proxies to reduce memory usage when multiple components need the same model.

## Common Patterns

### Pattern 1: Frequently Used Model
```python
proxy = InferenceSessionProxy(model_path, ttl=-1)
# Model stays loaded, manual cleanup when done
```

### Pattern 2: Occasionally Used Model
```python
proxy = InferenceSessionProxy(model_path, ttl=60)
# Stays loaded during bursts, auto-unloads when idle
```

### Pattern 3: Rarely Used Model
```python
proxy = InferenceSessionProxy(model_path, ttl=0)
# Minimal memory footprint, loads on demand
```

### Pattern 4: Batch Processing
```python
with proxy.session() as sess:
    for item in batch:
        result = sess.run([output], {input: item})
# Auto-cleanup after batch
```

## Logging

All examples are configured with logging to show lifecycle events. You'll see:

- **INFO**: Model loading/unloading
- **DEBUG**: TTL expiration, context manager ref counts

Adjust logging level in each example to see more or less detail.
