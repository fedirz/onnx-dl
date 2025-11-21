"""
Context manager usage example for InferenceSessionProxy.

The context manager provides explicit control over model lifecycle,
keeping the model loaded for the entire duration of the with block.

Run with: uv run python examples/02_context_manager.py
"""

import logging
from pathlib import Path
import time

import numpy as np

from onnx_dl import InferenceSessionProxy

# Configure logging to see lifecycle events
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/"
    "snapshots/f0c48c298fd835726c27956a5d617bad7115627e/"
    "voxceleb_resnet34_LM.onnx"
).expanduser()


def demo_batch_processing() -> None:
    """
    Process a batch of data while keeping the model loaded.

    This is ideal for processing multiple items where you want to ensure
    the model stays in memory for the entire batch.
    """
    print("\n=== Demo 1: Batch Processing with Context Manager ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=2)

    # Create batch of dummy data
    rng = np.random.default_rng(42)
    batch_size = 5

    print(f"Processing batch of {batch_size} items...")
    print(f"Before context: is_loaded = {proxy.is_loaded}")

    # Use context manager to keep model loaded during batch
    with proxy.session() as sess:
        print(f"Inside context: is_loaded = {proxy.is_loaded}")

        # Get input/output info
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        input_shape = sess.get_inputs()[0].shape

        # Process each item in the batch
        for i in range(batch_size):
            dummy_input = rng.standard_normal((1, 200, int(input_shape[2])), dtype=np.float32)
            result = sess.run([output_name], {input_name: dummy_input})
            print(f"  Processed item {i + 1}/{batch_size}, result shape: {result[0].shape}")  # pyright: ignore[reportAttributeAccessIssue]

        # Simulate some processing time
        time.sleep(0.5)
        print(f"Still in context: is_loaded = {proxy.is_loaded}")

    print(f"After context: is_loaded = {proxy.is_loaded} (TTL timer starts now)")

    # Wait for TTL to expire
    print("Waiting for TTL to expire...")
    time.sleep(2.5)
    print(f"After TTL: is_loaded = {proxy.is_loaded}")


def demo_nested_contexts() -> None:
    """
    Demonstrate nested context managers with reference counting.

    The model stays loaded until all contexts exit.
    """
    print("\n=== Demo 2: Nested Context Managers ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=1)

    print(f"Before outer context: is_loaded = {proxy.is_loaded}")

    with proxy.session() as sess1:
        print(f"In outer context: is_loaded = {proxy.is_loaded}")

        # Nested context
        with proxy.session() as sess2:
            print(f"In nested context: is_loaded = {proxy.is_loaded}")

            # Both reference the same session
            print(f"Same session object: {sess1 is sess2}")

            # Wait longer than TTL
            time.sleep(1.5)
            print(f"After {1.5}s (longer than TTL) but still in context: is_loaded = {proxy.is_loaded}")

        print(f"After inner context exits: is_loaded = {proxy.is_loaded}")

    print(f"After outer context exits: is_loaded = {proxy.is_loaded} (TTL timer starts)")

    # Wait for TTL
    time.sleep(1.5)
    print(f"After TTL expires: is_loaded = {proxy.is_loaded}")


def demo_mixed_usage() -> None:
    """
    Mix context manager usage with direct method calls.

    Shows that you can use both approaches with the same proxy.
    """
    print("\n=== Demo 3: Mixed Usage Patterns ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=2)

    # Direct method call
    print("Using direct method call...")
    _ = proxy.get_inputs()
    print(f"After direct call: is_loaded = {proxy.is_loaded}")

    time.sleep(1)

    # Use context manager
    print("\nUsing context manager...")
    with proxy.session() as sess:
        _ = sess.get_outputs()
        print(f"In context: is_loaded = {proxy.is_loaded}")

    # Another direct call
    print("\nUsing direct method call again...")
    _ = proxy.get_providers()
    print(f"After direct call: is_loaded = {proxy.is_loaded}")

    # Manual unload
    proxy.unload()
    print(f"After manual unload: is_loaded = {proxy.is_loaded}")


def main() -> None:
    """Run all demonstrations."""
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please download the model first or update the MODEL_PATH")
        return

    demo_batch_processing()
    demo_nested_contexts()
    demo_mixed_usage()

    print("\n=== All demos completed ===")


if __name__ == "__main__":
    main()
