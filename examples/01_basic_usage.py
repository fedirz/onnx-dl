"""
Basic usage example of InferenceSessionProxy.

This example demonstrates the three TTL modes:
- TTL = -1: Model stays loaded indefinitely (persistent mode)
- TTL = 0: Model unloads immediately after each use
- TTL > 0: Model unloads after N seconds of inactivity

Run with: uv run python examples/01_basic_usage.py
"""

import logging
from pathlib import Path
import time

import numpy as np

from onnx_dl import InferenceSessionProxy

# Configure logging to see the proxy's lifecycle events
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Use a real ONNX model for demonstration
MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/"
    "snapshots/f0c48c298fd835726c27956a5d617bad7115627e/"
    "voxceleb_resnet34_LM.onnx"
).expanduser()


def demo_persistent_mode() -> None:
    """
    TTL = -1: Model stays loaded indefinitely.

    Best for frequently used models that should remain in memory.
    """
    print("\n=== Demo 1: Persistent Mode (TTL = -1) ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=-1)

    print("Before first use: is_loaded =", proxy.is_loaded)

    # First access - model loads
    _ = proxy.get_inputs()
    print(f"After first use: is_loaded = {proxy.is_loaded}")

    # Wait a bit and check again
    time.sleep(2)
    print(f"After 2 seconds: is_loaded = {proxy.is_loaded} (still loaded)")

    # Multiple accesses - model stays loaded
    for i in range(3):
        _ = proxy.get_outputs()
        print(f"Access {i + 1}: is_loaded = {proxy.is_loaded}")

    # Manual cleanup when done
    proxy.unload()
    print(f"After manual unload: is_loaded = {proxy.is_loaded}")


def demo_immediate_unload() -> None:
    """
    TTL = 0: Model unloads immediately after each use.

    Best for rarely used models to minimize memory footprint.
    """
    print("\n=== Demo 2: Immediate Unload Mode (TTL = 0) ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=0)

    print("Before first use: is_loaded =", proxy.is_loaded)

    # Access model - loads and immediately unloads
    _ = proxy.get_inputs()
    time.sleep(0.2)  # Give it a moment to unload
    print(f"After use: is_loaded = {proxy.is_loaded} (unloaded immediately)")

    # Each access loads and unloads
    for i in range(3):
        _ = proxy.get_outputs()
        time.sleep(0.2)
        print(f"Access {i + 1}: is_loaded = {proxy.is_loaded} (unloaded)")


def demo_timed_unload() -> None:
    """
    TTL > 0: Model unloads after N seconds of inactivity.

    Best for models with bursty usage patterns - stays loaded during active
    periods but unloads when idle.
    """
    print("\n=== Demo 3: Timed Unload Mode (TTL = 3 seconds) ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=3)

    print("Before first use: is_loaded =", proxy.is_loaded)

    # First access
    _ = proxy.get_inputs()
    print(f"After first use: is_loaded = {proxy.is_loaded}")

    # Access within TTL window - timer resets
    time.sleep(2)
    print(f"After 2 seconds: is_loaded = {proxy.is_loaded} (still loaded)")

    _ = proxy.get_outputs()
    print("Accessed again - TTL timer reset")

    # Wait for TTL to expire
    print("Waiting for TTL to expire...")
    time.sleep(3.5)
    print(f"After TTL expired: is_loaded = {proxy.is_loaded} (unloaded)")


def demo_inference() -> None:
    """
    Demonstrate actual inference with the proxy.

    The proxy works exactly like InferenceSession for running models.
    """
    print("\n=== Demo 4: Running Inference ===")
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=-1)

    # Get input/output metadata
    inputs = proxy.get_inputs()
    outputs = proxy.get_outputs()

    input_name = inputs[0].name
    output_name = outputs[0].name
    input_shape = inputs[0].shape

    print(f"Model input: {input_name}, shape: {input_shape}")
    print(f"Model output: {output_name}")

    # Create dummy input
    rng = np.random.default_rng(42)
    dummy_input = rng.standard_normal((1, 200, int(input_shape[2])), dtype=np.float32)

    # Run inference
    print("Running inference...")
    result = proxy.run([output_name], {input_name: dummy_input})
    print(f"Result shape: {result[0].shape}")

    proxy.unload()


def main() -> None:
    """Run all demonstrations."""
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please download the model first or update the MODEL_PATH")
        return

    demo_persistent_mode()
    demo_immediate_unload()
    demo_timed_unload()
    demo_inference()

    print("\n=== All demos completed ===")


if __name__ == "__main__":
    main()
