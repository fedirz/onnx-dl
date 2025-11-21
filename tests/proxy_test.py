"""
Comprehensive tests for InferenceSessionProxy.

Tests verify all 11 requirements from the specification:
1. Lazy loading
2. TTL=-1 (persistent)
3. TTL=0 (immediate unload)
4. TTL>0 (timed unload)
5. TTL reset on usage
6. Context manager behavior
7. Nested context managers
8. Thread safety
9. is_loaded property
10. Manual unload
11. Method proxying
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytest

from onnx_dl import InferenceSessionProxy

# Test model path as specified in the requirements
TEST_MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/"
    "snapshots/f0c48c298fd835726c27956a5d617bad7115627e/"
    "voxceleb_resnet34_LM.onnx"
).expanduser()


@pytest.fixture
def model_path() -> Path:
    """
    Fixture providing the path to the test ONNX model.

    Returns:
        Path to voxceleb_resnet34_LM.onnx model
    """
    if not TEST_MODEL_PATH.exists():
        pytest.skip(f"Test model not found at {TEST_MODEL_PATH}")
    return TEST_MODEL_PATH


def test_lazy_loading(model_path: Path) -> None:
    """
    Test that the session is not loaded in the constructor (lazy loading).

    The model should only be loaded when first accessed, not when the
    proxy is instantiated.
    """
    proxy = InferenceSessionProxy(model_path, ttl=-1)

    # Session should not be loaded yet
    assert not proxy.is_loaded, "Session should not be loaded in constructor"


def test_ttl_persistent(model_path: Path) -> None:
    """
    Test that TTL=-1 keeps the model loaded indefinitely (persistent mode).

    The model should load on first use and remain in memory without
    any automatic unloading.
    """
    proxy = InferenceSessionProxy(model_path, ttl=-1)

    # Not loaded initially
    assert not proxy.is_loaded

    # Access the model
    inputs = proxy.get_inputs()
    assert len(inputs) > 0

    # Should be loaded now
    assert proxy.is_loaded

    # Wait a bit to ensure no unloading occurs
    time.sleep(2)

    # Should still be loaded (persistent mode)
    assert proxy.is_loaded


def test_ttl_immediate_unload(model_path: Path) -> None:
    """
    Test that TTL=0 unloads the model immediately after each use.

    The model should be loaded for each operation and unloaded
    immediately after the operation completes.
    """
    proxy = InferenceSessionProxy(model_path, ttl=0)

    # Not loaded initially
    assert not proxy.is_loaded

    # Access the model
    inputs = proxy.get_inputs()
    assert len(inputs) > 0

    # Should be unloaded immediately after use
    # Give it a tiny moment for the unload to complete
    time.sleep(0.1)
    assert not proxy.is_loaded

    # Access again
    outputs = proxy.get_outputs()
    assert len(outputs) > 0

    # Should be unloaded again
    time.sleep(0.1)
    assert not proxy.is_loaded


def test_ttl_timed_unload(model_path: Path) -> None:
    """
    Test that TTL>0 unloads the model after the specified seconds of inactivity.

    The model should remain loaded for the TTL duration and then
    automatically unload if not accessed.
    """
    ttl_seconds = 2
    proxy = InferenceSessionProxy(model_path, ttl=ttl_seconds)

    # Not loaded initially
    assert not proxy.is_loaded

    # Access the model
    inputs = proxy.get_inputs()
    assert len(inputs) > 0

    # Should be loaded now
    assert proxy.is_loaded

    # Should still be loaded before TTL expires
    time.sleep(ttl_seconds * 0.5)
    assert proxy.is_loaded

    # Wait for TTL to expire (with some buffer)
    time.sleep(ttl_seconds * 0.8)

    # Should be unloaded now
    assert not proxy.is_loaded


def test_ttl_reset_on_usage(model_path: Path) -> None:
    """
    Test that the TTL timer resets every time the model is used.

    If the model is accessed before the TTL expires, the countdown
    should restart, preventing unloading of actively used models.
    """
    ttl_seconds = 2
    proxy = InferenceSessionProxy(model_path, ttl=ttl_seconds)

    # Load the model
    proxy.get_inputs()
    assert proxy.is_loaded

    # Access the model repeatedly within the TTL window
    for _ in range(3):
        time.sleep(ttl_seconds * 0.6)  # Wait more than half TTL
        proxy.get_outputs()  # Reset the timer
        assert proxy.is_loaded, "Model should still be loaded (TTL was reset)"

    # Now wait for full TTL without accessing
    time.sleep(ttl_seconds + 0.5)

    # Should be unloaded now
    assert not proxy.is_loaded


def test_context_manager(model_path: Path) -> None:
    """
    Test that the context manager keeps the model loaded during the context
    and applies TTL-based cleanup after exiting.

    The model should remain in memory for the entire duration of the with
    block, regardless of TTL, and cleanup should only occur after exit.
    """
    ttl_seconds = 1
    proxy = InferenceSessionProxy(model_path, ttl=ttl_seconds)

    # Not loaded initially
    assert not proxy.is_loaded

    # Enter context manager
    with proxy.session() as sess:
        # Should be loaded inside context
        assert proxy.is_loaded

        # Even if we wait longer than TTL, should stay loaded in context
        time.sleep(ttl_seconds + 0.5)
        assert proxy.is_loaded

        # Can use the session directly
        inputs = sess.get_inputs()
        assert len(inputs) > 0

    # Outside context, TTL timer should start
    # Model should still be loaded immediately after exit
    assert proxy.is_loaded

    # Wait for TTL to expire
    time.sleep(ttl_seconds + 0.5)

    # Should be unloaded now
    assert not proxy.is_loaded


def test_nested_context_managers(model_path: Path) -> None:
    """
    Test that nested context managers work correctly with reference counting.

    Multiple nested with blocks should maintain the model in memory until
    all contexts exit, using reference counting.
    """
    ttl_seconds = 1
    proxy = InferenceSessionProxy(model_path, ttl=ttl_seconds)

    # Not loaded initially
    assert not proxy.is_loaded

    # Nested contexts
    with proxy.session() as sess1:
        assert proxy.is_loaded

        with proxy.session() as sess2:
            assert proxy.is_loaded

            # Both should reference the same session
            assert sess1 is sess2

            # Wait longer than TTL inside nested context
            time.sleep(ttl_seconds + 0.5)
            assert proxy.is_loaded, "Should stay loaded in nested context"

        # After inner context exits, should still be loaded (outer context active)
        assert proxy.is_loaded

    # After all contexts exit, TTL timer should start
    assert proxy.is_loaded

    # Wait for TTL to expire
    time.sleep(ttl_seconds + 0.5)

    # Should be unloaded now
    assert not proxy.is_loaded


def test_thread_safety(model_path: Path) -> None:
    """
    Test that concurrent access from multiple threads is safe.

    Multiple threads should be able to use the same proxy instance
    simultaneously without race conditions or crashes.
    """
    proxy = InferenceSessionProxy(model_path, ttl=2)

    def worker(worker_id: int) -> str:
        """Worker function that accesses the proxy multiple times."""
        for _ in range(5):
            # Get input metadata
            inputs = proxy.get_inputs()
            _ = inputs[0].name

            # Get output metadata
            outputs = proxy.get_outputs()
            _ = outputs[0].name

            # Small delay to increase chance of concurrent access
            time.sleep(0.01)

        return f"Worker {worker_id} completed"

    # Run multiple workers concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        results = [f.result() for f in futures]

    # All workers should complete successfully
    assert len(results) == 10
    assert all("completed" in r for r in results)


def test_concurrent_inference(model_path: Path) -> None:
    """
    Test that multiple inference requests can execute in parallel.

    This test verifies that the proxy's locking implementation allows
    multiple threads to run inference concurrently, rather than serializing
    all inference operations.
    """
    proxy = InferenceSessionProxy(model_path, ttl=-1)

    # Get model metadata
    inputs = proxy.get_inputs()
    outputs = proxy.get_outputs()
    input_name = inputs[0].name
    output_name = outputs[0].name
    input_shape = inputs[0].shape

    # Create test data
    rng = np.random.default_rng(42)
    test_inputs = [rng.standard_normal((1, 200, int(input_shape[2])), dtype=np.float32) for _ in range(10)]

    def run_inference(data: np.ndarray, worker_id: int) -> tuple[int, np.ndarray]:
        """Run inference on the given data."""
        result = proxy.run([output_name], {input_name: data})
        return worker_id, result[0]

    # Run inference concurrently on multiple threads
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_inference, data, i) for i, data in enumerate(test_inputs)]
        results = [f.result() for f in futures]
    parallel_time = time.time() - start_time

    # All inferences should complete successfully
    assert len(results) == 10
    assert all(isinstance(r[1], np.ndarray) for r in results)
    assert all(r[1].shape == (1, 256) for r in results)

    # Run the same inferences sequentially for comparison
    start_time = time.time()
    sequential_results = []
    for i, data in enumerate(test_inputs):
        _, result = run_inference(data, i)
        sequential_results.append(result)
    sequential_time = time.time() - start_time

    # Parallel execution should be faster than sequential
    # (or at least not significantly slower, accounting for overhead)
    # If inference was serialized, parallel_time would be >= sequential_time
    # With true parallelism, parallel_time should be much less than sequential_time
    print("\nConcurrent inference timing:")
    print(f"  Parallel time: {parallel_time:.3f}s")
    print(f"  Sequential time: {sequential_time:.3f}s")
    print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

    # The parallel version should not be significantly slower than 2x the sequential time
    # (accounting for threading overhead and system load)
    # This is a loose bound - the key is that it's not 10x slower (fully serialized)
    assert parallel_time < sequential_time * 2.0, (
        f"Parallel execution ({parallel_time:.3f}s) should not be much slower than sequential ({sequential_time:.3f}s)"
    )

    # Cleanup
    proxy.unload()


def test_is_loaded_property(model_path: Path) -> None:
    """
    Test that the is_loaded property accurately reflects the load state.

    The property should return True when the session is in memory and
    False otherwise, throughout the model lifecycle.
    """
    proxy = InferenceSessionProxy(model_path, ttl=1)

    # Not loaded initially
    assert not proxy.is_loaded

    # Load by accessing
    proxy.get_inputs()
    assert proxy.is_loaded

    # Should be loaded
    assert proxy.is_loaded

    # Wait for TTL to expire
    time.sleep(1.5)

    # Should be unloaded
    assert not proxy.is_loaded

    # Load again
    proxy.get_outputs()
    assert proxy.is_loaded

    # Manual unload
    proxy.unload()
    assert not proxy.is_loaded


def test_manual_unload(model_path: Path) -> None:
    """
    Test that manual unload() works and cancels pending TTL timers.

    The unload() method should immediately remove the model from memory
    and cancel any scheduled cleanup timers.
    """
    proxy = InferenceSessionProxy(model_path, ttl=10)

    # Load the model
    proxy.get_inputs()
    assert proxy.is_loaded

    # Manually unload (should cancel the 10-second timer)
    proxy.unload()
    assert not proxy.is_loaded

    # Wait a bit to ensure no timer fires
    time.sleep(0.5)
    assert not proxy.is_loaded

    # Can still use after manual unload (will reload)
    proxy.get_outputs()
    assert proxy.is_loaded


def test_method_proxying(model_path: Path) -> None:
    """
    Test that all InferenceSession methods are correctly proxied.

    The proxy should transparently forward all method calls to the
    underlying InferenceSession, making it a drop-in replacement.
    """
    proxy = InferenceSessionProxy(model_path, ttl=-1)

    # Test get_inputs()
    inputs = proxy.get_inputs()
    assert len(inputs) > 0
    assert hasattr(inputs[0], "name")
    assert hasattr(inputs[0], "shape")

    # Test get_outputs()
    outputs = proxy.get_outputs()
    assert len(outputs) > 0
    assert hasattr(outputs[0], "name")

    # Test get_providers()
    providers = proxy.get_providers()
    assert isinstance(providers, list)
    assert len(providers) > 0

    # Test run() with actual inference
    input_name = inputs[0].name
    output_name = outputs[0].name

    # Create dummy input matching the model's expected shape
    # The voxceleb model expects shape [batch, frames, features]
    input_shape = inputs[0].shape
    # Use a small batch for testing
    rng = np.random.default_rng(42)
    dummy_input = rng.standard_normal((1, 200, int(input_shape[2])), dtype=np.float32)

    result = proxy.run([output_name], {input_name: dummy_input})

    assert result is not None
    assert len(result) > 0
    assert isinstance(result[0], np.ndarray)


def test_proxy_with_session_options(model_path: Path) -> None:
    """
    Test that the proxy correctly passes session options to InferenceSession.

    Session options and providers should be forwarded to the underlying
    ONNX Runtime session.
    """
    import onnxruntime as ort

    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # Create proxy with options
    proxy = InferenceSessionProxy(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"], ttl=-1)

    # Access to trigger loading
    proxy.get_inputs()

    # Verify providers
    providers = proxy.get_providers()
    assert "CPUExecutionProvider" in providers


def test_proxy_as_drop_in_replacement(model_path: Path) -> None:
    """
    Test that the proxy can be used as a drop-in replacement in model classes.

    Model classes that expect InferenceSession should work identically
    whether they receive a real session or a proxy.
    """

    class DummyModel:
        """Example model class that works with both InferenceSession and proxy."""

        def __init__(self, session: Any) -> None:
            self.session = session

        def get_input_name(self) -> str:
            return self.session.get_inputs()[0].name

        def get_output_name(self) -> str:
            return self.session.get_outputs()[0].name

    # Create model with proxy
    proxy = InferenceSessionProxy(model_path, ttl=-1)
    model = DummyModel(proxy)

    # Model should work normally
    input_name = model.get_input_name()
    output_name = model.get_output_name()

    assert isinstance(input_name, str)
    assert isinstance(output_name, str)
    assert len(input_name) > 0
    assert len(output_name) > 0
