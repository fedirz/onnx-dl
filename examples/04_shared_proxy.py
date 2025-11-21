"""
Shared proxy example for InferenceSessionProxy.

This example demonstrates sharing a single proxy instance across multiple
pipeline or model instances. This is useful for:
- Reducing memory usage (single model instance in memory)
- Coordinated lifecycle management
- Resource pooling

Run with: uv run python examples/04_shared_proxy.py
"""

import logging
from pathlib import Path

import numpy as np

from onnx_dl import InferenceSessionProxy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/"
    "snapshots/f0c48c298fd835726c27956a5d617bad7115627e/"
    "voxceleb_resnet34_LM.onnx"
).expanduser()


class AudioProcessor:
    """
    Example audio processing class that uses an embedding model.
    """

    def __init__(self, session: InferenceSessionProxy, name: str, process_type: str) -> None:
        """
        Initialize the processor.

        Args:
            session: Shared InferenceSessionProxy instance
            name: Processor name for logging
            process_type: Type of processing performed
        """
        self.session = session
        self.name = name
        self.process_type = process_type
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    def process(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Process audio features.

        Args:
            audio_features: Input features

        Returns:
            Processed output
        """
        print(f"[{self.name}] Processing with {self.process_type}...")
        result = self.session.run([self._output_name], {self._input_name: audio_features})
        return result[0]


def demo_shared_across_instances() -> None:
    """
    Share a single proxy across multiple processor instances.

    This ensures only one model is loaded in memory even though
    multiple processors use it.
    """
    print("\n=== Demo 1: Shared Proxy Across Multiple Instances ===")

    # Create a single shared proxy
    shared_proxy = InferenceSessionProxy(MODEL_PATH, ttl=5)
    print("Created shared proxy with TTL=5 seconds")

    # Create multiple processors using the same proxy
    processor1 = AudioProcessor(shared_proxy, "Processor-1", "speaker-verification")
    processor2 = AudioProcessor(shared_proxy, "Processor-2", "speaker-identification")
    processor3 = AudioProcessor(shared_proxy, "Processor-3", "speaker-clustering")

    print("Created 3 processors sharing the same proxy")
    print(f"Model loaded: {shared_proxy.is_loaded}")

    # Use the processors
    rng = np.random.default_rng(42)
    audio_features = rng.standard_normal((1, 200, 80), dtype=np.float32)

    # All processors use the same underlying model session
    result1 = processor1.process(audio_features)
    print(f"Result 1 shape: {result1.shape}, Model loaded: {shared_proxy.is_loaded}")

    result2 = processor2.process(audio_features)
    print(f"Result 2 shape: {result2.shape}, Model loaded: {shared_proxy.is_loaded}")

    result3 = processor3.process(audio_features)
    print(f"Result 3 shape: {result3.shape}, Model loaded: {shared_proxy.is_loaded}")

    print("\nAll processors share the same model instance in memory")
    print("TTL timer manages lifecycle based on any processor's usage")


def demo_independent_vs_shared() -> None:
    """
    Compare independent proxies vs shared proxy.

    Shows memory efficiency of sharing.
    """
    print("\n=== Demo 2: Independent vs Shared Proxies ===")

    print("\n--- Independent Proxies (3 separate models in memory) ---")
    # Each processor gets its own proxy
    proxy1 = InferenceSessionProxy(MODEL_PATH, ttl=-1)
    proxy2 = InferenceSessionProxy(MODEL_PATH, ttl=-1)
    proxy3 = InferenceSessionProxy(MODEL_PATH, ttl=-1)

    processor1 = AudioProcessor(proxy1, "Independent-1", "type-A")
    processor2 = AudioProcessor(proxy2, "Independent-2", "type-B")
    processor3 = AudioProcessor(proxy3, "Independent-3", "type-C")

    print(f"Processor 1 - Model loaded: {proxy1.is_loaded}")
    print(f"Processor 2 - Model loaded: {proxy2.is_loaded}")
    print(f"Processor 3 - Model loaded: {proxy3.is_loaded}")
    print("Result: 3 separate model instances (if all used)")

    # Cleanup
    proxy1.unload()
    proxy2.unload()
    proxy3.unload()

    print("\n--- Shared Proxy (1 model in memory) ---")
    # All processors share one proxy
    shared_proxy = InferenceSessionProxy(MODEL_PATH, ttl=-1)

    processor1 = AudioProcessor(shared_proxy, "Shared-1", "type-A")
    processor2 = AudioProcessor(shared_proxy, "Shared-2", "type-B")
    processor3 = AudioProcessor(shared_proxy, "Shared-3", "type-C")

    rng = np.random.default_rng(42)
    audio_features = rng.standard_normal((1, 200, 80), dtype=np.float32)

    _ = processor1.process(audio_features)
    _ = processor2.process(audio_features)
    _ = processor3.process(audio_features)

    print(f"All processors use same proxy - Model loaded: {shared_proxy.is_loaded}")
    print("Result: 1 shared model instance in memory")

    shared_proxy.unload()


def demo_coordinated_lifecycle() -> None:
    """
    Demonstrate coordinated lifecycle with context manager.

    When using a shared proxy, the context manager keeps the model
    loaded for all instances.
    """
    print("\n=== Demo 3: Coordinated Lifecycle Management ===")

    shared_proxy = InferenceSessionProxy(MODEL_PATH, ttl=2)

    processor1 = AudioProcessor(shared_proxy, "Pipeline-A", "real-time")
    processor2 = AudioProcessor(shared_proxy, "Pipeline-B", "batch")

    print(f"Before context: Model loaded = {shared_proxy.is_loaded}")

    # Use context manager for batch processing
    with shared_proxy.session():
        print(f"In context: Model loaded = {shared_proxy.is_loaded}")

        # Both processors can use the model without worrying about unloading
        rng = np.random.default_rng(42)
        for _ in range(3):
            audio_features = rng.standard_normal((1, 200, 80), dtype=np.float32)

            _ = processor1.process(audio_features)
            _ = processor2.process(audio_features)

        print(f"Still in context: Model loaded = {shared_proxy.is_loaded}")

    print(f"After context: Model loaded = {shared_proxy.is_loaded}")
    print("TTL timer now active - model will unload after 2 seconds of inactivity")


def demo_mixed_ttl_strategies() -> None:
    """
    Use different TTL strategies for different shared proxies.

    Shows how to optimize memory usage for different model usage patterns.
    """
    print("\n=== Demo 4: Mixed TTL Strategies ===")

    # Frequently used model - keep in memory
    frequent_proxy = InferenceSessionProxy(MODEL_PATH, ttl=-1)
    print("Frequent model: TTL=-1 (persistent)")

    # Occasionally used model - unload after inactivity
    occasional_proxy = InferenceSessionProxy(MODEL_PATH, ttl=10)
    print("Occasional model: TTL=10s (auto-unload)")

    # Rarely used model - unload immediately
    rare_proxy = InferenceSessionProxy(MODEL_PATH, ttl=0)
    print("Rare model: TTL=0 (immediate unload)")

    # Multiple processors can share each proxy
    _ = AudioProcessor(frequent_proxy, "Freq-1", "real-time")
    _ = AudioProcessor(frequent_proxy, "Freq-2", "streaming")

    _ = AudioProcessor(occasional_proxy, "Occ-1", "periodic")

    _ = AudioProcessor(rare_proxy, "Rare-1", "one-off")

    print(f"\nFrequent processors share proxy with TTL=-1: {frequent_proxy.is_loaded}")
    print(f"Occasional processor uses proxy with TTL=10s: {occasional_proxy.is_loaded}")
    print(f"Rare processor uses proxy with TTL=0: {rare_proxy.is_loaded}")

    print("\nEach shared proxy manages its own lifecycle independently")

    # Cleanup frequent proxy
    frequent_proxy.unload()


def main() -> None:
    """Run all demonstrations."""
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please download the model first or update the MODEL_PATH")
        return

    demo_shared_across_instances()
    demo_independent_vs_shared()
    demo_coordinated_lifecycle()
    demo_mixed_ttl_strategies()

    print("\n=== All demos completed ===")


if __name__ == "__main__":
    main()
