"""
Drop-in replacement example for InferenceSessionProxy.

This example shows how InferenceSessionProxy can be used as a transparent
replacement for onnxruntime.InferenceSession in existing code.

Run with: uv run python examples/03_drop_in_replacement.py
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from onnx_dl import InferenceSessionProxy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet34-LM/"
    "snapshots/f0c48c298fd835726c27956a5d617bad7115627e/"
    "voxceleb_resnet34_LM.onnx"
).expanduser()


class EmbeddingModel:
    """
    Example model class that works with both InferenceSession and InferenceSessionProxy.

    This class is completely agnostic to whether it receives a real
    InferenceSession or a proxy - the interface is identical.
    """

    def __init__(self, session: ort.InferenceSession | InferenceSessionProxy) -> None:
        """
        Initialize the model with a session.

        Args:
            session: Either an onnxruntime.InferenceSession or InferenceSessionProxy
        """
        self.session = session
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self._input_shape = self.session.get_inputs()[0].shape

    def extract_embedding(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from audio features.

        Args:
            audio_features: Input audio features

        Returns:
            Embedding vector
        """
        result = self.session.run([self._output_name], {self._input_name: audio_features})
        return result[0]  # pyright: ignore[reportReturnType]

    def get_input_shape(self) -> list[int | str]:
        """Get the expected input shape."""
        return self._input_shape


def demo_with_real_session() -> None:
    """
    Use the model class with a real InferenceSession.

    This is the traditional approach without lifecycle management.
    """
    print("\n=== Demo 1: Using Real InferenceSession ===")

    # Create a real ONNX Runtime session
    session = ort.InferenceSession(str(MODEL_PATH))
    model = EmbeddingModel(session)

    print(f"Model input shape: {model.get_input_shape()}")

    # Create dummy input
    rng = np.random.default_rng(42)
    audio_features = rng.standard_normal((1, 200, 80), dtype=np.float32)

    # Extract embedding
    print("Extracting embedding...")
    embedding = model.extract_embedding(audio_features)
    print(f"Embedding shape: {embedding.shape}")

    # Session stays loaded until explicitly deleted or program exits
    print("Session remains in memory (no automatic unloading)")


def demo_with_proxy_persistent() -> None:
    """
    Use the model class with InferenceSessionProxy in persistent mode.

    Identical usage to InferenceSession but with explicit lifecycle control.
    """
    print("\n=== Demo 2: Using InferenceSessionProxy (Persistent) ===")

    # Create proxy in persistent mode
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=-1)
    model = EmbeddingModel(proxy)

    print(f"Model input shape: {model.get_input_shape()}")

    # Create dummy input
    rng = np.random.default_rng(42)
    audio_features = rng.standard_normal((1, 200, 80), dtype=np.float32)

    # Extract embedding - works exactly the same
    print("Extracting embedding...")
    embedding = model.extract_embedding(audio_features)
    print(f"Embedding shape: {embedding.shape}")

    # Can manually unload when done
    print("Manually unloading...")
    proxy.unload()
    print("Session unloaded from memory")


def demo_with_proxy_auto_unload() -> None:
    """
    Use the model class with InferenceSessionProxy in auto-unload mode.

    Same interface, but model automatically unloads after inactivity.
    """
    print("\n=== Demo 3: Using InferenceSessionProxy (Auto-unload) ===")

    # Create proxy with 2-second TTL
    proxy = InferenceSessionProxy(MODEL_PATH, ttl=2)
    model = EmbeddingModel(proxy)

    print(f"Model input shape: {model.get_input_shape()}")

    # Create dummy input
    rng = np.random.default_rng(42)
    audio_features = rng.standard_normal((1, 200, 80), dtype=np.float32)

    # Extract embedding
    print("Extracting embedding...")
    embedding = model.extract_embedding(audio_features)
    print(f"Embedding shape: {embedding.shape}")

    # Model will automatically unload after 2 seconds of inactivity
    print("Model will auto-unload after 2 seconds of inactivity")
    print("(No manual cleanup needed)")


def demo_multiple_models() -> None:
    """
    Use multiple model instances with different lifecycle policies.

    Shows how you can have different policies for different models.
    """
    print("\n=== Demo 4: Multiple Models with Different Policies ===")

    # Frequently used model - keep in memory
    proxy1 = InferenceSessionProxy(MODEL_PATH, ttl=-1)
    model1 = EmbeddingModel(proxy1)
    print("Model 1: Persistent (TTL = -1)")

    # Occasionally used model - unload after 5 seconds
    proxy2 = InferenceSessionProxy(MODEL_PATH, ttl=5)
    model2 = EmbeddingModel(proxy2)
    print("Model 2: Auto-unload after 5 seconds (TTL = 5)")

    # Rarely used model - unload immediately
    proxy3 = InferenceSessionProxy(MODEL_PATH, ttl=0)
    model3 = EmbeddingModel(proxy3)
    print("Model 3: Immediate unload (TTL = 0)")

    # All models work identically despite different policies
    print("\nAll models have same interface and work identically")
    print(f"Model 1 input shape: {model1.get_input_shape()}")
    print(f"Model 2 input shape: {model2.get_input_shape()}")
    print(f"Model 3 input shape: {model3.get_input_shape()}")

    # Cleanup
    proxy1.unload()
    print("\nCleaned up Model 1")
    print("Models 2 and 3 will auto-cleanup based on their TTL policies")


def main() -> None:
    """Run all demonstrations."""
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please download the model first or update the MODEL_PATH")
        return

    demo_with_real_session()
    demo_with_proxy_persistent()
    demo_with_proxy_auto_unload()
    demo_multiple_models()

    print("\n=== All demos completed ===")


if __name__ == "__main__":
    main()
