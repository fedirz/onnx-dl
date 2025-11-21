"""
ONNX Dynamic Loading - Transparent proxy for onnxruntime.InferenceSession with lifecycle management.
"""

from onnx_dl.proxy import InferenceSessionProxy

__all__ = ["InferenceSessionProxy"]
