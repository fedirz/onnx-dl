"""
Transparent proxy wrapper for onnxruntime.InferenceSession with automatic lifecycle management.

This module provides InferenceSessionProxy, which manages ONNX model loading and unloading
based on configurable time-to-live (TTL) policies while being completely transparent to
the calling code.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Any

import onnxruntime as ort

logger = logging.getLogger(__name__)


class InferenceSessionProxy:
    """
    A transparent proxy wrapper around onnxruntime.InferenceSession that manages
    model lifecycle independently of the inference pipeline.

    The proxy handles automatic loading and unloading of ONNX models based on
    configurable time-to-live (TTL) policies, while implementing the same interface
    as onnxruntime.InferenceSession.

    Thread-safe for concurrent access from multiple threads.

    Args:
        model_path: Path to the ONNX model file
        sess_options: ONNX Runtime session options (optional)
        providers: List of execution providers (optional)
        ttl: Time-to-live in seconds for model lifecycle management
            -1: Never unload (persistent, loaded once and kept in memory)
             0: Unload immediately after each use
            >0: Unload after N seconds of inactivity

    Example:
        >>> # Persistent model (never unload)
        >>> proxy = InferenceSessionProxy("model.onnx", ttl=-1)
        >>>
        >>> # Unload immediately after use
        >>> proxy = InferenceSessionProxy("model.onnx", ttl=0)
        >>>
        >>> # Keep for 60 seconds after last use
        >>> proxy = InferenceSessionProxy("model.onnx", ttl=60)
        >>>
        >>> # Use like a regular InferenceSession
        >>> result = proxy.run([output_name], {input_name: data})
        >>>
        >>> # Or use context manager for batch processing
        >>> with proxy.session() as sess:
        ...     for data in batch:
        ...         result = sess.run([output_name], {input_name: data})
    """

    def __init__(
        self,
        model_path: str | Path,
        sess_options: ort.SessionOptions | None = None,
        providers: list[str] | None = None,
        ttl: int = -1,
    ) -> None:
        # Model configuration
        self._model_path = Path(model_path).expanduser()
        self._sess_options = sess_options
        self._providers = providers
        self._ttl = ttl

        # State management
        self._session: ort.InferenceSession | None = None
        self._lock = threading.Lock()  # Protects all shared state
        self._timer: threading.Timer | None = None  # TTL cleanup timer
        self._ref_count = 0  # Active context manager references
        self._active_operations = 0  # Active operations (inference, metadata access, etc.)
        self._last_access_time = 0.0  # Track last usage for TTL verification

    def _load_session(self) -> None:
        """
        Load the ONNX model into memory.

        This method is thread-safe and idempotent - multiple calls will not
        load the model multiple times.

        Called automatically on first use (lazy loading).
        """
        with self._lock:
            # Only load if not already loaded
            if self._session is None:
                logger.info(f"Loading ONNX model from {self._model_path}")
                self._session = ort.InferenceSession(
                    str(self._model_path),
                    sess_options=self._sess_options,
                    providers=self._providers,
                )
                logger.info(f"Model loaded successfully: {self._model_path.name}")
            # Cancel any pending cleanup timer since we're using the model
            self._cancel_timer()
            # Update last access time for TTL tracking
            self._last_access_time = time.time()

    def _cancel_timer(self) -> None:
        """
        Cancel any pending TTL cleanup timer.

        Must be called while holding self._lock.
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_unload(self) -> None:
        """
        Schedule model unloading based on TTL policy.

        This method implements the TTL-based lifecycle management:
        - ttl = -1: No-op (never unload)
        - ttl = 0: Unload immediately
        - ttl > 0: Schedule unload after ttl seconds

        Must be called while holding self._lock.
        """
        # Don't unload if there are active context manager references or operations
        if self._ref_count > 0 or self._active_operations > 0:
            return

        # Cancel any existing timer before scheduling new one
        self._cancel_timer()

        if self._ttl == -1:
            # Persistent mode - never unload
            return
        elif self._ttl == 0:
            # Immediate unload mode
            self._unload_session()
        else:
            # Timed unload mode - schedule cleanup after TTL seconds
            # Store the access time to verify TTL hasn't been reset before unloading
            access_time_at_scheduling = self._last_access_time
            self._timer = threading.Timer(
                self._ttl,
                self._ttl_expired_callback,
                args=(access_time_at_scheduling,),
            )
            self._timer.daemon = True  # Don't block program exit
            self._timer.start()

    def _ttl_expired_callback(self, scheduled_access_time: float) -> None:
        """
        Callback invoked when TTL timer expires.

        Verifies that the TTL has actually elapsed (guards against race conditions
        where the model was used again before the timer fired) before unloading.

        Args:
            scheduled_access_time: The last_access_time when this timer was scheduled
        """
        with self._lock:
            # Verify TTL hasn't been reset by checking if access time has changed
            if self._last_access_time == scheduled_access_time:
                # Don't unload if there are active context manager references or operations
                if self._ref_count == 0 and self._active_operations == 0:
                    logger.debug(f"TTL expired for {self._model_path.name} (ttl={self._ttl}s), unloading model")
                    self._unload_session()
                else:
                    logger.debug(
                        f"TTL expired for {self._model_path.name}, but model has {self._ref_count} active references "
                        f"and {self._active_operations} active operations"
                    )

    def _unload_session(self) -> None:
        """
        Unload the ONNX model from memory.

        Must be called while holding self._lock.
        """
        if self._session is not None:
            logger.info(f"Unloading ONNX model: {self._model_path.name}")
            self._session = None
        # Cancel any pending timer
        self._cancel_timer()

    def _schedule_cleanup_if_needed(self) -> None:
        """
        Schedule cleanup based on TTL policy after the session has been used.

        Must be called after using the session to ensure proper TTL-based lifecycle.
        Should be called from a finally block to guarantee execution.
        """
        with self._lock:
            self._schedule_unload()

    @property
    def is_loaded(self) -> bool:
        """
        Check if the session is currently loaded in memory.

        Returns:
            True if the session is loaded, False otherwise
        """
        with self._lock:
            return self._session is not None

    def unload(self) -> None:
        """
        Manually unload the session immediately.

        Cancels any pending TTL timers and removes the model from memory.
        Safe to call even if the session is not loaded.
        """
        with self._lock:
            self._unload_session()

    @contextmanager
    def session(self) -> Generator[ort.InferenceSession, None, None]:
        """
        Context manager that provides explicit control over model lifecycle.

        The model is loaded (if not already) when entering the context and kept
        in memory for the entire duration of the with block. TTL-based cleanup
        only applies after exiting the context.

        Supports nested contexts through reference counting.

        Yields:
            ort.InferenceSession: The underlying InferenceSession instance

        Example:
            >>> with proxy.session() as sess:
            ...     for data in batch:
            ...         result = sess.run([output_name], {input_name: data})
            ... # TTL timer starts here
        """
        # Load the session if not already loaded
        self._load_session()

        # Increment reference count to prevent unloading during context
        # Get session reference while holding lock, but don't hold lock during yield
        with self._lock:
            self._ref_count += 1
            logger.debug(f"Entering context for {self._model_path.name}, active sessions: {self._ref_count}")
            session_ref = self._session
            assert session_ref is not None

        try:
            # Yield the session without holding the lock to avoid deadlocks
            yield session_ref
        finally:
            # Decrement reference count and schedule cleanup if this was the last context
            with self._lock:
                self._ref_count -= 1
                logger.debug(f"Exiting context for {self._model_path.name}, active sessions: {self._ref_count}")
                # Only schedule cleanup when last context exits
                if self._ref_count == 0:
                    self._schedule_unload()

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, Any],
        run_options: ort.RunOptions | None = None,
    ) -> list[Any]:
        """
        Run inference on the model.

        This method transparently proxies to the underlying InferenceSession.run()
        method, handling model loading and lifecycle management automatically.

        Args:
            output_names: Names of the outputs to compute (None = all outputs)
            input_feed: Dictionary mapping input names to input data
            run_options: Optional RunOptions for this inference call

        Returns:
            List of output values
        """
        # Load session if needed (may acquire lock during loading)
        self._load_session()

        # Get session reference and increment operation counter under lock
        with self._lock:
            session_ref = self._session
            assert session_ref is not None
            self._active_operations += 1

        try:
            # Run inference WITHOUT holding the lock (allows concurrent inference)
            return session_ref.run(output_names, input_feed, run_options)  # pyright: ignore[reportReturnType]
        finally:
            # Decrement counter and schedule cleanup under lock
            with self._lock:
                self._active_operations -= 1
                self._schedule_unload()

    def get_inputs(self) -> list[ort.NodeArg]:
        """
        Get input metadata.

        Returns:
            List of input NodeArg objects describing the model inputs
        """
        self._load_session()

        with self._lock:
            session_ref = self._session
            assert session_ref is not None
            self._active_operations += 1

        try:
            return session_ref.get_inputs()  # pyright: ignore[reportReturnType]
        finally:
            with self._lock:
                self._active_operations -= 1
                self._schedule_unload()

    def get_outputs(self) -> list[ort.NodeArg]:
        """
        Get output metadata.

        Returns:
            List of output NodeArg objects describing the model outputs
        """
        self._load_session()

        with self._lock:
            session_ref = self._session
            assert session_ref is not None
            self._active_operations += 1

        try:
            return session_ref.get_outputs()  # pyright: ignore[reportReturnType]
        finally:
            with self._lock:
                self._active_operations -= 1
                self._schedule_unload()

    def get_providers(self) -> Sequence[str]:
        """
        Get execution providers.

        Returns:
            List of execution provider names
        """
        self._load_session()

        with self._lock:
            session_ref = self._session
            assert session_ref is not None
            self._active_operations += 1

        try:
            return session_ref.get_providers()
        finally:
            with self._lock:
                self._active_operations -= 1
                self._schedule_unload()

    def __getattr__(self, name: str) -> Any:
        """
        Proxy all other InferenceSession methods and attributes.

        This allows the proxy to be fully transparent and support all
        InferenceSession methods without explicitly implementing each one.

        Args:
            name: Attribute or method name

        Returns:
            The attribute or method from the underlying session

        Note:
            If the returned value is a callable method, calling it will NOT
            hold the lock. This means the method can execute concurrently, but
            there's a small risk of the session being unloaded during execution
            if TTL expires. For critical methods, implement them explicitly like
            run(), get_inputs(), etc.
        """
        self._load_session()

        with self._lock:
            session_ref = self._session
            self._active_operations += 1

        try:
            return getattr(session_ref, name)
        finally:
            with self._lock:
                self._active_operations -= 1
                self._schedule_unload()

    def __del__(self) -> None:
        """
        Cleanup on object destruction.

        Ensures that timers are cancelled and the session is unloaded
        when the proxy object is garbage collected.
        """
        try:
            with self._lock:
                self._unload_session()
        except Exception:  # noqa: BLE001, S110
            # Ignore all errors during cleanup to avoid issues during interpreter shutdown.
            # This is intentional - destructors should never raise exceptions.
            pass
