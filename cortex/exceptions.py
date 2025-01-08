"""
Hierarchical exception architecture for CortexRuntime.
"""

class CortexError(Exception):
    """Base exception for all internal CortexRuntime errors."""
    pass


class ConfigurationError(CortexError):
    """Raised when configuration parameters are invalid or unresolvable."""
    pass


class ContextOverflowError(CortexError):
    """Raised when prompt context exceeds hardware budget."""
    pass


class InferenceEngineError(CortexError):
    """Raised on unrecoverable backend inference transport failures."""
    pass


class CircuitBreakerOpenError(InferenceEngineError):
    """Raised when inference backend is temporarily disabled by circuit breaker."""
    pass


class AudioPipelineError(CortexError):
    """Raised on audio device, buffer overflow, or synthesis failure."""
    pass


class StorageError(CortexError):
    """Raised on persistence engine failure."""
    pass
