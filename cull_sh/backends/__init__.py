from cull_sh.backends.base import VisionBackend, VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.config import BackendConfig


def build_backend(config: BackendConfig) -> VisionBackend:
    provider = config.provider.lower()
    if provider == "ollama":
        return OllamaVisionBackend(
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            think=config.think,
            max_output_tokens=config.max_output_tokens,
            context_tokens=config.context_tokens,
        )
    raise ValueError(f"unsupported backend provider: {config.provider}")


__all__ = [
    "OllamaVisionBackend",
    "VisionBackend",
    "VisionBackendError",
    "build_backend",
]
