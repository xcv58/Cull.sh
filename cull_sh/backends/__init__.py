from cull_sh.backends.base import VisionBackend
from cull_sh.backends.base import VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.config import BackendConfig


def build_backend(config: BackendConfig) -> VisionBackend:
    provider = config.provider.lower()
    if provider == "ollama":
        return OllamaVisionBackend(
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    raise ValueError(f"unsupported backend provider: {config.provider}")


__all__ = [
    "OllamaVisionBackend",
    "VisionBackend",
    "VisionBackendError",
    "build_backend",
]
