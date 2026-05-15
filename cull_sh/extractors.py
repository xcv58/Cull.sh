from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import subprocess


class PreviewExtractionError(RuntimeError):
    """Raised when a preview cannot be extracted from a RAW file."""


class PreviewExtractor(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_preview_bytes(self, raw_path: Path) -> bytes:
        raise NotImplementedError


class ExifToolPreviewExtractor(PreviewExtractor):
    preview_tags = ("PreviewImage", "JpgFromRaw", "OtherImage")

    def __init__(self, executable: str = "exiftool") -> None:
        self.executable = executable

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def extract_preview_bytes(self, raw_path: Path) -> bytes:
        """
        Extract the embedded preview without modifying the RAW file.

        This is the intended primary path for Phase 1. The exact preview flags may
        need adjustment per vendor once real camera samples are tested.
        """
        errors: list[str] = []

        for tag in self.preview_tags:
            command = [
                self.executable,
                "-b",
                f"-{tag}",
                str(raw_path),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            if result.stderr:
                errors.append(result.stderr.decode(errors="ignore").strip())

        joined_errors = "; ".join(error for error in errors if error)
        raise PreviewExtractionError(
            f"exiftool preview extraction failed for {raw_path}: {joined_errors}"
        )


class SipsPreviewExtractor(PreviewExtractor):
    def __init__(self, executable: str = "sips") -> None:
        self.executable = executable

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def extract_preview_bytes(self, raw_path: Path) -> bytes:
        raise PreviewExtractionError(
            "sips fallback is scaffolded but not implemented yet"
        )


def build_default_extractor() -> PreviewExtractor:
    primary = ExifToolPreviewExtractor()
    if primary.is_available():
        return primary

    fallback = SipsPreviewExtractor()
    if fallback.is_available():
        return fallback

    raise PreviewExtractionError("no supported preview extractor is available")
