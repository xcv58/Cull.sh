from __future__ import annotations

from pathlib import Path

from cull_sh.models import RawAsset


def discover_raw_assets(
    root: Path,
    extensions: tuple[str, ...],
) -> list[RawAsset]:
    """Recursively discover supported RAW files under the target path."""
    normalized = {extension.lower() for extension in extensions}
    assets: list[RawAsset] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized:
            continue
        assets.append(RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp")))

    assets.sort(key=lambda item: str(item.raw_path))
    return assets
