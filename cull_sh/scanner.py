from __future__ import annotations

from pathlib import Path

from cull_sh.models import AssetKind
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
        if path.name.startswith("._"):
            # macOS writes AppleDouble companions on some external volumes.
            # They share the photo extension but contain resource-fork data,
            # not an image that the pipeline can process.
            continue
        if path.suffix.lower() not in normalized:
            continue
        assets.append(RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp")))

    assets.sort(key=lambda item: str(item.raw_path))
    return assets


def discover_jpeg_assets(
    root: Path,
    extensions: tuple[str, ...],
    raw_assets: list[RawAsset] | None = None,
    mirror_paired_jpegs: bool = True,
) -> list[RawAsset]:
    normalized = {extension.lower() for extension in extensions}
    raw_by_stem = {
        (asset.raw_path.parent, asset.raw_path.stem.lower()): asset
        for asset in raw_assets or []
    }
    assets: list[RawAsset] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in normalized:
            continue
        paired_raw = raw_by_stem.get((path.parent, path.stem.lower()))
        paired_raw_path = (
            paired_raw.raw_path
            if mirror_paired_jpegs and paired_raw is not None
            else None
        )
        assets.append(
            RawAsset(
                raw_path=path,
                xmp_path=path,
                kind=AssetKind.JPEG,
                paired_raw_path=paired_raw_path,
            )
        )

    assets.sort(key=lambda item: str(item.raw_path))
    return assets


def discover_photo_assets(
    root: Path,
    raw_extensions: tuple[str, ...],
    include_jpegs: bool,
    jpeg_extensions: tuple[str, ...],
    mirror_paired_jpegs: bool = True,
) -> list[RawAsset]:
    raw_assets = discover_raw_assets(root, raw_extensions)
    if not include_jpegs:
        return raw_assets

    assets = [
        *raw_assets,
        *discover_jpeg_assets(
            root,
            jpeg_extensions,
            raw_assets=raw_assets,
            mirror_paired_jpegs=mirror_paired_jpegs,
        ),
    ]
    assets.sort(key=lambda item: str(item.raw_path))
    return assets
