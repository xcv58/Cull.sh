from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from cull_sh.config import DEFAULT_EXTENSIONS
from cull_sh.models import AssetKind
from cull_sh.models import RawAsset


class SidecarCollisionError(ValueError):
    """Distinct RAW files share a sidecar name."""


def validate_unique_sidecar_targets(assets: Iterable[RawAsset]) -> None:
    claims: dict[str, Path] = {}
    directories: set[Path] = set()

    def check_claim(sidecar: Path, raw: Path) -> str:
        # macOS commonly treats different casing as the same local filename.
        key = str(sidecar.resolve()).casefold()
        owner = claims.get(key)
        if owner is not None and owner.resolve() != raw.resolve():
            raise SidecarCollisionError(
                f"RAW files {owner} and {raw} share the XMP sidecar {sidecar}. "
                "Give them distinct basenames or move them into separate folders."
            )
        return key

    for asset in assets:
        if not asset.is_jpeg:
            claims[check_claim(asset.xmp_path, asset.raw_path)] = asset.raw_path
            directories.add(asset.xmp_path.parent.resolve())

    # A limited manifest can omit a sibling RAW that would still read the same XMP.
    for directory in sorted(directories):
        if directory.is_dir():
            for sibling in directory.iterdir():
                if sibling.suffix.lower() in DEFAULT_EXTENSIONS and sibling.is_file():
                    check_claim(sibling.with_suffix(".xmp"), sibling)


def discover_raw_assets(
    root: Path,
    extensions: tuple[str, ...],
) -> list[RawAsset]:
    """Recursively discover supported RAW files under the target path."""
    normalized = {extension.lower() for extension in extensions}
    assets: list[RawAsset] = []

    for path in _discover_source_files(root, normalized):
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

    for path in _discover_source_files(root, normalized):
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


def _discover_source_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue  # Includes macOS AppleDouble companions (._photo.ARW).
        if any(part.casefold() == "output" for part in relative.parts[:-1]):
            continue
        yield path


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
