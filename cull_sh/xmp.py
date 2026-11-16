from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import EditSuggestion
from cull_sh.models import FinalDecision
from cull_sh.models import LightroomEditScope
from cull_sh.models import RawAsset


ADOBE_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
XMP_DM_NS = "http://ns.adobe.com/xmp/1.0/DynamicMedia/"
CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
NAMESPACES = {
    "x": ADOBE_NS,
    "rdf": RDF_NS,
    "xmp": XMP_NS,
    "xmpDM": XMP_DM_NS,
    "crs": CRS_NS,
}
LENS_CORRECTION_SETTINGS = {
    "LensProfileEnable": "1",
    "LensProfileSetup": "LensDefaults",
    "LensProfileDistortionScale": "100",
    "LensProfileVignettingScale": "100",
    "LensManualDistortionAmount": "0",
    "HasSettings": "True",
}

ET.register_namespace("x", ADOBE_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("xmpDM", XMP_DM_NS)
ET.register_namespace("crs", CRS_NS)


class DngMetadataError(RuntimeError):
    """The DNG metadata could not be read or safely written."""


@dataclass(slots=True)
class EmbeddedDngState:
    rejected: bool = False
    has_develop_settings: bool = False
    has_adaptive_color: bool = False
    has_lens_corrections: bool = False
    has_culling_metadata: bool = False

    @property
    def has_active_metadata(self) -> bool:
        return self.has_develop_settings or self.has_lens_corrections or self.has_culling_metadata


def read_embedded_dng_state(raw_path: Path) -> EmbeddedDngState | None:
    if raw_path.suffix.lower() != ".dng":
        return None

    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise DngMetadataError(f"ExifTool is required to check embedded edits in {raw_path}")

    try:
        result = subprocess.run(
            [
                exiftool,
                "-j",
                "-XMP-crs:AILookActive",
                "-XMP-crs:AILookAILookData",
                "-XMP-crs:HasSettings",
                "-XMP-crs:LookName",
                "-XMP-crs:LensProfileEnable",
                "-XMP-xmp:Rating",
                "-XMP-xmpDM:Pick",
                "-XMP-xmpDM:Good",
                "-XMP-xmp:Label",
                str(raw_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DngMetadataError(f"Unable to read embedded metadata in {raw_path}: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or f"ExifTool exited with status {result.returncode}"
        raise DngMetadataError(f"Unable to read embedded metadata in {raw_path}: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DngMetadataError(f"Invalid ExifTool response for {raw_path}") from exc
    if not payload:
        raise DngMetadataError(f"ExifTool returned no embedded DNG metadata for {raw_path}")

    metadata = payload[0]
    return EmbeddedDngState(
        rejected=metadata.get("Rating") == -1 or metadata.get("Pick") == -1,
        has_develop_settings=(
            metadata.get("HasSettings") is True or metadata.get("AILookActive") is True
        ),
        has_adaptive_color=(
            metadata.get("AILookActive") is True
            and metadata.get("LookName") == "Adaptive Color"
            and bool(metadata.get("AILookAILookData"))
        ),
        has_lens_corrections=metadata.get("LensProfileEnable") == 1,
        has_culling_metadata=any(tag in metadata for tag in ("Rating", "Pick", "Good", "Label")),
    )


def read_active_embedded_dng_state(asset: RawAsset) -> EmbeddedDngState | None:
    if asset.raw_path.suffix.lower() != ".dng" or asset.xmp_path.exists():
        return None
    return read_embedded_dng_state(asset.raw_path)


def culling_requires_embedded_dng(asset: RawAsset, *, allow_embedded: bool = False) -> bool:
    embedded = read_active_embedded_dng_state(asset)
    requires_embedded = embedded is not None and embedded.has_active_metadata
    if requires_embedded and not allow_embedded:
        raise DngMetadataError(
            f"{asset.raw_path} has embedded Lightroom edits or culling tags "
            "that a new XMP sidecar would hide. "
            "Run cull in dry-run mode, then replay the manifest with "
            "repair-sidecars --embed-dng-metadata."
        )
    return requires_embedded


def write_xmp_sidecar(
    path: Path,
    decision: FinalDecision,
    apply_lightroom_edit: bool = False,
    lightroom_edit_scope: LightroomEditScope = LightroomEditScope.ALL,
) -> None:
    """Create or update an XMP sidecar while preserving unrelated metadata."""
    tree = _load_or_create_tree(path)
    root = tree.getroot()
    description = _find_or_create_description(root)
    label = _lightroom_label(decision)

    description.set(f"{{{RDF_NS}}}about", description.get(f"{{{RDF_NS}}}about", ""))
    if decision.bucket == DecisionBucket.REJECT:
        description.set(f"{{{XMP_NS}}}Rating", str(decision.rating))
        description.set(f"{{{XMP_DM_NS}}}Pick", "-1")
        description.set(f"{{{XMP_DM_NS}}}good", "False")
    elif decision.bucket == DecisionBucket.PICK:
        description.set(f"{{{XMP_NS}}}Rating", str(decision.rating))
        description.set(f"{{{XMP_DM_NS}}}Pick", "1")
        description.set(f"{{{XMP_DM_NS}}}good", "True")
    else:
        description.attrib.pop(f"{{{XMP_NS}}}Rating", None)
        description.attrib.pop(f"{{{XMP_DM_NS}}}Pick", None)
        description.attrib.pop(f"{{{XMP_DM_NS}}}good", None)
    if label is not None:
        description.set(f"{{{XMP_NS}}}Label", label.value)
    else:
        description.attrib.pop(f"{{{XMP_NS}}}Label", None)

    if (
        apply_lightroom_edit
        and (lightroom_edit_scope == LightroomEditScope.ALL or decision.keep)
    ):
        apply_lightroom_sidecar_edit(description)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_photo_metadata(
    asset: RawAsset,
    decision: FinalDecision,
    apply_lightroom_edit: bool = False,
    lightroom_edit_scope: LightroomEditScope = LightroomEditScope.ALL,
) -> None:
    if asset.is_jpeg:
        write_jpeg_metadata(asset.raw_path, decision)
        return

    culling_requires_embedded_dng(asset)
    write_xmp_sidecar(
        asset.xmp_path,
        decision,
        apply_lightroom_edit=apply_lightroom_edit,
        lightroom_edit_scope=lightroom_edit_scope,
    )


def write_jpeg_metadata(path: Path, decision: FinalDecision) -> None:
    write_embedded_cull_metadata(path, decision)


def write_embedded_cull_metadata(path: Path, decision: FinalDecision) -> None:
    """Update culling tags without replacing Lightroom's embedded develop packet."""
    _write_embedded_xmp(path, _jpeg_decision_args(decision))


def write_embedded_lens_corrections(path: Path) -> None:
    """Enable lens defaults in a DNG whose culling tags are embedded."""
    _write_embedded_xmp(
        path,
        [f"-XMP-crs:{name}={value}" for name, value in LENS_CORRECTION_SETTINGS.items()],
    )


def _write_embedded_xmp(path: Path, arguments: list[str]) -> None:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise RuntimeError("exiftool is required to write embedded metadata")

    command = [
        exiftool,
        "-overwrite_original",
        *arguments,
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"exiftool failed to write embedded metadata: {error}")


def write_lightroom_edit_sidecar(path: Path) -> None:
    """Apply safe Lightroom sidecar edit settings without changing culling state."""
    tree = _load_or_create_tree(path)
    root = tree.getroot()
    description = _find_or_create_description(root)
    apply_lightroom_sidecar_edit(description)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_develop_sidecar(path: Path, suggestion: EditSuggestion) -> None:
    """Write suggested global develop adjustments into an XMP sidecar.

    These are standard Camera Raw settings that Lightroom reads as fully
    reversible edits; culling state in the sidecar is left untouched.
    """
    tree = _load_or_create_tree(path)
    root = tree.getroot()
    description = _find_or_create_description(root)
    apply_develop_settings(description, suggestion)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def apply_develop_settings(description: ET.Element, suggestion: EditSuggestion) -> None:
    """Set the safely interoperable Camera Raw subset of a RapidRAW recipe."""
    description.set(
        f"{{{CRS_NS}}}Version",
        description.get(f"{{{CRS_NS}}}Version", "18.3"),
    )
    description.set(
        f"{{{CRS_NS}}}ProcessVersion",
        description.get(f"{{{CRS_NS}}}ProcessVersion", "15.4"),
    )
    description.set(f"{{{CRS_NS}}}Exposure2012", f"{suggestion.exposure:+.2f}")
    description.set(f"{{{CRS_NS}}}Contrast2012", str(suggestion.contrast))
    description.set(f"{{{CRS_NS}}}Highlights2012", str(suggestion.highlights))
    description.set(f"{{{CRS_NS}}}Shadows2012", str(suggestion.shadows))
    description.set(f"{{{CRS_NS}}}Vibrance", str(suggestion.vibrance))
    if suggestion.has_crop:
        description.set(f"{{{CRS_NS}}}HasCrop", "True")
        description.set(f"{{{CRS_NS}}}CropTop", _format_crop_value(suggestion.crop_top))
        description.set(f"{{{CRS_NS}}}CropLeft", _format_crop_value(suggestion.crop_left))
        description.set(
            f"{{{CRS_NS}}}CropBottom",
            _format_crop_value(suggestion.crop_bottom),
        )
        description.set(
            f"{{{CRS_NS}}}CropRight",
            _format_crop_value(suggestion.crop_right),
        )
        description.set(
            f"{{{CRS_NS}}}CropAngle",
            _format_crop_value(suggestion.crop_angle),
        )
    description.set(f"{{{CRS_NS}}}HasSettings", "True")


def _format_crop_value(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def sidecar_is_rejected(path: Path) -> bool:
    """Return True when an existing sidecar marks the photo as rejected."""
    if not path.exists():
        return False

    tree = ET.parse(path)
    description = _find_description(tree.getroot())
    if description is None:
        return False

    rating = description.get(f"{{{XMP_NS}}}Rating")
    pick = description.get(f"{{{XMP_DM_NS}}}Pick")
    return rating == "-1" or pick == "-1"


def sidecar_is_picked(path: Path) -> bool:
    """Return True when an existing sidecar marks the photo as picked.

    Lightroom Desktop Local mode treats xmpDM:good as its visible flag state.
    When present, it is authoritative because older Rating/Pick attributes may
    remain after a human changes the flag.
    """
    if not path.exists():
        return False

    tree = ET.parse(path)
    description = _find_description(tree.getroot())
    if description is None:
        return False

    good = _metadata_boolean(description.get(f"{{{XMP_DM_NS}}}good"))
    if good is not None:
        return good

    rating = _metadata_integer(description.get(f"{{{XMP_NS}}}Rating"))
    pick = _metadata_integer(description.get(f"{{{XMP_DM_NS}}}Pick"))
    return (rating is not None and rating > 0) or (pick is not None and pick > 0)


def jpeg_is_rejected(path: Path) -> bool:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise RuntimeError("exiftool is required to read JPEG metadata")

    result = subprocess.run(
        [
            exiftool,
            "-j",
            "-XMP-xmp:Rating",
            "-XMP-xmpDM:Pick",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"exiftool failed to read JPEG metadata: {error}")

    import json

    payload = json.loads(result.stdout)
    if not payload:
        return False
    metadata = payload[0]
    return _metadata_is_negative(metadata.get("Rating")) or _metadata_is_negative(
        metadata.get("Pick")
    )


def jpeg_is_picked(path: Path) -> bool:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise RuntimeError("exiftool is required to read JPEG metadata")

    result = subprocess.run(
        [
            exiftool,
            "-j",
            "-XMP-xmp:Rating",
            "-XMP-xmpDM:Pick",
            "-XMP-xmpDM:good",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"exiftool failed to read JPEG metadata: {error}")

    import json

    payload = json.loads(result.stdout)
    if not payload:
        return False
    metadata = payload[0]
    good = _metadata_boolean(metadata.get("Good"))
    if good is not None:
        return good
    rating = _metadata_integer(metadata.get("Rating"))
    pick = _metadata_integer(metadata.get("Pick"))
    return (rating is not None and rating > 0) or (pick is not None and pick > 0)


def photo_is_rejected(asset: RawAsset) -> bool:
    if asset.is_jpeg:
        return jpeg_is_rejected(asset.raw_path)
    return sidecar_is_rejected(asset.xmp_path)


def photo_is_picked(asset: RawAsset) -> bool:
    if asset.is_jpeg or (
        asset.raw_path.suffix.lower() == ".dng" and not asset.xmp_path.exists()
    ):
        return jpeg_is_picked(asset.raw_path)
    return sidecar_is_picked(asset.xmp_path)


def sidecar_has_adaptive_color_payload(path: Path) -> bool:
    """Verify the active Adaptive Color reference and its Lightroom RAW sidecar."""
    if not path.exists():
        return False

    tree = ET.parse(path)
    description = _find_description(tree.getroot())
    if description is None:
        return False

    ai_look = description.find("crs:AILook", NAMESPACES)
    if (
        ai_look is None
        or _metadata_boolean(ai_look.get(f"{{{CRS_NS}}}Active")) is not True
        or not ai_look.get(f"{{{CRS_NS}}}AILookData", "").strip()
        or not _has_adaptive_color_look(description)
    ):
        return False

    acr_path = path.with_suffix(".acr")
    return acr_path.is_file() and acr_path.stat().st_size > 0


def sidecar_has_lens_corrections(path: Path) -> bool:
    """Return True when an existing sidecar enables lens profile corrections."""
    if not path.exists():
        return False

    tree = ET.parse(path)
    description = _find_description(tree.getroot())
    if description is None:
        return False

    return description.get(f"{{{CRS_NS}}}LensProfileEnable") == "1"


def apply_lightroom_sidecar_edit(description: ET.Element) -> None:
    """Write stable Camera Raw settings for the safe sidecar edit.

    Adaptive Color has per-image AI data that Lightroom computes itself. Do not
    synthesize that profile from static XMP; Lightroom must apply it directly.
    """
    description.set(
        f"{{{CRS_NS}}}Version",
        description.get(f"{{{CRS_NS}}}Version", "18.3"),
    )
    description.set(
        f"{{{CRS_NS}}}ProcessVersion",
        description.get(f"{{{CRS_NS}}}ProcessVersion", "15.4"),
    )
    for name, value in LENS_CORRECTION_SETTINGS.items():
        description.set(f"{{{CRS_NS}}}{name}", value)

    has_ai_look = _has_ai_look(description)
    if _has_adaptive_color_look(description) and not has_ai_look:
        _remove_child_elements(description, f"{{{CRS_NS}}}Look")
    if (
        not has_ai_look
        and description.get(f"{{{CRS_NS}}}CameraProfile") == "Adobe Standard"
    ):
        description.attrib.pop(f"{{{CRS_NS}}}CameraProfile", None)
        description.attrib.pop(f"{{{CRS_NS}}}CameraProfileDigest", None)


def _lightroom_label(decision: FinalDecision) -> ColorLabel | None:
    if decision.bucket == DecisionBucket.PICK:
        return decision.label
    if decision.bucket == DecisionBucket.REVIEW:
        return decision.label
    if decision.source == DecisionSource.LOCAL:
        return ColorLabel.RED
    return ColorLabel.YELLOW


def _jpeg_decision_args(decision: FinalDecision) -> list[str]:
    label = _lightroom_label(decision)
    if decision.bucket == DecisionBucket.REJECT:
        return [
            f"-XMP-xmp:Rating={decision.rating}",
            "-XMP-xmpDM:Pick=-1",
            "-XMP-xmpDM:good=False",
            f"-XMP-xmp:Label={label.value if label else ''}",
        ]
    if decision.bucket == DecisionBucket.PICK:
        return [
            f"-XMP-xmp:Rating={decision.rating}",
            "-XMP-xmpDM:Pick=1",
            "-XMP-xmpDM:good=True",
            f"-XMP-xmp:Label={label.value if label else ''}",
        ]
    return [
        "-XMP-xmp:Rating=",
        "-XMP-xmpDM:Pick=",
        "-XMP-xmpDM:good=",
        "-XMP-xmp:Label=",
    ]


def _metadata_is_negative(value: object) -> bool:
    if value == -1 or value == "-1":
        return True
    return False


def _metadata_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _metadata_boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _load_or_create_tree(path: Path) -> ET.ElementTree:
    if path.exists():
        return ET.parse(path)

    root = ET.Element(f"{{{ADOBE_NS}}}xmpmeta")
    root.set(f"{{{ADOBE_NS}}}xmptk", "Cull.sh")
    rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
    description = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
    description.set(f"{{{RDF_NS}}}about", "")
    return ET.ElementTree(root)


def _find_or_create_description(root: ET.Element) -> ET.Element:
    rdf = root.find("rdf:RDF", NAMESPACES)
    if rdf is None:
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")

    description = rdf.find("rdf:Description", NAMESPACES)
    if description is None:
        description = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
    return description


def _find_description(root: ET.Element) -> ET.Element | None:
    rdf = root.find("rdf:RDF", NAMESPACES)
    if rdf is None:
        return None
    return rdf.find("rdf:Description", NAMESPACES)


def _remove_child_elements(parent: ET.Element, tag: str) -> None:
    for child in list(parent):
        if child.tag == tag:
            parent.remove(child)


def _has_adaptive_color_look(description: ET.Element) -> bool:
    for look in description.findall("crs:Look", NAMESPACES):
        look_description = look.find("rdf:Description", NAMESPACES)
        if look_description is None:
            continue
        if look_description.get(f"{{{CRS_NS}}}Name") == "Adaptive Color":
            return True
    return False


def _has_ai_look(description: ET.Element) -> bool:
    return description.find("crs:AILook", NAMESPACES) is not None
