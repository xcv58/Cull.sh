from __future__ import annotations

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

ET.register_namespace("x", ADOBE_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("xmp", XMP_NS)
ET.register_namespace("xmpDM", XMP_DM_NS)
ET.register_namespace("crs", CRS_NS)


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

    write_xmp_sidecar(
        asset.xmp_path,
        decision,
        apply_lightroom_edit=apply_lightroom_edit,
        lightroom_edit_scope=lightroom_edit_scope,
    )


def write_jpeg_metadata(path: Path, decision: FinalDecision) -> None:
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise RuntimeError("exiftool is required to write JPEG metadata")

    command = [
        exiftool,
        "-overwrite_original",
        *_jpeg_decision_args(decision),
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"exiftool failed to write JPEG metadata: {error}")


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
    """Set Camera Raw develop attributes for a suggested global edit."""
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


def photo_is_rejected(asset: RawAsset) -> bool:
    if asset.is_jpeg:
        return jpeg_is_rejected(asset.raw_path)
    return sidecar_is_rejected(asset.xmp_path)


def sidecar_has_adaptive_color_payload(path: Path) -> bool:
    """Return True when Lightroom has generated a real Adaptive Color payload."""
    if not path.exists():
        return False

    tree = ET.parse(path)
    description = _find_description(tree.getroot())
    if description is None:
        return False

    return _has_ai_look(description) and _has_adaptive_color_look(description)


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
    description.set(f"{{{CRS_NS}}}LensProfileEnable", "1")
    description.set(f"{{{CRS_NS}}}LensProfileSetup", "LensDefaults")
    description.set(f"{{{CRS_NS}}}LensProfileDistortionScale", "100")
    description.set(f"{{{CRS_NS}}}LensProfileVignettingScale", "100")
    description.set(f"{{{CRS_NS}}}LensManualDistortionAmount", "0")
    description.set(f"{{{CRS_NS}}}HasSettings", "True")

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
