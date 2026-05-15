from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LightroomEditScope


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


def write_lightroom_edit_sidecar(path: Path) -> None:
    """Apply safe Lightroom sidecar edit settings without changing culling state."""
    tree = _load_or_create_tree(path)
    root = tree.getroot()
    description = _find_or_create_description(root)
    apply_lightroom_sidecar_edit(description)

    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


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
