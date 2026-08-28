"""Renderer-specific instructions and measured evidence, never aesthetic verdicts."""

import math
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

IMAGE_TRANSPORT_POLICY = "overview-4mp-native-detail-v1"


def model_image_bytes(data: bytes) -> bytes:
    """Bound request size; never replace the full-resolution render on disk.

    Small images (including native-pixel detail sheets) pass through unchanged.
    Large overviews are bounded to 4 MP, 2560 pixels per edge, and 4 MiB JPEG.
    """
    max_pixels, max_edge, max_bytes = 4_000_000, 2560, 4 * 1024 * 1024
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
        if (
            width * height <= max_pixels
            and max(width, height) <= max_edge
            and len(data) <= max_bytes
        ):
            return data
        scale = min(
            1.0, max_edge / max(width, height), math.sqrt(max_pixels / (width * height))
        )
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.convert("RGB").resize(target, Image.Resampling.LANCZOS)
        for quality in (90, 85, 80):
            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=quality,
                icc_profile=image.info.get("icc_profile"),
            )
            encoded = output.getvalue()
            if len(encoded) <= max_bytes:
                return encoded
    raise ValueError("image exceeds the bounded vision transport size")


CONTROL_GUIDANCE = (
    "RapidRAW control semantics (not Lightroom slider values):\n"
    "- exposure: linear RAW EV; brightness: perceptual/filmic exposure.\n"
    "- temperature: relative shift, positive warmer/yellower, negative cooler/bluer; NOT Kelvin. "
    "tint: positive magenta, negative green. Do not warm every image by default.\n"
    "- vignette_amount: NEGATIVE darkens edges; POSITIVE mixes edges toward WHITE. "
    "Use positive only for an explicitly intended white-edge effect, never for ordinary dark framing. "
    "Avoid either vignette unless it solves a visible composition problem.\n"
    "- highlights/whites and shadows/blacks are distinct tonal controls. "
    "Protect highlight texture while making the subject and midtones readable; "
    "do not use global darkening or dehaze as a substitute for balanced tones.\n"
    "- clarity, structure, dehaze and sharpness can introduce halos, darkening or harsh detail. "
    "Inspect their actual rendered effects. Preserve natural night/sunset mood.\n"
)

DELIVERY_GUIDANCE = (
    "FINAL DELIVERY CHECK: accept only if the edited image itself is suitable for a natural travel album, "
    "not merely better than the baseline. An unchanged baseline may be accepted if already suitable. "
    "Inspect subject readability, daylight whites, unwanted yellow/green cast, highlight detail, "
    "white/milky corner haze, halos, noise, and crop/leveling. Preserve intentional silhouettes. "
    "A bright sun, dark sky or low average brightness is not automatically a defect. "
    "Use reject or refine if a material defect remains. This is the last check: no further automatic "
    "refinement will be executed, and a non-accept verdict blocks delivery. Explain the actual pixels."
)


def image_diagnostics(data: bytes) -> dict[str, object]:
    with Image.open(BytesIO(data)) as image:
        size = image.size
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        rgb = np.asarray(image, dtype=np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    h, w = luma.shape
    eh, ew = max(1, h // 8), max(1, w // 8)
    corners = [luma[:eh, :ew], luma[:eh, -ew:], luma[-eh:, :ew], luma[-eh:, -ew:]]
    return {
        "dimensions": list(size),
        "luma_percentiles_10_50_90": np.percentile(luma, [10, 50, 90])
        .round(2)
        .tolist(),
        "near_white_fraction": round(float((luma >= 252).mean()), 6),
        "near_black_fraction": round(float((luma <= 3).mean()), 6),
        "clipped_channel_fraction": round(float((rgb >= 254).any(axis=2).mean()), 6),
        "corner_mean_luma_TL_TR_BL_BR": [round(float(c.mean()), 2) for c in corners],
        "interpretation": "Descriptive evidence only. Bright sun, water, sky and silhouettes may be intentional.",
    }


def leveling_evidence(data: bytes) -> dict[str, object]:
    """Detect near-horizontal line consensus as an advisory, not an auto-rotation."""
    import cv2

    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("cannot decode image for leveling evidence")
    scale = min(1.0, 1200 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = cv2.HoughLinesP(
        cv2.Canny(gray, 70, 160),
        1,
        np.pi / 1800,
        60,
        minLineLength=max(35, gray.shape[1] // 6),
        maxLineGap=12,
    )
    candidates = []
    for line in [] if lines is None else lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        angle = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 90) % 180 - 90
        if abs(angle) <= 10:
            candidates.append((angle, math.hypot(x2 - x1, y2 - y1)))
    if len(candidates) < 3:
        return {"status": "insufficient_evidence", "line_count": len(candidates)}
    angles = np.array([a for a, _ in candidates])
    weights = np.array([w for _, w in candidates])
    order = np.argsort(angles)
    median = float(
        angles[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]]
    )
    agreement = float(weights[np.abs(angles - median) < 0.6].sum() / weights.sum())
    return {
        "status": "advisory_only",
        "line_count": len(candidates),
        "clockwise_line_slope_degrees": round(median, 2),
        "agreement": round(agreement, 3),
        "instruction": "These may be perspective lines, not a horizon. Visually confirm before leveling; "
        "never flatten intentional diagonals. A correction would oppose the observed slope.",
    }


def detail_sheet(data: bytes) -> bytes:
    """Native-pixel patches: corners and center, no enlargement or resampling."""
    with Image.open(BytesIO(data)) as source:
        source = source.convert("RGB")
        w, h = source.size
        side = min(384, w, h)
        positions = [
            (0, 0, "top left"),
            (w - side, 0, "top right"),
            ((w - side) // 2, (h - side) // 2, "center"),
            (0, h - side, "bottom left"),
            (w - side, h - side, "bottom right"),
        ]
        sheet = Image.new("RGB", (side * 3, side * 2 + 56), "#303030")
        draw = ImageDraw.Draw(sheet)
        for i, (x, y, label) in enumerate(positions):
            dx, dy = (i % 3) * side, (i // 3) * (side + 28)
            draw.text(
                (dx + 5, dy + 5), f"{label} - native pixels ({x},{y})", fill="white"
            )
            sheet.paste(source.crop((x, y, x + side, y + side)), (dx, dy + 28))
    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=93)
    return buffer.getvalue()


def tag_srgb_jpeg(path: Path) -> None:
    """Add an ICC APP2 marker without recompressing pixels or dropping EXIF."""
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError("sRGB tagging requires a JPEG")
        if image.info.get("icc_profile"):
            return
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    body = b"ICC_PROFILE\x00\x01\x01" + profile
    marker = b"\xff\xe2" + (len(body) + 2).to_bytes(2, "big") + body
    temporary = path.with_name(path.name + ".icc-tmp")
    temporary.write_bytes(data[:2] + marker + data[2:])
    temporary.replace(path)
