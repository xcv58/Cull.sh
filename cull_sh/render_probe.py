"""Opt-in installed-renderer integration probe. Writes only isolated output copies."""

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cull_sh.extractors import ExifToolPreviewExtractor
from cull_sh.models import EditSuggestion
from cull_sh.rapidraw import rapidraw_adjustments_from_suggestion
from cull_sh.render_diagnostics import (
    image_diagnostics,
    leveling_evidence,
    tag_srgb_jpeg,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument(
        "--stems", nargs="+", default=["DSC03806", "DSC03859", "DSC04120"]
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if root == args.source.resolve() or args.source.resolve() in root.parents:
        raise ValueError("probe output must be outside source")
    root.mkdir(parents=True, exist_ok=False)
    (root / "input").mkdir()

    def render(source, name, recipe):
        path = root / f"{name}.jpg"
        adj = root / f"{name}.json"
        adj.write_text(json.dumps(recipe, indent=2) + "\n")
        result = subprocess.run(
            [
                str(args.binary),
                "export",
                str(source),
                "--output",
                str(path),
                "--format",
                "jpeg",
                "--quality",
                "95",
                "--adjustments",
                str(adj),
            ],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        (root / f"{name}.log").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode or not path.is_file():
            raise RuntimeError(f"probe render failed: {name}; see log")
        tag_srgb_jpeg(path)
        print("Rendered", name, flush=True)
        return path

    # End-to-end tests of the installed app, not merely copied shader formulas.
    gray = root / "input/control.png"
    Image.new("RGB", (1200, 800), (100, 100, 100)).save(gray)
    controls = {}
    for amount in [0, -30, 30]:
        recipe = rapidraw_adjustments_from_suggestion(
            asdict(EditSuggestion(filename=gray.name, vignette_amount=amount))
        )
        controls[amount] = render(gray, f"vignette-{amount}", recipe)
    levels = {}
    for amount, path in controls.items():
        with Image.open(path) as im:
            a = np.asarray(im.convert("L"), dtype=float)
        levels[amount] = float(a[:60, :60].mean())
    assert levels[-30] < levels[0] < levels[30], levels
    color_tests = {}
    for field in ["temperature", "tint"]:
        averages = {}
        for amount in [-30, 30]:
            suggestion = EditSuggestion(filename=gray.name, **{field: amount})
            path = render(
                gray,
                f"{field}-{amount}",
                rapidraw_adjustments_from_suggestion(asdict(suggestion)),
            )
            with Image.open(path) as im:
                averages[amount] = (
                    np.asarray(im.convert("RGB"), dtype=float)
                    .mean(axis=(0, 1))
                    .tolist()
                )
        if field == "temperature":
            assert (
                averages[30][0] - averages[30][2] > averages[-30][0] - averages[-30][2]
            )
        else:
            magenta = lambda rgb: (rgb[0] + rgb[2]) / 2 - rgb[1]
            assert magenta(averages[30]) > magenta(averages[-30])
        color_tests[field] = averages
    tilted = root / "input/tilted-lines.png"
    fixture = Image.new("RGB", (1200, 800), (160, 160, 160))
    draw_lines = ImageDraw.Draw(fixture)
    for y in [200, 400, 600]:
        draw_lines.line((60, y, 1140, y + 38), fill=(30, 30, 30), width=5)
    fixture.save(tilted)
    rotations = {}
    for angle in [0, -2, 2]:
        suggestion = EditSuggestion(
            filename=tilted.name,
            has_crop=True,
            crop_left=0.06,
            crop_right=0.94,
            crop_top=0.06,
            crop_bottom=0.94,
            crop_angle=angle,
        )
        path = render(
            tilted,
            f"rotation-{angle}",
            rapidraw_adjustments_from_suggestion(
                asdict(suggestion), image_size=(1200, 800)
            ),
        )
        rotations[angle] = leveling_evidence(path.read_bytes())
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB"))
            # No rotation fill in a fixture whose real pixels are never black.
            assert (
                min(
                    pixels[0].min(),
                    pixels[-1].min(),
                    pixels[:, 0].min(),
                    pixels[:, -1].min(),
                )
                > 5
            )
    slope = lambda angle: abs(rotations[angle]["clockwise_line_slope_degrees"])
    assert slope(-2) < 0.4 and slope(2) > slope(0) + 1, rotations
    rows = []
    for stem in [] if args.controls_only else args.stems:
        raw = root / "input" / f"{stem}.ARW"
        shutil.copy2(args.source / raw.name, raw)
        row = {}
        for name, options in [
            ("basic", {"toneMapper": "basic"}),
            ("agx", {"toneMapper": "agx"}),
            ("basic-brightness", {"toneMapper": "basic", "brightness": 0.6}),
        ]:
            recipe = rapidraw_adjustments_from_suggestion(
                asdict(EditSuggestion(filename=raw.name))
            )
            recipe.update(options)
            row[name] = render(raw, f"{stem}-{name}", recipe)
        camera = root / f"{stem}-camera.jpg"
        camera.write_bytes(ExifToolPreviewExtractor().extract_preview_bytes(raw))
        row["camera"] = camera
        rows.append((stem, row))
    report = {
        "vignette_corner_luma": levels,
        "vignette_sign_test": "passed",
        "white_balance_sign_tests": color_tests,
        "rotation_sign_and_safe_edges_test": rotations,
        "photos": [
            {
                "filename": stem,
                "variants": {
                    name: {
                        "path": str(path),
                        "metrics": image_diagnostics(path.read_bytes()),
                    }
                    for name, path in row.items()
                },
            }
            for stem, row in rows
        ],
    }
    (root / "probe-report.json").write_text(json.dumps(report, indent=2) + "\n")
    sheet = Image.new("RGB", (1800, len(rows) * 350 + 45), "#222222")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    for col, name in enumerate(["basic", "agx", "basic-brightness", "camera"]):
        draw.text((col * 450 + 8, 10), name, font=font, fill="white")
    for i, (stem, row) in enumerate(rows):
        for col, path in enumerate(row.values()):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((435, 300))
                sheet.paste(
                    im,
                    (
                        col * 450 + (450 - im.width) // 2,
                        i * 350 + 75 + (300 - im.height) // 2,
                    ),
                )
            draw.text((col * 450 + 8, i * 350 + 48), stem, font=font, fill="white")
    sheet.save(root / "comparison.jpg", quality=95)
    print("Probe complete:", root, flush=True)


if __name__ == "__main__":
    main()
