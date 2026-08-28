import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from cull_sh.edit_feedback import _validate_delivery_pixels
from cull_sh.rapidraw import rapidraw_adjustments_from_suggestion
from cull_sh.render_diagnostics import (
    detail_sheet,
    image_diagnostics,
    leveling_evidence,
    tag_srgb_jpeg,
)


def jpeg(image):
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


class RenderDiagnosticsTests(unittest.TestCase):
    def test_vignette_sign_not_inverted_by_adapter(self):
        for value in [-10, 0, 10]:
            self.assertEqual(
                rapidraw_adjustments_from_suggestion({"vignette_amount": value})[
                    "vignetteAmount"
                ],
                value,
            )

    def test_bright_scene_is_evidence_not_automatic_failure(self):
        facts = image_diagnostics(jpeg(Image.new("RGB", (200, 100), "white")))
        self.assertEqual(facts["near_white_fraction"], 1)
        self.assertNotIn("failed", facts)

    def test_native_detail_sheet_contains_five_patches(self):
        data = detail_sheet(jpeg(Image.new("RGB", (1000, 800), "gray")))
        with Image.open(BytesIO(data)) as image:
            self.assertEqual(image.size, (1152, 824))

    def test_line_slope_advisory_not_forced_rotation(self):
        im = Image.new("RGB", (1000, 700), "white")
        draw = ImageDraw.Draw(im)
        for y in [100, 200, 300, 400, 500]:
            draw.line((30, y, 970, y + 33), fill="black", width=4)
        facts = leveling_evidence(jpeg(im))
        self.assertEqual(facts["status"], "advisory_only")
        self.assertAlmostEqual(facts["clockwise_line_slope_degrees"], 2, delta=0.3)
        self.assertGreater(facts["agreement"], 0.8)

    def test_blank_image_has_no_rotation_evidence(self):
        self.assertEqual(
            leveling_evidence(jpeg(Image.new("RGB", (200, 100), "gray")))["status"],
            "insufficient_evidence",
        )

    def test_icc_injection_is_lossless_and_preserves_exif(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo.jpg"
            exif = Image.Exif()
            exif[271] = "Test camera"
            Image.new("RGB", (80, 60), (30, 110, 160)).save(path, exif=exif)
            with Image.open(path) as im:
                before = np.asarray(im).copy()
            tag_srgb_jpeg(path)
            with Image.open(path) as im:
                self.assertTrue(im.info["icc_profile"])
                self.assertEqual(im.getexif()[271], "Test camera")
                np.testing.assert_array_equal(before, np.asarray(im))
            saved = path.read_bytes()
            tag_srgb_jpeg(path)
            self.assertEqual(saved, path.read_bytes())

    def test_export_requires_matching_pixels_not_just_dimensions(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.jpg", Path(tmp) / "b.jpg"
            Image.new("RGB", (80, 60), "gray").save(a, quality=88)
            Image.new("RGB", (80, 60), "gray").save(b, quality=95)
            _validate_delivery_pixels(a, b)
            Image.new("RGB", (80, 60), "green").save(b)
            with self.assertRaisesRegex(ValueError, "differ materially"):
                _validate_delivery_pixels(a, b)
