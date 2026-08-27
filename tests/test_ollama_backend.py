from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
from PIL import Image

from cull_sh.backends import build_backend
from cull_sh.backends.base import VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.backends.ollama import _image_boundary_facts
from cull_sh.backends.ollama import _parse_batch_payload
from cull_sh.backends.ollama import _parse_edit_payload
from cull_sh.backends.ollama import _normalize_label
from cull_sh.config import BackendConfig
from cull_sh.config import DEFAULT_PRODUCTION_MODEL
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import EditReviewPair
from cull_sh.models import EditReviewVerdict
from cull_sh.models import EditSuggestion
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset


class OllamaBackendTests(unittest.TestCase):
    def test_image_boundary_facts_distinguish_real_pixels_from_display_padding(
        self,
    ) -> None:
        clean = _jpeg_bytes((40, 20), "white")
        bordered_image = Image.new("RGB", (40, 20), "white")
        for y in range(20):
            bordered_image.putpixel((0, y), (0, 0, 0))
            bordered_image.putpixel((39, y), (0, 0, 0))
        output = BytesIO()
        bordered_image.save(output, format="PNG")

        clean_facts = _image_boundary_facts(clean)
        bordered_facts = _image_boundary_facts(output.getvalue())

        self.assertEqual(clean_facts["width"], 40)
        self.assertFalse(clean_facts["solid_near_black_edge_detected"])
        self.assertTrue(bordered_facts["solid_near_black_edge_detected"])

    def test_review_edits_returns_one_bounded_refinement(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        pair = EditReviewPair(
            asset=preview.asset,
            baseline_bytes=b"baseline",
            edited_bytes=b"edited",
            suggestion=EditSuggestion(
                filename="frame.ARW",
                asset_id=preview.asset.raw_path.as_posix(),
                exposure=0.2,
                brightness=0.1,
                contrast=10,
                highlights=-20,
                shadows=15,
                whites=5,
                blacks=-4,
                temperature=3,
                tint=-2,
                vibrance=5,
                saturation=2,
                clarity=4,
                dehaze=3,
                structure=2,
                sharpness=8,
                luma_noise_reduction=5,
                color_noise_reduction=4,
                vignette_amount=-3,
            ),
        )
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"reviews":[{"id":"image-1","filename":"frame.ARW",'
                    '"verdict":"refine","exposure":1.4,"brightness":1.2,"contrast":80,'
                    '"highlights":-70,"shadows":55,"whites":60,"blacks":-60,'
                    '"temperature":50,"tint":-50,"vibrance":45,"saturation":40,'
                    '"clarity":50,"dehaze":45,"structure":40,"sharpness":60,'
                    '"luma_noise_reduction":70,"color_noise_reduction":60,'
                    '"vignette_amount":-50,"additional_edits":["Mask the subject."],'
                    '"has_crop":true,"crop_left":0.05,"crop_top":0.05,'
                    '"crop_right":0.95,"crop_bottom":0.95,"crop_angle":8,'
                    '"summary":"Reduce the remaining darkness."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            reviews = backend.review_edits("natural", [pair])

        self.assertEqual(reviews[0].verdict, EditReviewVerdict.REFINE)
        self.assertEqual(reviews[0].final_suggestion.exposure, 0.7)
        self.assertEqual(reviews[0].final_suggestion.brightness, 0.6)
        self.assertEqual(reviews[0].final_suggestion.contrast, 40)
        self.assertEqual(reviews[0].final_suggestion.highlights, -50)
        self.assertEqual(reviews[0].final_suggestion.whites, 35)
        self.assertEqual(reviews[0].final_suggestion.temperature, 33)
        self.assertEqual(reviews[0].final_suggestion.luma_noise_reduction, 35)
        self.assertEqual(reviews[0].final_suggestion.crop_angle, 5.0)
        self.assertTrue(reviews[0].final_suggestion.has_crop)
        self.assertEqual(
            reviews[0].final_suggestion.additional_edits, ["Mask the subject."]
        )
        request_payload = fake_client.post.call_args.kwargs["json"]
        self.assertEqual(len(request_payload["messages"][1]["images"]), 2)
        self.assertIn("baseline", request_payload["messages"][1]["content"])
        self.assertIn(
            "Judge the rendered result",
            request_payload["messages"][0]["content"],
        )
        self.assertNotIn(
            "Prefer accept",
            request_payload["messages"][0]["content"],
        )

    def test_review_prompt_supplies_actual_pixel_bounds_and_ignores_padding(
        self,
    ) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        pair = EditReviewPair(
            asset=preview.asset,
            baseline_bytes=_jpeg_bytes((40, 20), "white"),
            edited_bytes=_jpeg_bytes((40, 12), "white"),
            suggestion=EditSuggestion(
                filename="frame.ARW",
                has_crop=True,
                crop_bottom=0.6,
            ),
        )
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"reviews":[{"id":"image-1","filename":"frame.ARW",'
                    '"verdict":"accept","exposure":0,"contrast":0,'
                    '"highlights":0,"shadows":0,"vibrance":0,'
                    '"has_crop":true,"crop_left":0,"crop_top":0,'
                    '"crop_right":1,"crop_bottom":0.6,"crop_angle":0,'
                    '"summary":"The crop is clean."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            backend.review_edits("natural", [pair])

        request = fake_client.post.call_args.kwargs["json"]
        content = request["messages"][1]["content"]
        self.assertIn('"width": 40', content)
        self.assertIn('"height": 12', content)
        self.assertIn('"solid_near_black_edge_detected": false', content)
        self.assertIn("display padding is not part of the photo", content)
        self.assertIn("when one small bounded tone", content)
        self.assertIn("whether it stops slightly early", content)

    def test_review_reject_reverts_recipe_but_preserves_additional_intents(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        pair = EditReviewPair(
            asset=preview.asset,
            baseline_bytes=b"baseline",
            edited_bytes=b"edited",
            suggestion=EditSuggestion(
                filename="frame.ARW",
                exposure=0.3,
                additional_edits=["Generate a subject mask."],
            ),
        )
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"reviews":[{"id":"image-1","filename":"frame.ARW",'
                    '"verdict":"reject","exposure":0,"contrast":0,'
                    '"highlights":0,"shadows":0,"vibrance":0,'
                    '"additional_edits":["Use a tighter subject mask."],'
                    '"has_crop":false,"crop_left":0,"crop_top":0,'
                    '"crop_right":1,"crop_bottom":1,"crop_angle":0,'
                    '"summary":"The baseline is better."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            review = backend.review_edits("natural", [pair])[0]

        self.assertEqual(review.verdict, EditReviewVerdict.REJECT)
        self.assertTrue(review.final_suggestion.is_noop)
        self.assertEqual(
            review.final_suggestion.additional_edits,
            ["Use a tighter subject mask."],
        )

    def test_review_accept_fills_abbreviated_unchanged_recipe(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen",
            timeout_seconds=300.0,
            max_attempts=1,
        )
        preview = _build_preview("frame.ARW")
        suggestion = EditSuggestion(
            filename="frame.ARW",
            asset_id=preview.asset.raw_path.as_posix(),
            exposure=0.3,
            contrast=12,
            vibrance=8,
            has_crop=True,
            crop_right=0.9,
        )
        pair = EditReviewPair(
            asset=preview.asset,
            baseline_bytes=b"baseline",
            edited_bytes=b"edited",
            suggestion=suggestion,
        )
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"reviews":[{"id":"image-1","verdict":"accept",'
                    '"comment":"The rendered edit is a natural improvement.",'
                    '"additional_edits":["Inspect sharpening at full resolution."]}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            review = backend.review_edits("natural", [pair])[0]

        self.assertEqual(review.verdict, EditReviewVerdict.ACCEPT)
        self.assertEqual(review.final_suggestion, suggestion)
        self.assertEqual(
            review.summary,
            "The rendered edit is a natural improvement.",
        )

    def test_review_refine_does_not_fill_abbreviated_recipe(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen",
            timeout_seconds=300.0,
            max_attempts=1,
        )
        preview = _build_preview("frame.ARW")
        pair = EditReviewPair(
            asset=preview.asset,
            baseline_bytes=b"baseline",
            edited_bytes=b"edited",
            suggestion=EditSuggestion(filename="frame.ARW", exposure=0.3),
        )
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"reviews":[{"id":"image-1","verdict":"refine",'
                    '"comment":"Reduce the contrast slightly."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            with self.assertRaisesRegex(
                VisionBackendError, "invalid structured edit review"
            ):
                backend.review_edits("natural", [pair])

    def test_production_backend_defaults_to_qwen_thinking_and_one_attempt(self) -> None:
        backend = build_backend(BackendConfig())

        self.assertIsInstance(backend, OllamaVisionBackend)
        assert isinstance(backend, OllamaVisionBackend)
        self.assertEqual(backend.model, DEFAULT_PRODUCTION_MODEL)
        self.assertEqual(backend.max_attempts, 1)
        self.assertTrue(backend.think)
        self.assertEqual(backend.max_output_tokens, 2048)

    def test_normalize_label_accepts_null_like_values(self) -> None:
        self.assertIsNone(_normalize_label(None))
        self.assertIsNone(_normalize_label(""))
        self.assertIsNone(_normalize_label("null"))
        self.assertIsNone(_normalize_label("None"))

    def test_normalize_label_maps_supported_colors(self) -> None:
        self.assertEqual(_normalize_label("green"), ColorLabel.GREEN)
        self.assertEqual(_normalize_label("Purple"), ColorLabel.PURPLE)

    def test_score_batch_retries_timeout_then_succeeds(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="gemma4:latest",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        success_response = MagicMock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "message": {
                "content": (
                    '{"decisions":[{"id":"image-1","filename":"frame.ARW","bucket":"review",'
                    '"rating":0,"label":null,"summary":""}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.side_effect = [
            httpx.ReadTimeout("timed out"),
            success_response,
        ]

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            with patch("cull_sh.backends.ollama.time.sleep") as sleep:
                decisions = backend.score_batch("prompt", [preview])

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].bucket, DecisionBucket.REVIEW)
        self.assertEqual(fake_client.post.call_count, 2)
        sleep.assert_called_once_with(2.0)

    def test_score_batch_forwards_benchmark_inference_controls(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen3.8:27b",
            max_attempts=1,
            temperature=0.25,
            think=False,
            max_output_tokens=321,
        )
        preview = _build_preview("frame.ARW")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"decisions":[{"id":"image-1","filename":"frame.ARW",'
                    '"bucket":"review","rating":0,"label":null,"summary":""}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            backend.score_batch("prompt", [preview])

        payload = fake_client.post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 0.25)
        self.assertEqual(payload["options"]["num_predict"], 321)
        self.assertFalse(payload["think"])

    def test_score_batch_raises_after_repeated_server_errors(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="gemma4:latest",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        error = httpx.HTTPStatusError(
            "server error",
            request=request,
            response=httpx.Response(500, request=request),
        )
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.side_effect = [error, error, error]

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            with patch("cull_sh.backends.ollama.time.sleep") as sleep:
                with self.assertRaises(VisionBackendError) as context:
                    backend.score_batch("prompt", [preview])

        self.assertIn("after 3 attempt(s)", str(context.exception))
        self.assertEqual(fake_client.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_suggest_edits_maps_adjustments_in_order(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="gemma4:12b",
            timeout_seconds=300.0,
        )
        previews = [_build_preview("a.ARW"), _build_preview("b.ARW")]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"edits":['
                    '{"id":"image-2","filename":"b.ARW","exposure":0.0,"contrast":0,"highlights":0,'
                    '"shadows":0,"vibrance":0,"summary":"ok"},'
                    '{"id":"image-1","filename":"a.ARW","exposure":0.5,"contrast":10,"highlights":-40,'
                    '"shadows":25,"vibrance":8,"summary":"recover sky"}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            suggestions = backend.suggest_edits("prompt", previews)

        # Results are remapped back to input order by the short image id.
        self.assertEqual([s.filename for s in suggestions], ["a.ARW", "b.ARW"])
        self.assertEqual(suggestions[0].exposure, 0.5)
        self.assertEqual(suggestions[0].highlights, -40)
        self.assertTrue(suggestions[1].is_noop)

    def test_suggest_edits_maps_expanded_rapidraw_recipe_and_other_intents(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="qwen",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.2,"brightness":0.1,"contrast":8,'
                    '"highlights":-15,"shadows":12,"whites":6,"blacks":-4,'
                    '"temperature":5,"tint":-3,"vibrance":7,"saturation":2,'
                    '"clarity":9,"dehaze":4,"structure":3,"sharpness":11,'
                    '"luma_noise_reduction":14,"color_noise_reduction":8,'
                    '"vignette_amount":-5,"additional_edits":'
                    '["  Add a subtle subject mask.  ",""],'
                    '"has_crop":true,"crop_left":0.2,"crop_top":0.2,'
                    '"crop_right":0.8,"crop_bottom":0.8,"crop_angle":-1.2,'
                    '"summary":"Correct the cool, flat rendering and level the frame."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            suggestion = backend.suggest_edits(
                "prompt", [preview], include_crop=True
            )[0]

        self.assertEqual(suggestion.brightness, 0.1)
        self.assertEqual(suggestion.whites, 6)
        self.assertEqual(suggestion.blacks, -4)
        self.assertEqual(suggestion.temperature, 5)
        self.assertEqual(suggestion.tint, -3)
        self.assertEqual(suggestion.clarity, 9)
        self.assertEqual(suggestion.luma_noise_reduction, 14)
        self.assertEqual(suggestion.color_noise_reduction, 8)
        self.assertEqual(suggestion.vignette_amount, -5)
        self.assertTrue(suggestion.has_crop)
        self.assertEqual(suggestion.crop_angle, -1.2)
        self.assertEqual(
            suggestion.additional_edits, ["Add a subtle subject mask."]
        )
        self.assertFalse(suggestion.is_noop)

    def test_suggest_edits_maps_crop_adjustments_when_requested(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="gemma4:12b",
            timeout_seconds=300.0,
        )
        preview = _build_preview("frame.ARW")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.2,"contrast":4,"highlights":-8,'
                    '"shadows":6,"vibrance":3,"has_crop":true,'
                    '"crop_left":0.1,"crop_top":0.05,'
                    '"crop_right":0.9,"crop_bottom":0.95,'
                    '"crop_angle":-1.5,'
                    '"summary":"Level horizon and tighten edges."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            suggestions = backend.suggest_edits("prompt", [preview], include_crop=True)

        self.assertEqual(len(suggestions), 1)
        self.assertTrue(suggestions[0].has_crop)
        self.assertEqual(suggestions[0].crop_left, 0.1)
        self.assertEqual(suggestions[0].crop_top, 0.05)
        self.assertEqual(suggestions[0].crop_right, 0.9)
        self.assertEqual(suggestions[0].crop_bottom, 0.95)
        self.assertEqual(suggestions[0].crop_angle, -1.5)
        request_payload = fake_client.post.call_args.kwargs["json"]
        self.assertIn(
            "independently evaluate crop and rotation",
            request_payload["messages"][0]["content"],
        )
        self.assertIn(
            "first-class optional edits",
            request_payload["messages"][0]["content"],
        )
        self.assertIn(
            "Composition/crop/rotation pass",
            request_payload["messages"][1]["content"],
        )
        self.assertIn(
            "actively check whether a crop would improve composition",
            request_payload["messages"][1]["content"],
        )
        self.assertIn(
            "Do not add a generic inset crop",
            request_payload["messages"][1]["content"],
        )
        self.assertIn(
            "Crop as much or as little as the visible issue warrants",
            request_payload["messages"][1]["content"],
        )
        self.assertIn("has_crop", str(request_payload["format"]))
        self.assertIn("additional_edits", str(request_payload["format"]))
        required_fields = {
            field
            for definition in request_payload["format"]["$defs"].values()
            for field in definition.get("required", [])
        }
        self.assertIn("temperature", required_fields)
        self.assertIn("clarity", required_fields)
        self.assertIn("additional_edits", required_fields)
        self.assertIn(
            "Do not narrate slider actions",
            request_payload["messages"][1]["content"],
        )
        self.assertIn(
            "Do not fall back to only exposure",
            request_payload["messages"][1]["content"],
        )
        self.assertIn(
            "Never place global exposure",
            request_payload["messages"][1]["content"],
        )
        self.assertIn(
            "exposure is a linear RAW EV shift",
            request_payload["messages"][1]["content"],
        )
        self.assertNotIn(
            'or "No global adjustment needed."',
            request_payload["messages"][1]["content"],
        )

    def test_parse_edit_payload_rejects_wrong_count(self) -> None:
        previews = [_build_preview("a.ARW"), _build_preview("b.ARW")]
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"a.ARW","exposure":0.0,"contrast":0,'
                    '"highlights":0,"shadows":0,"vibrance":0,'
                    '"summary":"No global adjustment needed."}]}'
                )
            }
        }

        with self.assertRaises(VisionBackendError) as context:
            _parse_edit_payload(payload, previews)

        self.assertIn("one result per input image", str(context.exception))

    def test_parse_batch_payload_rejects_missing_filename(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"decisions":[{"id":"image-1","filename":"other.ARW","bucket":"review",'
                    '"rating":0,"label":null,"summary":""}]}'
                )
            }
        }

        with self.assertRaises(VisionBackendError) as context:
            _parse_batch_payload(payload, [preview])

        self.assertIn("expected filenames", str(context.exception))

    def test_parse_edit_payload_recovers_first_json_object(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    "```json\n"
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.1,"contrast":4,"highlights":-10,'
                    '"shadows":5,"vibrance":3,"summary":"Slightly dark foreground."}]}'
                    "\n```\nextra text"
                )
            }
        }

        parsed = _parse_edit_payload(payload, [preview])

        self.assertEqual(parsed.edits[0].summary, "Slightly dark foreground.")
        self.assertEqual(parsed.edits[0].exposure, 0.1)

    def test_parse_edit_payload_accepts_crop_fields_when_requested(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.1,"contrast":4,"highlights":-10,'
                    '"shadows":5,"vibrance":3,"has_crop":true,'
                    '"crop_left":0.1,"crop_top":0.05,'
                    '"crop_right":0.9,"crop_bottom":0.95,'
                    '"crop_angle":-1.5,'
                    '"summary":"Slightly dark foreground."}]}'
                )
            }
        }

        parsed = _parse_edit_payload(payload, [preview], include_crop=True)

        self.assertTrue(parsed.edits[0].has_crop)
        self.assertEqual(parsed.edits[0].crop_left, 0.1)
        self.assertEqual(parsed.edits[0].crop_angle, -1.5)

    def test_parse_edit_payload_normalizes_disabled_crop_fields(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.1,"contrast":4,"highlights":-10,'
                    '"shadows":5,"vibrance":3,"has_crop":false,'
                    '"crop_left":0.2,"crop_top":0.2,'
                    '"crop_right":0.8,"crop_bottom":0.8,'
                    '"crop_angle":0,'
                    '"summary":"Slightly dark foreground."}]}'
                )
            }
        }

        parsed = _parse_edit_payload(payload, [preview], include_crop=True)

        self.assertFalse(parsed.edits[0].has_crop)
        self.assertEqual(parsed.edits[0].crop_left, 0.0)
        self.assertEqual(parsed.edits[0].crop_top, 0.0)
        self.assertEqual(parsed.edits[0].crop_right, 1.0)
        self.assertEqual(parsed.edits[0].crop_bottom, 1.0)
        self.assertEqual(parsed.edits[0].crop_angle, 0.0)

    def test_parse_edit_payload_rejects_rotation_without_crop(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.1,"contrast":4,"highlights":-10,'
                    '"shadows":5,"vibrance":3,"has_crop":false,'
                    '"crop_left":0,"crop_top":0,"crop_right":1,'
                    '"crop_bottom":1,"crop_angle":2,'
                    '"summary":"The horizon is tilted."}]}'
                )
            }
        }

        with self.assertRaisesRegex(VisionBackendError, "invalid structured output"):
            _parse_edit_payload(payload, [preview], include_crop=True)

    def test_parse_edit_payload_rejects_invalid_crop_rectangle(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.1,"contrast":4,"highlights":-10,'
                    '"shadows":5,"vibrance":3,"has_crop":true,'
                    '"crop_left":0.8,"crop_top":0.05,'
                    '"crop_right":0.1,"crop_bottom":0.95,'
                    '"crop_angle":0,'
                    '"summary":"Slightly dark foreground."}]}'
                )
            }
        }

        with self.assertRaises(VisionBackendError) as context:
            _parse_edit_payload(payload, [preview], include_crop=True)

        self.assertIn("invalid structured output", str(context.exception))

    def test_parse_edit_payload_rejects_missing_required_slider(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"contrast":4,"highlights":-10,"shadows":5,'
                    '"vibrance":3,"summary":"Slightly dark foreground."}]}'
                )
            }
        }

        with self.assertRaises(VisionBackendError) as context:
            _parse_edit_payload(payload, [preview])

        self.assertIn("invalid structured output", str(context.exception))

    def test_parse_edit_payload_rejects_blank_summary(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"edits":[{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.1,"contrast":4,"highlights":-10,'
                    '"shadows":5,"vibrance":3,"summary":"   "}]}'
                )
            }
        }

        with self.assertRaises(VisionBackendError) as context:
            _parse_edit_payload(payload, [preview])

        self.assertIn("invalid structured output", str(context.exception))

    def test_suggest_edits_uses_ids_for_duplicate_filenames(self) -> None:
        backend = OllamaVisionBackend(
            base_url="http://localhost:11434",
            model="gemma4:12b",
            timeout_seconds=300.0,
        )
        previews = [
            _build_preview_from_path("/tmp/a/frame.ARW"),
            _build_preview_from_path("/tmp/b/frame.ARW"),
        ]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": (
                    '{"edits":['
                    '{"id":"image-2","filename":"frame.ARW",'
                    '"exposure":-0.2,"contrast":1,"highlights":-5,'
                    '"shadows":2,"vibrance":1,"summary":"Bright upper frame."},'
                    '{"id":"image-1","filename":"frame.ARW",'
                    '"exposure":0.3,"contrast":8,"highlights":-20,'
                    '"shadows":10,"vibrance":5,"summary":"Dark lower frame."}]}'
                )
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.post.return_value = response

        with patch("cull_sh.backends.ollama.httpx.Client", return_value=fake_client):
            suggestions = backend.suggest_edits("prompt", previews)

        self.assertEqual(
            [suggestion.asset_id for suggestion in suggestions],
            ["/tmp/a/frame.ARW", "/tmp/b/frame.ARW"],
        )
        self.assertEqual([suggestion.exposure for suggestion in suggestions], [0.3, -0.2])


def _build_preview(filename: str) -> PreviewImage:
    return _build_preview_from_path(f"/tmp/{filename}")


def _build_preview_from_path(raw_path: str) -> PreviewImage:
    path = Path(raw_path)
    return PreviewImage(
        asset=RawAsset(
            raw_path=path,
            xmp_path=path.with_suffix(".xmp"),
        ),
        image_bytes=b"jpeg-bytes",
    )


def _jpeg_bytes(size: tuple[int, int], color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG", quality=95)
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
