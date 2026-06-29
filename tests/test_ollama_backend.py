from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

from cull_sh.backends.base import VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.backends.ollama import _parse_batch_payload
from cull_sh.backends.ollama import _parse_edit_payload
from cull_sh.backends.ollama import _normalize_label
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset


class OllamaBackendTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
