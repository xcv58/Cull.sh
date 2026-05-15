from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

from cull_sh.backends.base import VisionBackendError
from cull_sh.backends.ollama import OllamaVisionBackend
from cull_sh.backends.ollama import _parse_batch_payload
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
                    '{"decisions":[{"filename":"frame.ARW","bucket":"review",'
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

    def test_parse_batch_payload_rejects_missing_filename(self) -> None:
        preview = _build_preview("frame.ARW")
        payload = {
            "message": {
                "content": (
                    '{"decisions":[{"filename":"other.ARW","bucket":"review",'
                    '"rating":0,"label":null,"summary":""}]}'
                )
            }
        }

        with self.assertRaises(VisionBackendError) as context:
            _parse_batch_payload(payload, [preview])

        self.assertIn("expected filenames", str(context.exception))


def _build_preview(filename: str) -> PreviewImage:
    return PreviewImage(
        asset=RawAsset(
            raw_path=Path(f"/tmp/{filename}"),
            xmp_path=Path(f"/tmp/{Path(filename).stem}.xmp"),
        ),
        image_bytes=b"jpeg-bytes",
    )


if __name__ == "__main__":
    unittest.main()
