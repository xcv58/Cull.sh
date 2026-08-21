from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from cull_sh.backends import VisionBackendError
from cull_sh.config import BackendConfig
from cull_sh.config import PipelineConfig
from cull_sh.models import AssetKind
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LocalQualityMetrics
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset
from cull_sh.models import WorkItem
from cull_sh.models import WorkStatus
from cull_sh.pipeline import apply_local_rejection
from cull_sh.pipeline import apply_vision_decision
from cull_sh.pipeline import limit_items_to_scenes
from cull_sh.pipeline import mirror_paired_jpeg_decisions
from cull_sh.pipeline import persist_decisions
from cull_sh.pipeline import score_with_backend
from cull_sh.reporting import NullReporter


class PipelineDecisionTests(unittest.TestCase):
    def test_vision_model_failure_aborts_before_next_cohort(self) -> None:
        first = _ready_item("first.ARW", "scene-0001")
        second = _ready_item("second.ARW", "scene-0002")
        backend = MagicMock()
        backend.score_batch.side_effect = VisionBackendError("malformed response")
        config = PipelineConfig(
            path=Path("/tmp"),
            prompt="test",
            backend=BackendConfig(max_attempts=1, fail_fast=True),
            batch_size=1,
        )

        with TemporaryDirectory() as tmp_dir:
            with patch("cull_sh.pipeline.build_backend", return_value=backend):
                with self.assertRaisesRegex(VisionBackendError, "scene-0001"):
                    score_with_backend(
                        [first, second],
                        config,
                        Path(tmp_dir),
                        NullReporter(),
                    )

        backend.score_batch.assert_called_once()
        self.assertEqual(first.status, WorkStatus.FAILED)
        self.assertEqual(second.status, WorkStatus.READY_FOR_VISION)

    def test_apply_local_rejection_marks_red(self) -> None:
        item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.ARW"),
                xmp_path=Path("/tmp/frame.xmp"),
            )
        )

        apply_local_rejection(
            item,
            LocalQualityMetrics(
                blur_score=12.5,
                tenengrad_score=10.0,
            ),
        )

        self.assertEqual(item.status, WorkStatus.REJECTED_LOCAL)
        self.assertIsNotNone(item.decision)
        assert item.decision is not None
        self.assertEqual(item.decision.rating, -1)
        self.assertEqual(item.decision.label, ColorLabel.RED)
        self.assertEqual(item.decision.bucket, DecisionBucket.REJECT)

    def test_apply_vision_decision_marks_rejects_yellow(self) -> None:
        item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.ARW"),
                xmp_path=Path("/tmp/frame.xmp"),
            )
        )
        decision = FinalDecision(
            filename="frame.ARW",
            rating=2,
            label=None,
            bucket=DecisionBucket.REJECT,
            source=DecisionSource.VISION,
            summary="Rejected by vision",
        )

        apply_vision_decision(item, decision)

        self.assertEqual(item.status, WorkStatus.SCORED)
        self.assertIsNotNone(item.decision)
        assert item.decision is not None
        self.assertEqual(item.decision.rating, -1)
        self.assertEqual(item.decision.label, ColorLabel.YELLOW)
        self.assertEqual(item.decision.bucket, DecisionBucket.REJECT)

    def test_apply_vision_decision_normalizes_review(self) -> None:
        item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.ARW"),
                xmp_path=Path("/tmp/frame.xmp"),
            )
        )
        decision = FinalDecision(
            filename="frame.ARW",
            rating=4,
            label=ColorLabel.GREEN,
            bucket=DecisionBucket.REVIEW,
            source=DecisionSource.VISION,
            summary="Borderline keeper",
        )

        apply_vision_decision(item, decision)

        self.assertEqual(item.status, WorkStatus.SCORED)
        self.assertIsNotNone(item.decision)
        assert item.decision is not None
        self.assertEqual(item.decision.rating, 0)
        self.assertIsNone(item.decision.label)
        self.assertEqual(item.decision.bucket, DecisionBucket.REVIEW)

    def test_apply_vision_decision_defaults_pick_to_green(self) -> None:
        item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.ARW"),
                xmp_path=Path("/tmp/frame.xmp"),
            )
        )
        decision = FinalDecision(
            filename="frame.ARW",
            rating=0,
            label=None,
            bucket=DecisionBucket.PICK,
            source=DecisionSource.VISION,
            summary="Strong frame",
        )

        apply_vision_decision(item, decision)

        self.assertEqual(item.status, WorkStatus.SCORED)
        self.assertIsNotNone(item.decision)
        assert item.decision is not None
        self.assertEqual(item.decision.rating, 4)
        self.assertEqual(item.decision.label, ColorLabel.GREEN)
        self.assertEqual(item.decision.bucket, DecisionBucket.PICK)

    def test_limit_items_to_scenes_keeps_whole_scenes(self) -> None:
        items = [
            WorkItem(
                asset=RawAsset(
                    raw_path=Path("/tmp/a1.ARW"),
                    xmp_path=Path("/tmp/a1.xmp"),
                ),
                scene_id="scene-0001",
            ),
            WorkItem(
                asset=RawAsset(
                    raw_path=Path("/tmp/a2.ARW"),
                    xmp_path=Path("/tmp/a2.xmp"),
                ),
                scene_id="scene-0001",
            ),
            WorkItem(
                asset=RawAsset(
                    raw_path=Path("/tmp/b1.ARW"),
                    xmp_path=Path("/tmp/b1.xmp"),
                ),
                scene_id="scene-0002",
            ),
        ]

        limited_items, scene_count = limit_items_to_scenes(items, scene_limit=1)

        self.assertEqual(scene_count, 1)
        self.assertEqual([item.filename for item in limited_items], ["a1.ARW", "a2.ARW"])

    def test_mirror_paired_jpeg_decisions_copies_raw_decision(self) -> None:
        raw_item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.ARW"),
                xmp_path=Path("/tmp/frame.xmp"),
            ),
            status=WorkStatus.SCORED,
            decision=FinalDecision(
                filename="frame.ARW",
                rating=4,
                label=ColorLabel.GREEN,
                bucket=DecisionBucket.PICK,
                source=DecisionSource.VISION,
                summary="Strong frame",
            ),
        )
        jpeg_item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.JPG"),
                xmp_path=Path("/tmp/frame.JPG"),
                kind=AssetKind.JPEG,
                paired_raw_path=Path("/tmp/frame.ARW"),
            )
        )

        mirrored = mirror_paired_jpeg_decisions([raw_item, jpeg_item])

        self.assertEqual(mirrored, 1)
        self.assertIsNotNone(jpeg_item.decision)
        assert jpeg_item.decision is not None
        self.assertEqual(jpeg_item.decision.bucket, DecisionBucket.PICK)
        self.assertEqual(jpeg_item.decision.rating, 4)
        self.assertEqual(jpeg_item.decision.label, ColorLabel.GREEN)

    def test_persist_decisions_writes_jpeg_embedded_metadata(self) -> None:
        item = WorkItem(
            asset=RawAsset(
                raw_path=Path("/tmp/frame.JPG"),
                xmp_path=Path("/tmp/frame.JPG"),
                kind=AssetKind.JPEG,
            ),
            decision=FinalDecision(
                filename="frame.JPG",
                rating=-1,
                label=ColorLabel.RED,
                bucket=DecisionBucket.REJECT,
                source=DecisionSource.LOCAL,
                summary="Soft",
            ),
        )
        config = PipelineConfig(path=Path("/tmp"), prompt="test", dry_run=False)

        with patch("cull_sh.pipeline.write_photo_metadata") as write_metadata:
            written = persist_decisions([item], config)

        self.assertEqual(written, 1)
        write_metadata.assert_called_once()
        self.assertTrue(item.sidecar_written)
        self.assertFalse(item.lightroom_edit_written)


def _ready_item(filename: str, scene_id: str) -> WorkItem:
    asset = RawAsset(
        raw_path=Path("/tmp") / filename,
        xmp_path=(Path("/tmp") / filename).with_suffix(".xmp"),
    )
    return WorkItem(
        asset=asset,
        status=WorkStatus.READY_FOR_VISION,
        scene_id=scene_id,
        preview=PreviewImage(asset=asset, image_bytes=b"jpeg-bytes"),
    )


if __name__ == "__main__":
    unittest.main()
