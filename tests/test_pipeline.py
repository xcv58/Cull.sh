from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from cull_sh.models import AssetKind
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LocalQualityMetrics
from cull_sh.models import RawAsset
from cull_sh.models import WorkItem
from cull_sh.models import WorkStatus
from cull_sh.pipeline import apply_local_rejection
from cull_sh.pipeline import apply_vision_decision
from cull_sh.pipeline import limit_items_to_scenes
from cull_sh.pipeline import mirror_paired_jpeg_decisions
from cull_sh.pipeline import persist_decisions
from cull_sh.config import PipelineConfig


class PipelineDecisionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
