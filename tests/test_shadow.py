from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LocalQualityMetrics
from cull_sh.models import PreviewImage
from cull_sh.models import RawAsset
from cull_sh.models import WorkItem
from cull_sh.shadow import write_topiq_shadow_report


class TopiqShadowTests(unittest.TestCase):
    def test_report_finds_disagreements_without_changing_decisions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            items = [
                self._item("A.ARW", "scene-0001", 1.0, 100.0, DecisionBucket.PICK),
                self._item("B.ARW", "scene-0001", 9.0, 200.0, DecisionBucket.REJECT),
                self._item("C.ARW", "scene-0002", 8.0, 100.0, DecisionBucket.PICK),
                self._item("D.ARW", "scene-0002", 7.0, 300.0, DecisionBucket.REJECT),
            ]
            original = [item.decision.bucket for item in items if item.decision]

            summary = write_topiq_shadow_report(
                items,
                run_dir,
                low_percentile=0.25,
                high_percentile=0.75,
                max_items=10,
                rank_weight=0.25,
            )

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary.low_kept, 1)
            self.assertEqual(summary.high_rejected, 1)
            self.assertEqual(summary.scene_winner_disagreements, 1)
            self.assertEqual([item.decision.bucket for item in items if item.decision], original)
            payload = json.loads(summary.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision_influence"], "ranking_only")
            self.assertEqual(payload["rank_weight"], 0.25)
            self.assertEqual(payload["counts"]["review_items_written"], 4)
            self.assertEqual(
                payload["scene_winner_disagreements"][0]["topiq_winner"],
                "C.ARW",
            )
            self.assertEqual(
                payload["scene_winner_disagreements"][0]["local_winner"],
                "D.ARW",
            )
            html = summary.html_path.read_text(encoding="utf-8")
            self.assertIn("Ranking influence only", html)
            self.assertIn("25%", html)
            self.assertIn("A.ARW", html)
            self.assertEqual(len(list((run_dir / "topiq-shadow-thumbnails").glob("*.jpg"))), 4)

    def test_report_returns_none_without_topiq_scores(self) -> None:
        item = self._item("A.ARW", "scene-0001", 1.0, 100.0, DecisionBucket.PICK)
        assert item.metrics is not None
        item.metrics.topiq_score = None
        with TemporaryDirectory() as tmp_dir:
            self.assertIsNone(write_topiq_shadow_report([item], Path(tmp_dir)))

    @staticmethod
    def _item(
        filename: str,
        scene_id: str,
        topiq: float,
        local_rank: float,
        bucket: DecisionBucket,
    ) -> WorkItem:
        path = Path("/photos") / filename
        image = np.full((60, 90, 3), int(topiq * 20), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise AssertionError("failed to build test JPEG")
        asset = RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp"))
        return WorkItem(
            asset=asset,
            scene_id=scene_id,
            metrics=LocalQualityMetrics(
                topiq_score=topiq,
                musiq_score=50.0,
                nima_score=5.0,
                local_rank_score=local_rank,
            ),
            preview=PreviewImage(asset=asset, image_bytes=encoded.tobytes()),
            decision=FinalDecision(
                filename=filename,
                rating=-1 if bucket == DecisionBucket.REJECT else 4,
                label=None,
                bucket=bucket,
                source=DecisionSource.VISION,
                summary="Test decision",
            ),
        )


if __name__ == "__main__":
    unittest.main()
