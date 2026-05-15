from __future__ import annotations

from pathlib import Path
import unittest

from cull_sh.models import ColorLabel
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision
from cull_sh.models import LocalQualityMetrics
from cull_sh.models import RawAsset
from cull_sh.models import WorkItem
from cull_sh.models import WorkStatus
from cull_sh.pipeline import apply_local_rejection
from cull_sh.ranking import build_scene_cohorts
from cull_sh.ranking import rescue_scene_review_candidates
from cull_sh.ranking import sort_candidates_for_vision
from cull_sh.ranking import suppress_duplicates_and_rerank


def _item(
    filename: str,
    scene_id: str,
    rank_score: float,
    perceptual_hash: str,
) -> WorkItem:
    path = Path("/tmp") / filename
    return WorkItem(
        asset=RawAsset(raw_path=path, xmp_path=path.with_suffix(".xmp")),
        scene_id=scene_id,
        status=WorkStatus.READY_FOR_VISION,
        metrics=LocalQualityMetrics(
            blur_score=rank_score,
            tenengrad_score=rank_score / 10.0,
            local_rank_score=rank_score,
            perceptual_hash=perceptual_hash,
        ),
    )


class SceneRankingTests(unittest.TestCase):
    def test_duplicate_suppression_rejects_lower_ranked_match(self) -> None:
        items = [
            _item("A.ARW", "scene-0001", 400.0, "0000000000000000"),
            _item("B.ARW", "scene-0001", 300.0, "0000000000000001"),
        ]

        duplicates_rejected, rerank_rejected = suppress_duplicates_and_rerank(
            items,
            duplicate_hamming_threshold=1,
            max_scene_candidates=None,
            apply_local_rejection=apply_local_rejection,
        )

        self.assertEqual(duplicates_rejected, 1)
        self.assertEqual(rerank_rejected, 0)
        self.assertEqual(items[0].status, WorkStatus.READY_FOR_VISION)
        self.assertEqual(items[1].status, WorkStatus.REJECTED_LOCAL)
        self.assertIsNotNone(items[1].decision)
        assert items[1].decision is not None
        self.assertEqual(items[1].decision.label, ColorLabel.RED)
        self.assertEqual(items[1].decision.source, DecisionSource.LOCAL)

    def test_scene_candidate_cap_rejects_lower_ranked_remainder(self) -> None:
        items = [
            _item("A.ARW", "scene-0001", 500.0, "0000000000000000"),
            _item("B.ARW", "scene-0001", 400.0, "1111111111111111"),
            _item("C.ARW", "scene-0001", 300.0, "2222222222222222"),
        ]

        duplicates_rejected, rerank_rejected = suppress_duplicates_and_rerank(
            items,
            duplicate_hamming_threshold=0,
            max_scene_candidates=2,
            apply_local_rejection=apply_local_rejection,
        )

        self.assertEqual(duplicates_rejected, 0)
        self.assertEqual(rerank_rejected, 1)
        self.assertEqual(items[2].status, WorkStatus.REJECTED_LOCAL)

    def test_sort_candidates_for_vision_orders_by_scene_then_rank(self) -> None:
        first = _item("B.ARW", "scene-0001", 300.0, "0000000000000000")
        second = _item("A.ARW", "scene-0001", 500.0, "1111111111111111")
        third = _item("C.ARW", "scene-0002", 200.0, "2222222222222222")
        ordered = sort_candidates_for_vision([first, second, third])

        self.assertEqual([item.filename for item in ordered], ["A.ARW", "B.ARW", "C.ARW"])

    def test_build_scene_cohorts_keeps_scene_boundaries(self) -> None:
        items = [
            _item("A.ARW", "scene-0001", 500.0, "0000000000000000"),
            _item("B.ARW", "scene-0001", 400.0, "1111111111111111"),
            _item("C.ARW", "scene-0001", 300.0, "2222222222222222"),
            _item("D.ARW", "scene-0002", 200.0, "3333333333333333"),
        ]

        cohorts = build_scene_cohorts(items, cohort_size=2)

        self.assertEqual(
            [[item.filename for item in cohort] for cohort in cohorts],
            [["A.ARW", "B.ARW"], ["C.ARW"], ["D.ARW"]],
        )

    def test_scene_safeguard_restores_best_rejected_candidate(self) -> None:
        first = _item("A.ARW", "scene-0001", 95.0, "0000000000000000")
        second = _item("B.ARW", "scene-0001", 80.0, "1111111111111111")
        apply_local_rejection(first, first.metrics, "too soft")
        apply_local_rejection(second, second.metrics, "too soft")

        rescued = rescue_scene_review_candidates(
            [first, second],
            min_blur_score=110.0,
            min_tenengrad_score=45.0,
        )

        self.assertEqual([item.filename for item in rescued], ["A.ARW"])
        self.assertEqual(first.status, WorkStatus.READY_FOR_VISION)
        self.assertIsNone(first.decision)
        self.assertEqual(second.status, WorkStatus.REJECTED_LOCAL)

    def test_scene_safeguard_skips_truly_bad_scene(self) -> None:
        first = _item("A.ARW", "scene-0001", 20.0, "0000000000000000")
        second = _item("B.ARW", "scene-0001", 10.0, "1111111111111111")
        apply_local_rejection(first, first.metrics, "too soft")
        apply_local_rejection(second, second.metrics, "too soft")

        rescued = rescue_scene_review_candidates(
            [first, second],
            min_blur_score=110.0,
            min_tenengrad_score=45.0,
        )

        self.assertEqual(rescued, [])
        self.assertEqual(first.status, WorkStatus.REJECTED_LOCAL)
        self.assertEqual(second.status, WorkStatus.REJECTED_LOCAL)


if __name__ == "__main__":
    unittest.main()
