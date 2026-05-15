from __future__ import annotations

import unittest

import cv2
import numpy as np

from cull_sh.models import LocalQualityMetrics
from cull_sh.quality import analyze_local_quality
from cull_sh.quality import build_local_decision_trace
from cull_sh.quality import should_reject_for_local_quality


class LocalQualityTests(unittest.TestCase):
    def test_analyze_local_quality_returns_multiple_metrics(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[:, :40] = 255
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        metrics = analyze_local_quality(encoded.tobytes(), enable_learned_iqa=False)

        self.assertIsNotNone(metrics.blur_score)
        self.assertIsNotNone(metrics.tenengrad_score)
        self.assertIsNotNone(metrics.brightness_mean)
        self.assertIsNotNone(metrics.contrast_stddev)
        self.assertIsNotNone(metrics.local_rank_score)
        self.assertIsNotNone(metrics.perceptual_hash)

    def test_should_reject_for_local_quality_requires_both_sharpness_metrics(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[:, :40] = 255
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        metrics = analyze_local_quality(encoded.tobytes(), enable_learned_iqa=False)

        self.assertFalse(
            should_reject_for_local_quality(
                metrics,
                min_blur_score=metrics.blur_score - 1.0,
                min_tenengrad_score=metrics.tenengrad_score + 1000.0,
                min_musiq_score=45.0,
                min_nima_score=4.5,
                max_brisque_score=55.0,
                min_cpbd_score=0.3,
                use_brisque_for_reject=False,
                use_cpbd_for_reject=False,
                local_reject_required_support_votes=2,
            )
        )
        self.assertTrue(
            should_reject_for_local_quality(
                metrics,
                min_blur_score=metrics.blur_score + 1000.0,
                min_tenengrad_score=metrics.tenengrad_score + 1000.0,
                min_musiq_score=45.0,
                min_nima_score=4.5,
                max_brisque_score=55.0,
                min_cpbd_score=0.3,
                use_brisque_for_reject=False,
                use_cpbd_for_reject=False,
                local_reject_required_support_votes=2,
            )
        )

    def test_should_reject_for_local_quality_requires_all_available_signals_to_be_weak(self) -> None:
        metrics = analyze_local_quality(
            cv2.imencode(".jpg", np.zeros((80, 80, 3), dtype=np.uint8))[1].tobytes(),
            enable_learned_iqa=False,
        )
        metrics.musiq_score = 60.0
        metrics.nima_score = 5.0

        self.assertFalse(
            should_reject_for_local_quality(
                metrics,
                min_blur_score=metrics.blur_score + 1000.0,
                min_tenengrad_score=metrics.tenengrad_score + 1000.0,
                min_musiq_score=45.0,
                min_nima_score=4.5,
                max_brisque_score=55.0,
                min_cpbd_score=0.3,
                use_brisque_for_reject=False,
                use_cpbd_for_reject=False,
                local_reject_required_support_votes=2,
            )
        )

    def test_build_local_decision_trace_records_votes_and_explanation(self) -> None:
        metrics = LocalQualityMetrics(
            blur_score=5.0,
            tenengrad_score=10.0,
            musiq_score=40.0,
            nima_score=4.0,
            brisque_score=70.0,
            cpbd_score=0.2,
            local_rank_score=123.0,
        )

        trace = build_local_decision_trace(
            metrics,
            min_blur_score=110.0,
            min_tenengrad_score=45.0,
            min_musiq_score=50.0,
            min_nima_score=4.6,
            max_brisque_score=55.0,
            min_cpbd_score=0.3,
            use_brisque_for_reject=True,
            use_cpbd_for_reject=True,
            local_reject_required_support_votes=2,
        )

        self.assertTrue(trace["quality_reject"])
        self.assertEqual(trace["weak_support_count"], 4)
        self.assertIn("scores", trace)
        self.assertIn("quality_explanation", trace)


if __name__ == "__main__":
    unittest.main()
