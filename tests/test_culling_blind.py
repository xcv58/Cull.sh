from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.culling_blind import build_frozen_semantic_cohorts
from cull_sh.culling_blind import score_blind_culling_choices
from cull_sh.culling_blind import write_blind_culling_review
from cull_sh.models import RawAsset
from cull_sh.vlm_benchmark import VLMCohort
from cull_sh.vlm_benchmark import VLMModelSpec


class CullingBlindTests(unittest.TestCase):
    def test_frozen_cohorts_include_only_vision_candidates_without_reading_xmp(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frozen_run = root / "frozen"
            frozen_run.mkdir()
            records = []
            for index, source in enumerate(("vision", "failure", "local"), start=1):
                raw_path = root / f"DSC{index:04d}.ARW"
                raw_path.write_bytes(b"raw")
                record = {
                    "raw_path": str(raw_path),
                    "scene_id": "scene-0001",
                    "scene_index": index,
                    "decision": (
                        {"source": "vision", "bucket": "review"}
                        if source == "vision"
                        else {"source": "local", "bucket": "reject"}
                        if source == "local"
                        else None
                    ),
                    "error": (
                        "vision scoring failed: timeout" if source == "failure" else None
                    ),
                }
                records.append(record)
            (frozen_run / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            cohorts = build_frozen_semantic_cohorts(frozen_run, batch_size=1)

            self.assertEqual(len(cohorts), 2)
            self.assertEqual(
                [cohort.assets[0].filename for cohort in cohorts],
                ["DSC0001.ARW", "DSC0002.ARW"],
            )
            self.assertFalse(any(path.suffix == ".xmp" for path in root.iterdir()))

    def test_blind_review_hides_assignment_and_scores_exported_choices(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            assets = tuple(
                RawAsset(
                    raw_path=run_dir / filename,
                    xmp_path=(run_dir / filename).with_suffix(".xmp"),
                )
                for filename in ("DSC0001.ARW", "DSC0002.ARW")
            )
            cohort = VLMCohort(
                cohort_id="scene-0001-blind-001",
                scene_id="scene-0001",
                assets=assets,
                human_labels=("unreviewed", "unreviewed"),
            )
            specs = [
                VLMModelSpec(label="gemma4-12b", model="gemma4:12b"),
                VLMModelSpec(label="qwen3-8-27b-nothink", model="qwen:test", think=False),
            ]
            for spec, bucket in zip(specs, ("review", "pick"), strict=True):
                path = run_dir / f"{spec.label}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "cohort_id": cohort.cohort_id,
                            "status": "success",
                            "decisions": [
                                {
                                    "filename": asset.filename,
                                    "bucket": bucket,
                                    "rating": 1,
                                    "label": "Yellow",
                                    "summary": f"{bucket} explanation",
                                }
                                for asset in assets
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            page, answer_key_path = write_blind_culling_review(
                run_dir,
                [cohort],
                specs,
                seed=7,
            )

            html = page.read_text(encoding="utf-8")
            self.assertIn("Variant A", html)
            self.assertIn("Variant B", html)
            self.assertNotIn("gemma4-12b", html)
            self.assertNotIn("qwen3-8-27b-nothink", html)
            answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
            choices_path = run_dir / "choices.csv"
            with choices_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("filename", "scene_id", "choice"))
                writer.writeheader()
                first, second = (asset.filename for asset in assets)
                writer.writerow({"filename": first, "scene_id": "scene-0001", "choice": "A"})
                writer.writerow({"filename": second, "scene_id": "scene-0001", "choice": "tie"})

            score = score_blind_culling_choices(run_dir, choices_path)

            winner = answer_key["assignments"][first]["A"]
            self.assertEqual(score["reviewed"], 2)
            self.assertEqual(score["counts"][winner], 1)
            self.assertEqual(score["counts"]["tie"], 1)


if __name__ == "__main__":
    unittest.main()
