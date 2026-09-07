from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cull_sh.culling_blind import build_frozen_semantic_cohorts
from cull_sh.culling_blind import build_random_folder_cohorts
from cull_sh.culling_blind import score_blind_culling_choices
from cull_sh.culling_blind import score_ground_truth_culling_choices
from cull_sh.culling_blind import write_blind_culling_review
from cull_sh.culling_blind import write_ground_truth_culling_review
from cull_sh.models import RawAsset
from cull_sh.vlm_benchmark import VLMCohort
from cull_sh.vlm_benchmark import VLMModelSpec


class CullingBlindTests(unittest.TestCase):
    def test_random_folder_sample_is_seeded_spaced_and_excludes_completed_folders(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for folder_name in (
                "Trip A",
                "Trip B",
                "Trip C",
                "Trip D DONE",
                "Trip E EXPORTED",
            ):
                folder = root / folder_name
                folder.mkdir()
                for number in range(100, 130):
                    (folder / f"DSC{number:04d}.ARW").write_bytes(b"raw")
                (folder / "._DSC9999.ARW").write_bytes(b"apple-double")

            first = build_random_folder_cohorts(
                root,
                folder_count=2,
                photos_per_folder=2,
                min_sequence_gap=10,
                seed=42,
                excluded_folders=("Trip C",),
            )
            second = build_random_folder_cohorts(
                root,
                folder_count=2,
                photos_per_folder=2,
                min_sequence_gap=10,
                seed=42,
                excluded_folders=("Trip C",),
            )

            self.assertEqual(
                [asset.raw_path for cohort in first for asset in cohort.assets],
                [asset.raw_path for cohort in second for asset in cohort.assets],
            )
            by_folder: dict[str, list[int]] = {}
            for cohort in first:
                filename = cohort.assets[0].filename
                self.assertFalse(filename.startswith("._"))
                by_folder.setdefault(cohort.scene_id, []).append(
                    int(filename.removeprefix("DSC").removesuffix(".ARW"))
                )
            self.assertEqual(set(by_folder), {"Trip A", "Trip B"})
            self.assertTrue(
                all(abs(numbers[0] - numbers[1]) >= 10 for numbers in by_folder.values())
            )

    def test_balanced_total_sample_excludes_previous_raw_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for folder_name in ("Trip A", "Trip B", "Trip C"):
                folder = root / folder_name
                folder.mkdir()
                for number in range(100, 200):
                    (folder / f"DSC{number:04d}.ARW").write_bytes(b"raw")
            excluded = (root / "Trip A" / "DSC0100.ARW").resolve()

            cohorts = build_random_folder_cohorts(
                root,
                folder_count=3,
                photos_per_folder=1,
                total_photos=7,
                min_sequence_gap=10,
                seed=9,
                excluded_raw_paths=frozenset({excluded}),
            )

            counts: dict[str, int] = {}
            selected = []
            for cohort in cohorts:
                counts[cohort.scene_id] = counts.get(cohort.scene_id, 0) + 1
                selected.append(cohort.assets[0].raw_path.resolve())
            self.assertEqual(sorted(counts.values()), [2, 2, 3])
            self.assertEqual(len(cohorts), 7)
            self.assertNotIn(excluded, selected)

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
            self.assertIn("try{Object.assign(saved", html)
            self.assertIn("button.setAttribute('aria-pressed'", html)
            self.assertIn(".join('\\n')", html)
            self.assertNotIn(".join('\n')", html)
            self.assertNotIn("CSS.escape", html)
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

    def test_unavailable_variant_is_visible_but_excluded_from_scoring(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            asset = RawAsset(
                raw_path=run_dir / "DSC0001.ARW",
                xmp_path=run_dir / "DSC0001.xmp",
            )
            cohort = VLMCohort(
                cohort_id="scene-0001-blind-001",
                scene_id="scene-0001",
                assets=(asset,),
                human_labels=("unreviewed",),
            )
            specs = [
                VLMModelSpec(label="gemma4-12b", model="gemma4:12b"),
                VLMModelSpec(label="qwen3-8-27b-nothink", model="qwen:test"),
            ]
            (run_dir / "gemma4-12b.jsonl").write_text(
                json.dumps({"cohort_id": cohort.cohort_id, "status": "failed"}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "qwen3-8-27b-nothink.jsonl").write_text(
                json.dumps(
                    {
                        "cohort_id": cohort.cohort_id,
                        "status": "success",
                        "decisions": [
                            {
                                "filename": asset.filename,
                                "bucket": "review",
                                "rating": 1,
                                "label": "Yellow",
                                "summary": "Usable frame.",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            page, _ = write_blind_culling_review(run_dir, [cohort], specs, seed=1)
            choices_path = run_dir / "choices.csv"
            choices_path.write_text(
                "filename,scene_id,choice\nDSC0001.ARW,scene-0001,A\n",
                encoding="utf-8",
            )
            score = score_blind_culling_choices(run_dir, choices_path)

            self.assertIn("Excluded from preference scoring", page.read_text(encoding="utf-8"))
            self.assertEqual(score["reviewed"], 0)
            self.assertEqual(sum(score["counts"].values()), 0)

    def test_ground_truth_review_hides_outputs_and_scores_human_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            assets = tuple(
                RawAsset(
                    raw_path=run_dir / filename,
                    xmp_path=(run_dir / filename).with_suffix(".xmp"),
                )
                for filename in ("DSC0001.ARW", "DSC0002.ARW", "DSC0003.ARW")
            )
            cohorts = [
                VLMCohort(
                    cohort_id=f"cohort-{index}",
                    scene_id=f"Trip {index}",
                    assets=(asset,),
                    human_labels=("unreviewed",),
                )
                for index, asset in enumerate(assets, start=1)
            ]
            specs = [
                VLMModelSpec(label="gemma4-12b", model="gemma4:12b"),
                VLMModelSpec(label="qwen3-8-27b-nothink", model="qwen:test"),
            ]
            buckets_by_model = {
                "gemma4-12b": ("reject", "review", "pick"),
                "qwen3-8-27b-nothink": ("review", "review", "review"),
            }
            for spec in specs:
                records = []
                for cohort, bucket in zip(
                    cohorts,
                    buckets_by_model[spec.label],
                    strict=True,
                ):
                    records.append(
                        json.dumps(
                            {
                                "cohort_id": cohort.cohort_id,
                                "status": "success",
                                "decisions": [
                                    {
                                        "filename": cohort.assets[0].filename,
                                        "bucket": bucket,
                                        "summary": f"hidden {spec.label} output",
                                    }
                                ],
                            }
                        )
                    )
                (run_dir / f"{spec.label}.jsonl").write_text(
                    "\n".join(records) + "\n",
                    encoding="utf-8",
                )

            page, _ = write_ground_truth_culling_review(run_dir, cohorts, specs)
            html = page.read_text(encoding="utf-8")
            self.assertNotIn("gemma4-12b", html)
            self.assertNotIn("qwen3-8-27b-nothink", html)
            self.assertNotIn("hidden", html)
            self.assertIn("data-choice='reject'", html)
            self.assertIn("data-choice='review'", html)
            self.assertIn("data-choice='pick'", html)
            self.assertIn(".join('\\n')", html)
            choices = run_dir / "ground-truth.csv"
            choices.write_text(
                "filename,scene_id,choice\n"
                "DSC0001.ARW,Trip 1,reject\n"
                "DSC0002.ARW,Trip 2,review\n"
                "DSC0003.ARW,Trip 3,pick\n",
                encoding="utf-8",
            )

            score = score_ground_truth_culling_choices(run_dir, choices)

            self.assertEqual(score["models"]["gemma4-12b"]["exact_matches"], 3)
            self.assertEqual(score["models"]["qwen3-8-27b-nothink"]["exact_matches"], 1)
            self.assertEqual(score["paired"], {"gemma4-12b": 2, "qwen3-8-27b-nothink": 0, "tie": 1})


if __name__ == "__main__":
    unittest.main()
