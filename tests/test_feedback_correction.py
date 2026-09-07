import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from cull_sh.edit_feedback import run_feedback_pilot
from cull_sh.feedback_correction import prepare_correction_stage, run_correction
from cull_sh.models import EditReview, EditReviewVerdict, EditSuggestion, RawAsset


class Backend:
    def __init__(self, fail=True):
        self.fail = fail
        self.calls = []

    def suggest_edits(self, prompt, previews, **kwargs):
        return [EditSuggestion(filename=p.asset.filename, asset_id=str(p.asset.raw_path)) for p in previews]

    def review_edits(self, prompt, pairs):
        self.calls.append((prompt, pairs))
        return [EditReview(filename=p.asset.filename,
                           verdict=(EditReviewVerdict.REFINE if self.fail and p.delivery_check
                                    and p.asset.filename == "a.ARW" else EditReviewVerdict.ACCEPT),
                           final_suggestion=p.suggestion, summary="Too dark" if self.fail else "Looks natural")
                for p in pairs]


class CorrectionBackend(Backend):
    def review_edits(self, prompt, pairs):
        result = super().review_edits(prompt, pairs)
        if not pairs[0].delivery_check:
            result[0].verdict = EditReviewVerdict.REFINE
            result[0].final_suggestion = replace(pairs[0].suggestion, brightness=0.5)
        return result


class FeedbackCorrectionTests(unittest.TestCase):
    def setup_parent(self, root):
        source = root / "source"
        source.mkdir()
        assets = []
        for name in ("a.ARW", "b.ARW"):
            raw = source / name
            raw.write_bytes(name.encode())
            assets.append(RawAsset(raw, raw.with_suffix(".xmp")))
        binary = root / "RapidRAW"
        binary.write_bytes(b"fake binary")
        binary.chmod(0o755)

        def runner(command):
            Image.new("RGB", (80, 60), "gray").save(Path(command[command.index("--output") + 1]))
            return subprocess.CompletedProcess(command, 0, "", "")

        parent = root / "parent"
        with self.assertRaisesRegex(RuntimeError, "delivery quality check failed"):
            run_feedback_pilot(assets, parent, binary, Backend(), prompt="natural daylight",
                               model="qwen", cull_run=root / "cull", human_baseline=root / "backup",
                               command_runner=runner)
        return parent, binary, runner

    def test_only_failed_photo_gets_correction_and_unbiased_final_check(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, binary, runner = self.setup_parent(root)
            parent_files = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
            original = json.loads(parent_files[parent / "feedback-manifest.json"])
            backend = CorrectionBackend(fail=False)
            stage = root / "correction"
            result = run_correction(parent, stage, binary, backend, reason="repair daylight",
                                    command_runner=runner, progress=lambda _: None)
            saved = json.loads(result.manifest_path.read_text())
            self.assertEqual(len(backend.calls), 2)
            self.assertIn("previous_failure", backend.calls[0][0])
            self.assertEqual(backend.calls[1][0], "natural daylight")
            self.assertTrue(backend.calls[1][1][0].delivery_check)
            self.assertEqual(backend.calls[0][1][0].asset.filename, "a.ARW")
            passed = saved["records"][1]
            self.assertEqual(passed["final_suggestion"], original["records"][1]["final_suggestion"])
            self.assertEqual(passed["delivery_validation"], original["records"][1]["delivery_validation"])
            failed = saved["records"][0]
            self.assertEqual(failed["prior_experiment_record"]["delivery_validation"]["status"], "failed")
            self.assertEqual(failed["delivery_validation"]["status"], "passed")
            self.assertEqual(Path(failed["staged_raw"]).parent, (stage / "input").resolve())
            self.assertEqual(parent_files, {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()})
            backend.calls.clear()
            run_correction(parent, stage, binary, backend, reason="repair daylight",
                           command_runner=runner, progress=lambda _: None)
            self.assertEqual(backend.calls, [])

    def test_failed_correction_stays_failed_on_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, binary, runner = self.setup_parent(root)
            backend = CorrectionBackend(fail=True)
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "delivery quality check failed"):
                    run_correction(parent, root / "correction", binary, backend, reason="repair",
                                   command_runner=runner, progress=lambda _: None)
            self.assertEqual(len(backend.calls), 2)

    def test_changed_parent_pixels_or_reason_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, binary, _ = self.setup_parent(root)
            stage = root / "correction"
            prepare_correction_stage(parent, stage, binary, reason="repair")
            with self.assertRaisesRegex(ValueError, "reason or policy changed"):
                prepare_correction_stage(parent, stage, binary, reason="another reason")
            Image.new("RGB", (80, 60), "red").save(parent / "final/b.jpg")
            with self.assertRaisesRegex(ValueError, "parent final_render changed"):
                prepare_correction_stage(parent, stage, binary, reason="repair")

    def test_overlapping_or_untracked_output_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, binary, _ = self.setup_parent(root)
            with self.assertRaisesRegex(ValueError, "separate trees"):
                prepare_correction_stage(parent, parent / "nested", binary, reason="repair")
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "keep.txt").write_text("user data")
            with self.assertRaisesRegex(ValueError, "nonempty untracked"):
                prepare_correction_stage(parent, occupied, binary, reason="repair")
            self.assertEqual((occupied / "keep.txt").read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
