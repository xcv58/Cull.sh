from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from cull_sh.backends.base import VisionBackend
from cull_sh.edit_feedback import run_feedback_pilot
from cull_sh.edit_feedback import select_machine_picks
from cull_sh.models import EditReview
from cull_sh.models import EditReviewPair
from cull_sh.models import EditReviewVerdict
from cull_sh.models import EditSuggestion
from cull_sh.models import FinalDecision
from cull_sh.models import PreviewImage


class _FakeBackend(VisionBackend):
    def score_batch(
        self, prompt: str, previews: list[PreviewImage]
    ) -> list[FinalDecision]:
        raise NotImplementedError

    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
        include_crop: bool = False,
    ) -> list[EditSuggestion]:
        return [
            EditSuggestion(
                filename=preview.asset.filename,
                asset_id=str(preview.asset.raw_path),
                exposure=0.2,
                contrast=5,
                highlights=-10,
                shadows=12,
                vibrance=3,
                summary="Open the darker midtones.",
            )
            for preview in previews
        ]

    def review_edits(
        self, prompt: str, pairs: list[EditReviewPair]
    ) -> list[EditReview]:
        return [
            EditReview(
                filename=pair.asset.filename,
                verdict=EditReviewVerdict.REFINE,
                final_suggestion=EditSuggestion(
                    filename=pair.asset.filename,
                    asset_id=str(pair.asset.raw_path),
                    exposure=0.1,
                    contrast=3,
                    highlights=-8,
                    shadows=8,
                    vibrance=2,
                    summary="Use a gentler lift.",
                ),
                summary="The first pass is slightly too bright.",
            )
            for pair in pairs
        ]


class _FailIfCalledBackend(_FakeBackend):
    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
        include_crop: bool = False,
    ) -> list[EditSuggestion]:
        raise AssertionError("completed suggestions should be resumed")

    def review_edits(
        self, prompt: str, pairs: list[EditReviewPair]
    ) -> list[EditReview]:
        raise AssertionError("completed reviews should be resumed")


class EditFeedbackPilotTests(unittest.TestCase):
    def test_selects_machine_picks_not_present_in_human_baseline(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            for stem in ("human", "machine", "review"):
                (source / f"{stem}.ARW").write_bytes(stem.encode())
            _write_xmp(backup / "human.xmp", good="True")
            _write_xmp(backup / "machine.xmp")
            _write_xmp(backup / "review.xmp")
            _write_manifest(
                run,
                source,
                [("human", "pick"), ("machine", "pick"), ("review", "review")],
            )

            selected = select_machine_picks(run, backup, source)

            self.assertEqual([asset.filename for asset in selected], ["machine.ARW"])

    def test_runs_resumable_render_feedback_loop_without_touching_original(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            backup = root / "backup"
            run = root / "run"
            stage = root / "stage"
            source.mkdir()
            backup.mkdir()
            run.mkdir()
            raw = source / "machine.ARW"
            raw.write_bytes(b"original-raw")
            _write_xmp(backup / "machine.xmp")
            _write_manifest(run, source, [("machine", "pick")])
            binary = root / "RapidRAW"
            binary.write_bytes(b"fake")
            assets = select_machine_picks(run, backup, source)

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                output = Path(command[command.index("--output") + 1])
                Image.new("RGB", (32, 24), "gray").save(output)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            result = run_feedback_pilot(
                assets,
                stage,
                binary,
                _FakeBackend(),
                prompt="Natural edits",
                model="qwen",
                cull_run=run,
                human_baseline=backup,
                command_runner=runner,
            )

            self.assertEqual(raw.read_bytes(), b"original-raw")
            self.assertEqual(result.photos, 1)
            self.assertEqual(result.refined, 1)
            self.assertTrue(result.review_page.is_file())
            self.assertTrue((stage / "baseline/machine.jpg").is_file())
            self.assertTrue((stage / "first-pass/machine.jpg").is_file())
            self.assertTrue((stage / "final/machine.jpg").is_file())
            payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["originals_modified"])
            self.assertEqual(payload["maximum_refinements"], 1)
            self.assertEqual(payload["records"][0]["review"]["verdict"], "refine")

            resumed = run_feedback_pilot(
                assets,
                stage,
                binary,
                _FailIfCalledBackend(),
                prompt="Natural edits",
                model="qwen",
                cull_run=run,
                human_baseline=backup,
                command_runner=runner,
            )
            self.assertEqual(resumed.refined, 1)


def _write_manifest(
    run: Path, source: Path, decisions: list[tuple[str, str]]
) -> None:
    with (run / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for stem, bucket in decisions:
            handle.write(
                json.dumps(
                    {
                        "filename": f"{stem}.ARW",
                        "asset_kind": "raw",
                        "raw_path": str(source / f"{stem}.ARW"),
                        "xmp_path": str(source / f"{stem}.xmp"),
                        "decision": {
                            "rating": 4 if bucket == "pick" else 0,
                            "label": "Green" if bucket == "pick" else None,
                            "bucket": bucket,
                            "keep": bucket == "pick",
                            "source": "vision",
                            "summary": "fixture",
                        },
                    }
                )
                + "\n"
            )


def _write_xmp(path: Path, good: str | None = None) -> None:
    good_attribute = f'xmpDM:good="{good}"' if good is not None else ""
    path.write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
  <rdf:RDF><rdf:Description rdf:about="" {good_attribute}/></rdf:RDF>
</x:xmpmeta>
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
