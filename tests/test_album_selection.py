import json
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from cull_sh.album_selection import resolve_album_duplicates, run_album_selection
from cull_sh.manifests import load_manifest_records
from cull_sh.models import DecisionBucket, DecisionSource, FinalDecision


def jpeg(color):
    output = BytesIO()
    Image.new("RGB", (80, 60), color).save(output, format="JPEG")
    return output.getvalue()


class Backend:
    model = "qwen-test"
    calls = 0

    def select_album_batch(self, prompt, previews):
        self.calls += 1
        return [
            FinalDecision(
                filename=p.asset.filename,
                bucket=DecisionBucket.PICK,
                source=DecisionSource.VISION,
                rating=4,
                label=None,
                summary="Useful distinct travel moment.",
            )
            for p in previews
        ]


class Extractor:
    def extract_preview_bytes(self, path):
        return jpeg("gray")


class AlbumSelectionTests(unittest.TestCase):
    def test_preparation_does_not_call_model_and_can_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen, _, hashes = self.fixtures(root)
            backend = Backend()
            result = run_album_selection(
                frozen,
                source,
                root / "album",
                hashes,
                backend,
                extractor=Extractor(),
                progress=lambda _: None,
                prepare_only=True,
            )
            self.assertEqual(result["photos"], 3)
            self.assertEqual(backend.calls, 0)
            self.assertFalse((root / "album/manifest.jsonl").exists())
            # Simulate a crash between saving a preview and its checksum.
            state_path = root / "album/album-state.json"
            state = json.loads(state_path.read_text())
            del state["preview_hashes"]["DSC00001.ARW"]
            state_path.write_text(json.dumps(state))
            run_album_selection(
                frozen,
                source,
                root / "album",
                hashes,
                backend,
                extractor=Extractor(),
                progress=lambda _: None,
            )
            self.assertEqual(backend.calls, 1)

    def fixtures(self, root):
        source = root / "source"
        source.mkdir()
        frozen = root / "frozen"
        frozen.mkdir()
        rows = []
        hashes = {}
        for i, bucket in enumerate(["review", "reject", "pick"], 1):
            name = f"DSC0000{i}.ARW"
            (source / name).write_bytes(name.encode())
            hashes[name] = sha256(name.encode()).hexdigest()
            rows.append(
                {
                    "filename": name,
                    "raw_path": str(source / name),
                    "scene_id": "scene-1",
                    "combined_rank_score": i / 3,
                    "perceptual_hash": "0000000000000000",
                    "asset_kind": "raw",
                    "decision": {
                        "bucket": bucket,
                        "keep": bucket != "reject",
                        "source": "local",
                        "summary": "old triage",
                        "rating": 0,
                        "label": None,
                    },
                }
            )
        (frozen / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        return source, frozen, rows, hashes

    def test_resolves_review_and_reject_before_dedup_and_preserves_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen, _, hashes = self.fixtures(root)
            backend = Backend()
            before = (frozen / "manifest.jsonl").read_bytes()
            result = run_album_selection(
                frozen,
                source,
                root / "album",
                hashes,
                backend,
                extractor=Extractor(),
                progress=lambda _: None,
            )
            self.assertEqual(result["photos"], 3)
            self.assertEqual(result["selected"], 1)
            self.assertEqual(result["duplicates_represented"], 2)
            records = load_manifest_records(root / "album")
            selected = [
                r["filename"] for r in records if r["decision"]["bucket"] == "pick"
            ]
            self.assertEqual(selected, ["DSC00003.ARW"])
            self.assertEqual(records[0]["album_trace"]["represented_by"], selected[0])
            self.assertEqual(
                records[1]["album_trace"]["contextual_decision"]["bucket"], "pick"
            )
            self.assertEqual(before, (frozen / "manifest.jsonl").read_bytes())
            self.assertEqual(len(list(source.iterdir())), 3)
            run_album_selection(
                frozen,
                source,
                root / "album",
                hashes,
                backend,
                extractor=Extractor(),
                progress=lambda _: None,
            )
            self.assertEqual(backend.calls, 1)

    def test_unresolved_review_does_not_publish_manifest(self):
        class Deferred(Backend):
            def select_album_batch(self, prompt, previews):
                decisions = super().select_album_batch(prompt, previews)
                decisions[0].bucket = DecisionBucket.REVIEW
                return decisions

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen, _, hashes = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "final outcome"):
                run_album_selection(
                    frozen,
                    source,
                    root / "album",
                    hashes,
                    Deferred(),
                    extractor=Extractor(),
                    progress=lambda _: None,
                )
            self.assertFalse((root / "album/manifest.jsonl").exists())

    def test_changed_raw_blocks_reuse_of_cached_scores(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen, _, hashes = self.fixtures(root)
            (source / "DSC00001.ARW").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "RAW hash mismatch"):
                run_album_selection(
                    frozen,
                    source,
                    root / "album",
                    hashes,
                    Backend(),
                    extractor=Extractor(),
                    progress=lambda _: None,
                )

    def test_same_hash_different_color_not_suppressed(self):
        with TemporaryDirectory() as tmp:
            _, _, rows, _ = self.fixtures(Path(tmp))
            for row in rows:
                row["decision"]["bucket"] = "pick"
                row["album_trace"] = {}
            previews = dict(
                zip([r["filename"] for r in rows], map(jpeg, ["red", "green", "blue"]))
            )
            result = resolve_album_duplicates(rows, previews)
            self.assertTrue(all(r["decision"]["bucket"] == "pick" for r in result))

    def test_failed_state_blocks_consuming_old_final_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, frozen, _, hashes = self.fixtures(root)
            run_album_selection(
                frozen,
                source,
                root / "album",
                hashes,
                Backend(),
                extractor=Extractor(),
                progress=lambda _: None,
            )
            path = root / "album/album-state.json"
            state = json.loads(path.read_text())
            state["status"] = "failed"
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_manifest_records(root / "album")
