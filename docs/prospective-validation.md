# Prospective Validation Baseline

## Locked cohort

- Folder: `/Volumes/Sandisk 4T/RAW Photos/2026-05-06 Kuala Lumpur`
- Status at freeze: no `DONE` or `EXPORTED` suffix; human review not final
- Source inventory: 60 RAW, 60 XMP
- Cull.sh commit: `cceb6c850cf8494a68c101320bada2b62e2cb3ea`
- Run: `runs/20260821-145704-238706`
- Mode: dry-run, RAW-only, no learned MUSIQ/NIMA, TOPIQ rank weight 0.25,
  Gemma culling, batch size 1, 1024 maximum output tokens

## Frozen result

| Outcome | Count |
|---|---:|
| Local reject | 47 |
| Vision review | 12 |
| Vision pick | 0 |
| Failed | 1 |
| Metadata writes | 0 |
| TOPIQ review items | 10 |

The failed item is `DSC07689.ARW`: Ollama returned no message content after
three bounded attempts. It remains unscored in the manifest.

## Integrity hashes

```text
050697f89ea40d899cd77abdeba98c34cfeb7f1e3cc58f3ac7a8f7c06dfb6b89  config.json
590cfa137bcb1c3137daeb306b9a2097cffd581cd4d2552202731356ca1b1989  manifest.jsonl
954c6a26d419250541b5189743187779d084b27db112952d13c905b63b8a0438  topiq-shadow.json
15b91d61e7a32e8451eab0ce8026d19cb574afb7c2ba1836d1589d31bcb303f0  topiq-shadow.html
```

After completion, no source `.ARW`, `.xmp`, or `.acr` file was newer than the
run's `config.json`. The run wrote no source metadata by configuration and by
its completion counters.

## Evaluation after human review

Do not rescore or edit the frozen run. Once the folder's human XMP labels are
final, execute:

```bash
python main.py benchmark \
  --path "/Volumes/Sandisk 4T/RAW Photos/2026-05-06 Kuala Lumpur" \
  --shadow-run runs/20260821-145704-238706
```

Judge the combined pipeline on pick/reject AUC, within-scene top-1/top-3,
false-reject safety, and the incremental contribution of TOPIQ. Specifically
audit the 47 local rejects before altering the hard gate.

## Superseded diagnostic

`runs/20260821-144041-340315` is incomplete and not a baseline. Its second
multi-image Gemma cohort streamed for 13 minutes. That run motivated the bounded
Ollama generation fix in commit `cceb6c8`; the successful locked baseline above
was created afterward.
