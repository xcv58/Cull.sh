from __future__ import annotations

import base64
import json
from typing import Literal

import httpx
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from cull_sh.backends.base import VisionBackend
from cull_sh.backends.base import VisionBackendError
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import FinalDecision, PreviewImage


class OllamaDecisionPayload(BaseModel):
    filename: str
    bucket: Literal["reject", "review", "pick"]
    rating: int = Field(ge=0, le=5)
    label: str | None = None
    summary: str = ""


class OllamaBatchDecisionPayload(BaseModel):
    decisions: list[OllamaDecisionPayload]


class OllamaVisionBackend(VisionBackend):
    """
    Local vision backend through Ollama.

    This class is intentionally kept narrow so the pipeline can remain independent of
    Ollama-specific transport and response details.
    """

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def score_batch(
        self,
        prompt: str,
        previews: list[PreviewImage],
    ) -> list[FinalDecision]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            parsed_batch = self._score_cohort(client, prompt, previews)

        returned_by_filename = {decision.filename: decision for decision in parsed_batch.decisions}
        decisions: list[FinalDecision] = []
        for preview in previews:
            try:
                parsed = returned_by_filename[preview.asset.filename]
            except KeyError as exc:
                raise VisionBackendError(
                    f"ollama cohort response omitted filename: {preview.asset.filename}"
                ) from exc

            label = _normalize_label(parsed.label)
            bucket = DecisionBucket(parsed.bucket)
            rating = parsed.rating
            if bucket == DecisionBucket.REJECT and label is None:
                label = ColorLabel.YELLOW
            elif bucket == DecisionBucket.PICK and label is None:
                label = ColorLabel.GREEN

            decisions.append(
                FinalDecision(
                    filename=preview.asset.filename,
                    rating=rating,
                    label=label,
                    bucket=bucket,
                    source=DecisionSource.VISION,
                    summary=parsed.summary.strip(),
                )
            )
        return decisions

    def _score_cohort(
        self,
        client: httpx.Client,
        prompt: str,
        previews: list[PreviewImage],
    ) -> OllamaBatchDecisionPayload:
        cohort_lines = []
        encoded_images = []
        for index, preview in enumerate(previews, start=1):
            cohort_lines.append(f"Image {index}: {preview.asset.filename}")
            encoded_images.append(base64.b64encode(preview.image_bytes).decode("ascii"))

        try:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a photography culling assistant. "
                                "Evaluate this same-scene cohort together. "
                                "Return only valid JSON that matches the provided schema. "
                                "Use the provided filenames exactly and return one result per image. "
                                "Compare the images relative to each other before deciding. "
                                "Use triage, not binary culling: reject, review, or pick."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Scene cohort:\n"
                                + "\n".join(cohort_lines)
                                + "\n\n"
                                + f"User instructions: {prompt}\n"
                                + "Decision policy:\n"
                                + "- reject: obvious miss that does not need human attention\n"
                                + "- review: usable or uncertain image that still needs a human decision\n"
                                + "- pick: clearly one of the strongest images in this cohort and worth extra edit effort\n"
                                + "Use reject sparingly and only for obvious misses. "
                                + "Use pick sparingly and only for clear standouts. "
                                + "Use review as the default middle ground. "
                                + "Do not reject every image in the cohort unless they are all clearly unusable. "
                                + "If nothing is strong enough to pick but one image is still usable, keep the best image as review.\n"
                                + "Return a bucket of reject, review, or pick for each image. "
                                + "Use rating 4 or 5 for picks, and rating 0 for review or reject. "
                                + "Return a Lightroom label from Red, Yellow, Green, Blue, Purple, or null. "
                                + "Use null when unsure."
                            ),
                            "images": encoded_images,
                        },
                    ],
                    "format": OllamaBatchDecisionPayload.model_json_schema(),
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network/service dependent
            raise VisionBackendError(f"ollama request failed: {exc}") from exc

        payload = response.json()
        content = payload.get("message", {}).get("content")
        if not content:
            raise VisionBackendError("ollama response did not include message content")

        try:
            parsed = OllamaBatchDecisionPayload.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise VisionBackendError(f"ollama returned invalid structured output: {exc}") from exc

        if len(parsed.decisions) != len(previews):
            raise VisionBackendError(
                "ollama cohort response did not include one decision per input image"
            )
        return parsed


def _normalize_label(value: str | None) -> ColorLabel | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"", "null", "none"}:
        return None

    mapping = {
        "red": ColorLabel.RED,
        "yellow": ColorLabel.YELLOW,
        "green": ColorLabel.GREEN,
        "blue": ColorLabel.BLUE,
        "purple": ColorLabel.PURPLE,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise VisionBackendError(f"unsupported color label from ollama: {value}") from exc
