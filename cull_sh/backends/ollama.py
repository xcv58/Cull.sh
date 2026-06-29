from __future__ import annotations

import base64
import json
import time
from typing import Callable
from typing import Literal
from typing import TypeVar

import httpx
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

from cull_sh.backends.base import VisionBackend
from cull_sh.backends.base import VisionBackendError
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import EditSuggestion
from cull_sh.models import FinalDecision, PreviewImage


T = TypeVar("T")


class OllamaDecisionPayload(BaseModel):
    id: str = Field(min_length=1)
    filename: str
    bucket: Literal["reject", "review", "pick"]
    rating: int = Field(ge=0, le=5)
    label: str | None = None
    summary: str = ""

    @field_validator("id", "filename")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class OllamaBatchDecisionPayload(BaseModel):
    decisions: list[OllamaDecisionPayload]


class OllamaEditPayload(BaseModel):
    id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    exposure: float = Field(ge=-5.0, le=5.0)
    contrast: int = Field(ge=-100, le=100)
    highlights: int = Field(ge=-100, le=100)
    shadows: int = Field(ge=-100, le=100)
    vibrance: int = Field(ge=-100, le=100)
    summary: str = Field(min_length=1)

    @field_validator("id", "filename", "summary")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class OllamaBatchEditPayload(BaseModel):
    edits: list[OllamaEditPayload]


class OllamaVisionBackend(VisionBackend):
    """
    Local vision backend through Ollama.

    This class is intentionally kept narrow so the pipeline can remain independent of
    Ollama-specific transport and response details.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def score_batch(
        self,
        prompt: str,
        previews: list[PreviewImage],
    ) -> list[FinalDecision]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            parsed_batch = self._score_cohort(client, prompt, previews)

        returned_by_id = {decision.id: decision for decision in parsed_batch.decisions}
        decisions: list[FinalDecision] = []
        for index, preview in enumerate(previews, start=1):
            preview_id = _preview_id(index)
            try:
                parsed = returned_by_id[preview_id]
            except KeyError as exc:
                raise VisionBackendError(
                    f"ollama cohort response omitted image id: {preview_id}"
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
        image_specs = []
        encoded_images = []
        for index, preview in enumerate(previews, start=1):
            image_specs.append(
                {
                    "index": index,
                    "id": _preview_id(index),
                    "filename": preview.asset.filename,
                }
            )
            encoded_images.append(base64.b64encode(preview.image_bytes).decode("ascii"))

        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a photography culling assistant. "
                        "Evaluate this same-scene cohort together. "
                        "Return only one valid JSON object that matches the provided schema. "
                        "Do not use markdown or add commentary. "
                        "Use the provided short ids and filenames exactly and return one result per image. "
                        "Treat the id as the primary key. "
                        "Compare the images relative to each other before deciding. "
                        "Use triage, not binary culling: reject, review, or pick."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Scene cohort:\n"
                        + json.dumps(image_specs, ensure_ascii=False, indent=2)
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
        }

        return self._chat_structured(
            client,
            request_payload,
            lambda payload: _parse_batch_payload(payload, previews),
        )

    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
    ) -> list[EditSuggestion]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            parsed_batch = self._suggest_cohort(client, prompt, previews)

        returned_by_id = {edit.id: edit for edit in parsed_batch.edits}
        suggestions: list[EditSuggestion] = []
        for index, preview in enumerate(previews, start=1):
            preview_id = _preview_id(index)
            try:
                parsed = returned_by_id[preview_id]
            except KeyError as exc:
                raise VisionBackendError(
                    f"ollama edit response omitted image id: {preview_id}"
                ) from exc

            suggestions.append(
                EditSuggestion(
                    filename=preview.asset.filename,
                    asset_id=preview.asset.raw_path.as_posix(),
                    exposure=parsed.exposure,
                    contrast=parsed.contrast,
                    highlights=parsed.highlights,
                    shadows=parsed.shadows,
                    vibrance=parsed.vibrance,
                    summary=parsed.summary.strip(),
                )
            )
        return suggestions

    def _suggest_cohort(
        self,
        client: httpx.Client,
        prompt: str,
        previews: list[PreviewImage],
    ) -> OllamaBatchEditPayload:
        image_specs = []
        encoded_images = []
        for index, preview in enumerate(previews, start=1):
            image_specs.append(
                {
                    "index": index,
                    "id": _preview_id(index),
                    "filename": preview.asset.filename,
                }
            )
            encoded_images.append(base64.b64encode(preview.image_bytes).decode("ascii"))

        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a photo editing assistant for Adobe Lightroom. "
                        "Suggest subtle, image-specific global Develop adjustments. "
                        "Return only one valid JSON object that matches the provided schema. "
                        "Do not use markdown or add commentary. "
                        "Use the provided short ids and filenames exactly and return one result per image. "
                        "Treat the id as the primary key. "
                        "Judge each image on its own merits."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Photos to edit:\n"
                        + json.dumps(image_specs, ensure_ascii=False, indent=2)
                        + "\n\n"
                        + f"User instructions: {prompt}\n"
                        + "Adjustment ranges:\n"
                        + "- exposure: stops, about -5.0 to 5.0, usually -1.0 to 1.0\n"
                        + "- contrast, highlights, shadows, vibrance: -100 to 100\n"
                        + "Editing guidance:\n"
                        + "- Fill every required field for every image.\n"
                        + "- The summary must be one short sentence naming the observed image issue, "
                        + "or \"No global adjustment needed.\" Never leave it empty.\n"
                        + "- Inspect exposure, highlight detail, shadow detail, contrast, and color intensity separately.\n"
                        + "- Recover blown skies with negative highlights; open dark areas with positive shadows.\n"
                        + "- Lift or lower exposure only when the image is clearly under- or over-exposed.\n"
                        + "- Keep edits subtle and realistic unless the user asks for a stronger look.\n"
                        + "- Use 0 only when that slider already looks correct for that specific image.\n"
                        + "- Do not copy identical slider values across images unless the summaries explain the same observed issue.\n"
                        + "Return one set of adjustments per image."
                    ),
                    "images": encoded_images,
                },
            ],
            "format": OllamaBatchEditPayload.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0},
        }

        return self._chat_structured(
            client,
            request_payload,
            lambda payload: _parse_edit_payload(payload, previews),
        )

    def _chat_structured(
        self,
        client: httpx.Client,
        request_payload: dict[str, object],
        parse_fn: Callable[[dict[str, object]], T],
    ) -> T:
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=request_payload,
                )
                response.raise_for_status()
            except httpx.TimeoutException:  # pragma: no cover - network/service dependent
                last_error = (
                    f"ollama request timed out after {self.timeout_seconds:.0f}s"
                )
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network/service dependent
                status_code = exc.response.status_code
                if 500 <= status_code < 600:
                    last_error = f"ollama request failed with {status_code}"
                else:
                    raise VisionBackendError(f"ollama request failed: {exc}") from exc
            except httpx.HTTPError as exc:  # pragma: no cover - network/service dependent
                raise VisionBackendError(f"ollama request failed: {exc}") from exc
            else:
                try:
                    payload = response.json()
                    return parse_fn(payload)
                except VisionBackendError as exc:
                    last_error = str(exc)

            if attempt == self.max_attempts:
                break
            time.sleep(self.retry_backoff_seconds * attempt)

        raise VisionBackendError(
            f"{last_error or 'ollama request failed'} after {self.max_attempts} attempt(s)"
        )


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


def _preview_id(index: int) -> str:
    return f"image-{index}"


def _message_content(payload: dict[str, object]) -> str:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise VisionBackendError("ollama response did not include message content")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise VisionBackendError("ollama response did not include message content")
    return content


def _load_json_object(content: str) -> dict[str, object]:
    stripped = content.strip()
    decoder = json.JSONDecoder()
    first_error: json.JSONDecodeError | None = None
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc
            continue
        if not isinstance(parsed, dict):
            raise VisionBackendError("ollama structured output root was not a JSON object")
        return parsed

    if first_error is not None:
        raise VisionBackendError(
            f"ollama returned invalid structured output: {first_error}"
        ) from first_error
    raise VisionBackendError("ollama returned invalid structured output: no JSON object found")


def _parse_batch_payload(
    payload: dict[str, object],
    previews: list[PreviewImage],
) -> OllamaBatchDecisionPayload:
    content = _message_content(payload)

    try:
        parsed = OllamaBatchDecisionPayload.model_validate(_load_json_object(content))
    except ValidationError as exc:
        raise VisionBackendError(
            f"ollama returned invalid structured output: {exc}"
        ) from exc

    if len(parsed.decisions) != len(previews):
        raise VisionBackendError(
            "ollama cohort response did not include one decision per input image"
        )

    expected_by_id = {
        _preview_id(index): preview.asset.filename
        for index, preview in enumerate(previews, start=1)
    }
    returned_ids = [decision.id for decision in parsed.decisions]
    if set(returned_ids) != set(expected_by_id):
        raise VisionBackendError(
            "ollama cohort response did not include the expected image ids"
        )
    for decision in parsed.decisions:
        if decision.filename != expected_by_id[decision.id]:
            raise VisionBackendError(
                "ollama cohort response did not include the expected filenames"
            )

    return parsed


def _parse_edit_payload(
    payload: dict[str, object],
    previews: list[PreviewImage],
) -> OllamaBatchEditPayload:
    content = _message_content(payload)

    try:
        parsed = OllamaBatchEditPayload.model_validate(_load_json_object(content))
    except ValidationError as exc:
        raise VisionBackendError(
            f"ollama returned invalid structured output: {exc}"
        ) from exc

    if len(parsed.edits) != len(previews):
        raise VisionBackendError(
            "ollama edit response did not include one result per input image"
        )

    expected_by_id = {
        _preview_id(index): preview.asset.filename
        for index, preview in enumerate(previews, start=1)
    }
    returned_ids = [edit.id for edit in parsed.edits]
    if set(returned_ids) != set(expected_by_id):
        raise VisionBackendError(
            "ollama edit response did not include the expected image ids"
        )
    for edit in parsed.edits:
        if edit.filename != expected_by_id[edit.id]:
            raise VisionBackendError(
                "ollama edit response did not include the expected filenames"
            )

    return parsed
