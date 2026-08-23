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
from pydantic import model_validator

from cull_sh.backends.base import VisionBackend
from cull_sh.backends.base import VisionBackendError
from cull_sh.models import ColorLabel
from cull_sh.models import DecisionBucket
from cull_sh.models import DecisionSource
from cull_sh.models import EditReview
from cull_sh.models import EditReviewPair
from cull_sh.models import EditReviewVerdict
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
    brightness: float = Field(default=0.0, ge=-5.0, le=5.0)
    contrast: int = Field(ge=-100, le=100)
    highlights: int = Field(ge=-100, le=100)
    shadows: int = Field(ge=-100, le=100)
    whites: int = Field(default=0, ge=-100, le=100)
    blacks: int = Field(default=0, ge=-100, le=100)
    temperature: int = Field(default=0, ge=-100, le=100)
    tint: int = Field(default=0, ge=-100, le=100)
    vibrance: int = Field(ge=-100, le=100)
    saturation: int = Field(default=0, ge=-100, le=100)
    clarity: int = Field(default=0, ge=-100, le=100)
    dehaze: int = Field(default=0, ge=-100, le=100)
    structure: int = Field(default=0, ge=-100, le=100)
    sharpness: int = Field(default=0, ge=-100, le=100)
    luma_noise_reduction: int = Field(default=0, ge=0, le=100)
    color_noise_reduction: int = Field(default=0, ge=0, le=100)
    vignette_amount: int = Field(default=0, ge=-100, le=100)
    additional_edits: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)

    @field_validator("id", "filename", "summary")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("additional_edits")
    @classmethod
    def _normalize_additional_edits(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class OllamaEditWithCropPayload(OllamaEditPayload):
    has_crop: bool
    crop_left: float = Field(ge=0.0, le=1.0)
    crop_top: float = Field(ge=0.0, le=1.0)
    crop_right: float = Field(ge=0.0, le=1.0)
    crop_bottom: float = Field(ge=0.0, le=1.0)
    crop_angle: float = Field(ge=-45.0, le=45.0)

    @model_validator(mode="after")
    def _validate_crop_rectangle(self) -> "OllamaEditWithCropPayload":
        if not self.has_crop:
            if self.crop_angle != 0.0:
                raise ValueError("rotation requires a crop that removes rotated edges")
            self.crop_left = 0.0
            self.crop_top = 0.0
            self.crop_right = 1.0
            self.crop_bottom = 1.0
            return self
        if self.crop_left >= self.crop_right:
            raise ValueError("crop_left must be less than crop_right")
        if self.crop_top >= self.crop_bottom:
            raise ValueError("crop_top must be less than crop_bottom")
        width = self.crop_right - self.crop_left
        height = self.crop_bottom - self.crop_top
        if width < 0.2 or height < 0.2:
            raise ValueError("crop rectangle is too small")
        if (
            self.crop_left == 0.0
            and self.crop_top == 0.0
            and self.crop_right == 1.0
            and self.crop_bottom == 1.0
        ):
            if self.crop_angle != 0.0:
                raise ValueError("rotation requires non-default crop bounds")
            self.has_crop = False
        return self


class OllamaBatchEditPayload(BaseModel):
    edits: list[OllamaEditPayload]


class OllamaBatchEditWithCropPayload(BaseModel):
    edits: list[OllamaEditWithCropPayload]


class OllamaEditReviewPayload(OllamaEditWithCropPayload):
    verdict: Literal["accept", "refine", "reject"]


class OllamaBatchEditReviewPayload(BaseModel):
    reviews: list[OllamaEditReviewPayload]


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
        temperature: float = 0.0,
        think: bool | str | None = None,
        max_output_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.temperature = temperature
        self.think = think
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self.max_output_tokens = max_output_tokens

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
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        if self.think is not None:
            request_payload["think"] = self.think

        return self._chat_structured(
            client,
            request_payload,
            lambda payload: _parse_batch_payload(payload, previews),
        )

    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
        include_crop: bool = False,
    ) -> list[EditSuggestion]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            parsed_batch = self._suggest_cohort(
                client,
                prompt,
                previews,
                include_crop=include_crop,
            )

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
                    brightness=parsed.brightness,
                    contrast=parsed.contrast,
                    highlights=parsed.highlights,
                    shadows=parsed.shadows,
                    whites=parsed.whites,
                    blacks=parsed.blacks,
                    temperature=parsed.temperature,
                    tint=parsed.tint,
                    vibrance=parsed.vibrance,
                    saturation=parsed.saturation,
                    clarity=parsed.clarity,
                    dehaze=parsed.dehaze,
                    structure=parsed.structure,
                    sharpness=parsed.sharpness,
                    luma_noise_reduction=parsed.luma_noise_reduction,
                    color_noise_reduction=parsed.color_noise_reduction,
                    vignette_amount=parsed.vignette_amount,
                    has_crop=getattr(parsed, "has_crop", False),
                    crop_left=getattr(parsed, "crop_left", 0.0),
                    crop_top=getattr(parsed, "crop_top", 0.0),
                    crop_right=getattr(parsed, "crop_right", 1.0),
                    crop_bottom=getattr(parsed, "crop_bottom", 1.0),
                    crop_angle=getattr(parsed, "crop_angle", 0.0),
                    additional_edits=list(parsed.additional_edits),
                    summary=parsed.summary.strip(),
                )
            )
        return suggestions

    def _suggest_cohort(
        self,
        client: httpx.Client,
        prompt: str,
        previews: list[PreviewImage],
        include_crop: bool = False,
    ) -> OllamaBatchEditPayload | OllamaBatchEditWithCropPayload:
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

        system_crop_instruction = (
            " When composition fields are present in the schema, independently "
            "evaluate crop and rotation as first-class optional edits."
            if include_crop
            else ""
        )
        crop_guidance = (
            "Composition/crop/rotation pass:\n"
            "- For every image, actively check whether a crop would improve composition before deciding has_crop.\n"
            "- Use has_crop true when cropping or leveling addresses a specific visible issue: empty edge space, a partial distraction, weak subject placement, clutter, imbalance, or a tilted horizon.\n"
            "- Use crop_angle for clear horizon or architectural leveling only with has_crop true.\n"
            "- Every nonzero crop_angle must include non-default crop bounds that remove the rotated edges.\n"
            "- Crop as much or as little as the visible issue warrants while preserving important subjects, landmarks, heads, limbs, reflections, and useful context.\n"
            "- Set has_crop false for already well-framed images or when the crop reason is weak.\n"
            "- Do not add a generic inset crop just because crop fields are available; vary bounds only to match the visible issue in that image.\n"
            "- Crop bounds are normalized: crop_left/top/right/bottom are between 0 and 1.\n"
            "- If no crop or rotation is needed, set has_crop false, crop_left 0, crop_top 0, crop_right 1, crop_bottom 1, crop_angle 0.\n"
            "- If a crop or rotation is used, the summary must mention the specific composition or leveling reason.\n"
        )
        payload_schema = (
            OllamaBatchEditWithCropPayload.model_json_schema()
            if include_crop
            else OllamaBatchEditPayload.model_json_schema()
        )

        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a photo editing assistant generating reversible RapidRAW recipes. "
                        "Diagnose each image before suggesting natural, image-specific adjustments. "
                        "Return only one valid JSON object that matches the provided schema. "
                        "Do not use markdown or add commentary. "
                        "Use the provided short ids and filenames exactly and return one result per image. "
                        "Treat the id as the primary key. "
                        "Judge each image on its own merits."
                        + system_crop_instruction
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
                        + "- exposure and brightness: stops, -5.0 to 5.0, usually -1.0 to 1.0\n"
                        + "- signed integer controls: -100 to 100\n"
                        + "- luma_noise_reduction and color_noise_reduction: 0 to 100\n"
                        + "Editing guidance:\n"
                        + "- Fill every required field for every image.\n"
                        + "- First diagnose tonal balance, white balance/color cast, presence, detail/noise, and composition independently.\n"
                        + "- The summary must be one short, image-specific diagnostic sentence about the starting render. Do not narrate slider actions or claim that an adjustment was applied. Never leave it empty.\n"
                        + "- Inspect exposure, brightness, highlight and shadow detail, whites, blacks, contrast, temperature, tint, vibrance, and saturation separately.\n"
                        + "- exposure is a linear RAW EV shift; brightness is a filmic perceptual exposure control. Prefer one for the diagnosed need and move both only when their distinct roles are necessary.\n"
                        + "- Recover blown skies with negative highlights; open dark areas with positive shadows.\n"
                        + "- Lift or lower exposure when it improves the overall tonal balance.\n"
                        + "- Use clarity, dehaze, and structure only for a specific visible presence problem.\n"
                        + "- Use sharpening or noise reduction only when the available render provides enough evidence; otherwise leave those fields at zero and record a full-resolution inspection in additional_edits.\n"
                        + "- Keep edits realistic unless the user asks for a stronger look.\n"
                        + "- Use 0 only when that slider already looks correct for that specific image.\n"
                        + "- additional_edits is for useful edits outside the executable fields, including HSL, curves, color grading, masks, healing, lens corrections, or other local work. Use concise, actionable descriptions and never pretend they were applied.\n"
                        + "- Do not copy identical slider values across images unless the summaries explain the same observed issue.\n"
                        + (crop_guidance if include_crop else "")
                        + "Return one set of adjustments per image."
                    ),
                    "images": encoded_images,
                },
            ],
            "format": payload_schema,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        if self.think is not None:
            request_payload["think"] = self.think

        return self._chat_structured(
            client,
            request_payload,
            lambda payload: _parse_edit_payload(
                payload,
                previews,
                include_crop=include_crop,
            ),
        )

    def review_edits(
        self,
        prompt: str,
        pairs: list[EditReviewPair],
    ) -> list[EditReview]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            parsed_batch = self._review_edit_cohort(client, prompt, pairs)

        returned_by_id = {review.id: review for review in parsed_batch.reviews}
        reviews: list[EditReview] = []
        for index, pair in enumerate(pairs, start=1):
            review_id = _preview_id(index)
            try:
                parsed = returned_by_id[review_id]
            except KeyError as exc:
                raise VisionBackendError(
                    f"ollama edit review omitted image id: {review_id}"
                ) from exc

            verdict = EditReviewVerdict(parsed.verdict)
            if verdict == EditReviewVerdict.ACCEPT:
                final_suggestion = pair.suggestion
            elif verdict == EditReviewVerdict.REJECT:
                final_suggestion = EditSuggestion(
                    filename=pair.asset.filename,
                    asset_id=pair.asset.raw_path.as_posix(),
                    additional_edits=(
                        list(parsed.additional_edits)
                        if parsed.additional_edits
                        else list(pair.suggestion.additional_edits)
                    ),
                    summary="Reverted to the neutral RapidRAW baseline.",
                )
            else:
                final_suggestion = _bounded_refinement(pair, parsed)
            reviews.append(
                EditReview(
                    filename=pair.asset.filename,
                    verdict=verdict,
                    final_suggestion=final_suggestion,
                    summary=parsed.summary.strip(),
                )
            )
        return reviews

    def _review_edit_cohort(
        self,
        client: httpx.Client,
        prompt: str,
        pairs: list[EditReviewPair],
    ) -> OllamaBatchEditReviewPayload:
        image_specs: list[dict[str, object]] = []
        encoded_images: list[str] = []
        for index, pair in enumerate(pairs, start=1):
            review_id = _preview_id(index)
            image_specs.append(
                {
                    "id": review_id,
                    "filename": pair.asset.filename,
                    "image_order": ["baseline", "edited"],
                    "current_adjustments": _edit_payload(pair.suggestion),
                }
            )
            encoded_images.extend(
                [
                    base64.b64encode(pair.baseline_bytes).decode("ascii"),
                    base64.b64encode(pair.edited_bytes).decode("ascii"),
                ]
            )

        request_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are independently validating real photo edit renders. Each record has exactly "
                        "two consecutive images: the neutral RapidRAW baseline followed by the "
                        "RapidRAW render of the current adjustments. Return only valid JSON matching "
                        "the schema, with one review per record. Judge the rendered result rather than "
                        "defending the first recipe."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Edit pairs:\n"
                        + json.dumps(image_specs, ensure_ascii=False, indent=2)
                        + "\n\n"
                        + f"User instructions: {prompt}\n"
                        + "Compare each edited render only with its paired baseline. Diagnose the most important visible difference before choosing a verdict. "
                        + "Use accept only when the edit is a meaningful natural improvement without a new visible problem. "
                        + "Use reject when the neutral baseline is better or the edit has no meaningful benefit. "
                        + "Use refine only to correct a specific visible issue introduced or left by the edit. "
                        + "additional_edits are unrendered future-work notes: never count them as a visible improvement, but preserve or improve useful notes even when rejecting the executable recipe. "
                        + "For refine, return final absolute slider and crop values, not deltas. "
                        + "Keep refinements conservative: exposure and brightness within 0.5 stop, rotation within 5 degrees, and each integer slider "
                        + "within 30 points of the current recipe. Preserve or revise additional_edits based on visible evidence. Do not invent a stylistic change merely "
                        + "to make the values different. Fill every field and explain the verdict briefly."
                    ),
                    "images": encoded_images,
                },
            ],
            "format": OllamaBatchEditReviewPayload.model_json_schema(),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        if self.think is not None:
            request_payload["think"] = self.think
        return self._chat_structured(
            client,
            request_payload,
            lambda payload: _parse_edit_review_payload(payload, pairs),
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
    include_crop: bool = False,
) -> OllamaBatchEditPayload | OllamaBatchEditWithCropPayload:
    content = _message_content(payload)

    try:
        payload_model = (
            OllamaBatchEditWithCropPayload
            if include_crop
            else OllamaBatchEditPayload
        )
        parsed = payload_model.model_validate(_load_json_object(content))
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


def _parse_edit_review_payload(
    payload: dict[str, object],
    pairs: list[EditReviewPair],
) -> OllamaBatchEditReviewPayload:
    content = _message_content(payload)
    try:
        parsed = OllamaBatchEditReviewPayload.model_validate(
            _load_json_object(content)
        )
    except ValidationError as exc:
        raise VisionBackendError(
            f"ollama returned invalid structured edit review: {exc}"
        ) from exc

    if len(parsed.reviews) != len(pairs):
        raise VisionBackendError(
            "ollama edit review did not include one result per input pair"
        )
    expected_by_id = {
        _preview_id(index): pair.asset.filename
        for index, pair in enumerate(pairs, start=1)
    }
    returned_ids = [review.id for review in parsed.reviews]
    if set(returned_ids) != set(expected_by_id):
        raise VisionBackendError(
            "ollama edit review did not include the expected image ids"
        )
    for review in parsed.reviews:
        if review.filename != expected_by_id[review.id]:
            raise VisionBackendError(
                "ollama edit review did not include the expected filenames"
            )
    return parsed


def _edit_payload(suggestion: EditSuggestion) -> dict[str, object]:
    return {
        "exposure": suggestion.exposure,
        "brightness": suggestion.brightness,
        "contrast": suggestion.contrast,
        "highlights": suggestion.highlights,
        "shadows": suggestion.shadows,
        "whites": suggestion.whites,
        "blacks": suggestion.blacks,
        "temperature": suggestion.temperature,
        "tint": suggestion.tint,
        "vibrance": suggestion.vibrance,
        "saturation": suggestion.saturation,
        "clarity": suggestion.clarity,
        "dehaze": suggestion.dehaze,
        "structure": suggestion.structure,
        "sharpness": suggestion.sharpness,
        "luma_noise_reduction": suggestion.luma_noise_reduction,
        "color_noise_reduction": suggestion.color_noise_reduction,
        "vignette_amount": suggestion.vignette_amount,
        "has_crop": suggestion.has_crop,
        "crop_left": suggestion.crop_left,
        "crop_top": suggestion.crop_top,
        "crop_right": suggestion.crop_right,
        "crop_bottom": suggestion.crop_bottom,
        "crop_angle": suggestion.crop_angle,
        "additional_edits": suggestion.additional_edits,
    }


def _bounded_refinement(
    pair: EditReviewPair,
    parsed: OllamaEditReviewPayload,
) -> EditSuggestion:
    current = pair.suggestion

    def bounded_float(value: float, original: float, delta: float) -> float:
        return max(original - delta, min(original + delta, value))

    def bounded_int(value: int, original: int, delta: int) -> int:
        return max(original - delta, min(original + delta, value))

    return EditSuggestion(
        filename=pair.asset.filename,
        asset_id=pair.asset.raw_path.as_posix(),
        exposure=bounded_float(parsed.exposure, current.exposure, 0.5),
        brightness=bounded_float(parsed.brightness, current.brightness, 0.5),
        contrast=bounded_int(parsed.contrast, current.contrast, 30),
        highlights=bounded_int(parsed.highlights, current.highlights, 30),
        shadows=bounded_int(parsed.shadows, current.shadows, 30),
        whites=bounded_int(parsed.whites, current.whites, 30),
        blacks=bounded_int(parsed.blacks, current.blacks, 30),
        temperature=bounded_int(parsed.temperature, current.temperature, 30),
        tint=bounded_int(parsed.tint, current.tint, 30),
        vibrance=bounded_int(parsed.vibrance, current.vibrance, 30),
        saturation=bounded_int(parsed.saturation, current.saturation, 30),
        clarity=bounded_int(parsed.clarity, current.clarity, 30),
        dehaze=bounded_int(parsed.dehaze, current.dehaze, 30),
        structure=bounded_int(parsed.structure, current.structure, 30),
        sharpness=bounded_int(parsed.sharpness, current.sharpness, 30),
        luma_noise_reduction=bounded_int(
            parsed.luma_noise_reduction, current.luma_noise_reduction, 30
        ),
        color_noise_reduction=bounded_int(
            parsed.color_noise_reduction, current.color_noise_reduction, 30
        ),
        vignette_amount=bounded_int(
            parsed.vignette_amount, current.vignette_amount, 30
        ),
        has_crop=parsed.has_crop,
        crop_left=parsed.crop_left,
        crop_top=parsed.crop_top,
        crop_right=parsed.crop_right,
        crop_bottom=parsed.crop_bottom,
        crop_angle=bounded_float(parsed.crop_angle, current.crop_angle, 5.0),
        additional_edits=list(parsed.additional_edits),
        summary=parsed.summary.strip(),
    )
