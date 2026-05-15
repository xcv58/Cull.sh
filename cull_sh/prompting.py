from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GenrePreset(str, Enum):
    AUTO = "auto"
    MIXED = "mixed"
    PORTRAIT = "portrait"
    STREET = "street"
    FLOWERS = "flowers"
    WILDLIFE = "wildlife"
    LANDSCAPE = "landscape"
    EVENT = "event"
    PRODUCT = "product"


@dataclass(slots=True)
class PromptSelection:
    prompt: str
    genre: GenrePreset
    prefer: str | None
    source: str


PROMPT_TEMPLATES: dict[GenrePreset, str] = {
    GenrePreset.AUTO: (
        "Triage this photo set conservatively across mixed subjects. "
        "Use three outcomes only: reject, review, and pick. "
        "Reject only obvious misses that do not need human attention. "
        "Pick only images that are clearly worth extra editing effort. "
        "Use review for everything usable, borderline, or uncertain. "
        "Prefer frames with clear subjects, good timing, and compelling composition."
    ),
    GenrePreset.MIXED: (
        "Triage this mixed-subject photo set. "
        "Use reject only for obvious misses, pick only for clear standouts, and review for the middle ground. "
        "Favor images that are sharp, well-timed, and compositionally strong for their subject."
    ),
    GenrePreset.PORTRAIT: (
        "Triage this portrait set. "
        "Reject only obvious misses such as blinks, awkward expressions, missed focus, or unflattering poses. "
        "Pick only flattering portraits that are clearly worth extra edit effort. "
        "Use review for solid portraits that still need a human choice. "
        "Favor sharp eyes, natural expressions, clean framing, and strong subject separation."
    ),
    GenrePreset.STREET: (
        "Triage this street photography set. "
        "Reject only obvious technical misses or weak frames with no clear subject. "
        "Pick only frames with standout timing, gesture, layering, or composition. "
        "Use review for the usable middle ground."
    ),
    GenrePreset.FLOWERS: (
        "Triage this flower and botanical set. "
        "Reject only clear misses such as missed focus or distracting, unusable compositions. "
        "Pick only the sharpest frames with strong shape, symmetry, color, or subject clarity. "
        "Use review for solid but less certain frames."
    ),
    GenrePreset.WILDLIFE: (
        "Triage this wildlife set. "
        "Reject only obvious misses such as severe blur, obscured subjects, or weak unusable poses. "
        "Pick only the strongest frames with sharp detail, strong timing, and compelling pose or action. "
        "Use review for usable images that still need a human call."
    ),
    GenrePreset.LANDSCAPE: (
        "Triage this landscape set. "
        "Reject only clearly weak, muddy, or technically poor frames. "
        "Pick only frames with standout composition, clean edges, depth, and appealing light. "
        "Use review for the rest."
    ),
    GenrePreset.EVENT: (
        "Triage this event set. "
        "Reject only obvious misses such as blinks, awkward in-between moments, missed focus, or redundant unusable frames. "
        "Pick only frames with strong moments, clear subjects, good expressions, and useful coverage. "
        "Use review for everything else that may still deserve a human decision."
    ),
    GenrePreset.PRODUCT: (
        "Triage this product set. "
        "Reject only clear technical misses such as bad focus, distracting reflections, uneven framing, or inconsistent presentation. "
        "Pick only frames that are clearly sharp, clean, consistent, well-lit, and commercially strong. "
        "Use review for acceptable frames that still need a human choice."
    ),
}


def resolve_prompt(
    prompt: str | None,
    genre: GenrePreset,
    prefer: str | None = None,
    source: str | None = None,
) -> PromptSelection:
    cleaned_prompt = _clean(prompt)
    cleaned_prefer = _clean(prefer)

    if cleaned_prompt is not None:
        return PromptSelection(
            prompt=cleaned_prompt,
            genre=genre,
            prefer=cleaned_prefer,
            source=source or "custom",
        )

    base_prompt = PROMPT_TEMPLATES[genre]
    prompt_lines = [base_prompt]
    if cleaned_prefer is not None:
        prompt_lines.append(f"Additional priority: {cleaned_prefer}")

    return PromptSelection(
        prompt=" ".join(prompt_lines),
        genre=genre,
        prefer=cleaned_prefer,
        source=source or "preset",
    )


def parse_genre(value: str) -> GenrePreset:
    normalized = value.strip().lower()
    for genre in GenrePreset:
        if genre.value == normalized:
            return genre
    raise ValueError(
        f"unsupported genre: {value}. Expected one of: "
        + ", ".join(genre.value for genre in GenrePreset)
    )


def supported_genre_labels() -> str:
    return ", ".join(genre.value for genre in GenrePreset if genre != GenrePreset.AUTO)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
