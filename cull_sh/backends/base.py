from __future__ import annotations

from abc import ABC, abstractmethod

from cull_sh.models import EditSuggestion, FinalDecision, PreviewImage


class VisionBackendError(RuntimeError):
    """Raised when a vision backend cannot score one or more previews."""


class VisionBackend(ABC):
    @abstractmethod
    def score_batch(
        self,
        prompt: str,
        previews: list[PreviewImage],
    ) -> list[FinalDecision]:
        """
        Score a same-scene cohort in input order and return one decision per preview.

        Implementations should preserve input order so the pipeline can map results
        back to work items without depending on provider-specific ids.
        """
        raise NotImplementedError

    @abstractmethod
    def suggest_edits(
        self,
        prompt: str,
        previews: list[PreviewImage],
    ) -> list[EditSuggestion]:
        """
        Suggest gentle global develop adjustments, one per preview in input order.

        Edits are judged per image, so the cohort here is only a batching unit.
        """
        raise NotImplementedError
