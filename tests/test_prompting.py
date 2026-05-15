from __future__ import annotations

import unittest

from cull_sh.prompting import GenrePreset
from cull_sh.prompting import parse_genre
from cull_sh.prompting import resolve_prompt


class PromptingTests(unittest.TestCase):
    def test_explicit_prompt_wins(self) -> None:
        selection = resolve_prompt(
            prompt="Keep the dramatic black-and-white frames.",
            genre=GenrePreset.STREET,
            prefer="interesting gestures",
        )

        self.assertEqual(selection.prompt, "Keep the dramatic black-and-white frames.")
        self.assertEqual(selection.source, "custom")
        self.assertEqual(selection.genre, GenrePreset.STREET)
        self.assertEqual(selection.prefer, "interesting gestures")

    def test_preset_prompt_includes_preference(self) -> None:
        selection = resolve_prompt(
            prompt=None,
            genre=GenrePreset.PORTRAIT,
            prefer="natural expressions",
            source="interactive",
        )

        self.assertIn("portrait", selection.prompt.lower())
        self.assertIn("Additional priority: natural expressions", selection.prompt)
        self.assertIn("review", selection.prompt.lower())
        self.assertIn("pick", selection.prompt.lower())
        self.assertEqual(selection.source, "interactive")

    def test_parse_genre_is_case_insensitive(self) -> None:
        self.assertEqual(parse_genre("Flowers"), GenrePreset.FLOWERS)


if __name__ == "__main__":
    unittest.main()
