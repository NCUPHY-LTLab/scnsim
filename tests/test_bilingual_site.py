from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import bilingual_site


class BilingualSiteReadmeTests(unittest.TestCase):
    def make_project(self, root: Path) -> None:
        (root / "zh-TW").mkdir(parents=True)
        (root / "zh-TW/_quarto.yml").write_text("project: {}\n", encoding="utf-8")
        for relative in (
            "_extensions/arfiligol/askr",
            "tests/figures",
            "docs/assets",
            "examples/engineer/figures",
            "examples/engineer/chapter-07/position-figures",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        for name in ("LICENSE", "NOTICE", "pyproject.toml"):
            (root / name).write_text("fixture\n", encoding="utf-8")
        schemas = root / "src/scnsim/_schemas"
        schemas.mkdir(parents=True)
        for name in (
            "identity-common.schema.json",
            "identity-v2.schema.json",
            "identity-v3.schema.json",
        ):
            (schemas / name).write_text("{}\n", encoding="utf-8")

    def test_prepare_allows_only_root_gfm_readme_and_keeps_qmd_code_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_project(root)
            gfm = "# Read me\n\n```sh\nuv init --python 3.12\n```\n"
            (root / "README.md").write_text("# English\n\n```sh\necho source\n```\n", encoding="utf-8")
            (root / "zh-TW/README.md").write_text(gfm, encoding="utf-8")
            (root / "index.qmd").write_text("```python\nanswer = 42\n```\n", encoding="utf-8")
            (root / "zh-TW/index.qmd").write_text(
                "答案：\n{{< include _generated/canonical-code/index.qmd/block-001.qmd >}}\n",
                encoding="utf-8",
            )

            bilingual_site.prepare(root)

            self.assertEqual((root / "zh-TW/README.md").read_text(encoding="utf-8"), gfm)
            self.assertFalse((root / "zh-TW/_generated/canonical-code/README.md").exists())
            self.assertEqual(
                (root / "zh-TW/_generated/canonical-code/index.qmd/block-001.qmd").read_text(
                    encoding="utf-8"
                ),
                "```python\nanswer = 42\n```\n",
            )
            manifest = json.loads((root / "zh-TW/_generated/canonical-code.json").read_text())
            self.assertEqual([row["source"] for row in manifest], ["index.qmd"])

    def test_prepare_requires_the_canonical_root_readme(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_project(root)
            (root / "zh-TW/README.md").write_text(
                "# 獨立 Markdown\n\n```sh\nuv init --python 3.12\n```\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Canonical English source missing: README.md"):
                bilingual_site.prepare(root)

    def test_prepare_still_rejects_fenced_qmd_and_nested_readme(self) -> None:
        for relative in ("index.qmd", "docs/README.md"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_project(root)
                (root / "README.md").write_text("# Root\n", encoding="utf-8")
                (root / "zh-TW/README.md").write_text("# 根目錄\n", encoding="utf-8")
                canonical = root / relative
                translated = root / "zh-TW" / relative
                canonical.parent.mkdir(parents=True, exist_ok=True)
                translated.parent.mkdir(parents=True, exist_ok=True)
                canonical.write_text("```python\nx = 1\n```\n", encoding="utf-8")
                translated.write_text("```python\nx = 1\n```\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "canonical code includes"):
                    bilingual_site.prepare(root)

    def test_materialize_preserves_root_readme_and_converts_validated_qmd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_project(root)
            (root / "README.md").write_text("# Project\n\n```sh\necho canonical\n```\n", encoding="utf-8")
            (root / "index.qmd").write_text("```python\nanswer = 42\n```\n", encoding="utf-8")
            translations = root / "translation-input"
            translations.mkdir()
            gfm = "# 專案\n\n```sh\nuv init --python 3.12\n```\n"
            (translations / "README.md").write_text(gfm, encoding="utf-8")
            (translations / "index.qmd").write_text(
                "答案：\n```python\nanswer = 42\n```\n",
                encoding="utf-8",
            )

            bilingual_site.materialize(root, translations)

            self.assertEqual((root / "zh-TW/README.md").read_text(encoding="utf-8"), gfm)
            self.assertIn(
                "{{< include _generated/canonical-code/index.qmd/block-001.qmd >}}",
                (root / "zh-TW/index.qmd").read_text(encoding="utf-8"),
            )

    def test_materialize_rejects_mismatched_qmd_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_project(root)
            (root / "README.md").write_text("# Project\n", encoding="utf-8")
            (root / "index.qmd").write_text("```python\nanswer = 42\n```\n", encoding="utf-8")
            translations = root / "translation-input"
            translations.mkdir()
            (translations / "README.md").write_text(
                "# 專案\n\n```sh\nuv init --python 3.12\n```\n", encoding="utf-8"
            )
            (translations / "index.qmd").write_text(
                "```python\nanswer = 41\n```\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "differs from canonical English source"):
                bilingual_site.materialize(root, translations)


if __name__ == "__main__":
    unittest.main()
