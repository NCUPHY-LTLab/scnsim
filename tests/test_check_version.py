from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.check_version import repository_version, validate_line


class VersionLineTests(unittest.TestCase):
    def test_main_accepts_stable_and_explicitly_approved_alphas(self) -> None:
        for version in ("1.0.0", "1.0.0a2", "1.0.0a3"):
            with self.subTest(version=version):
                validate_line(version, "main")

    def test_main_rejects_unapproved_prereleases(self) -> None:
        for version in ("1.0.0.dev1", "1.0.0a1", "1.0.0a4", "1.0.0b1", "1.0.0rc1"):
            with self.subTest(version=version), self.assertRaisesRegex(
                ValueError, "main must use a stable version"
            ):
                validate_line(version, "main")

    def test_develop_rejects_stable_but_accepts_prerelease_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "develop must use"):
            validate_line("1.0.0", "develop")
        for version in ("1.0.0.dev1", "1.0.0a3", "1.0.0b1", "1.0.0rc1"):
            with self.subTest(version=version):
                validate_line(version, "develop")

    def test_repository_version_requires_metadata_and_lock_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "scnsim"\nversion = "1.0.0a3"\n',
                encoding="utf-8",
            )
            (root / "uv.lock").write_text(
                '[[package]]\nname = "scnsim"\nversion = "1.0.0a3"\n'
                'source = { editable = "." }\n',
                encoding="utf-8",
            )
            self.assertEqual(repository_version(root), "1.0.0a3")

            (root / "uv.lock").write_text(
                '[[package]]\nname = "scnsim"\nversion = "1.0.0a2"\n'
                'source = { editable = "." }\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                repository_version(root)


if __name__ == "__main__":
    unittest.main()
