"""Regression tests for the pinned Askr vendor boundary in the Pages publisher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import pages_site


REPOSITORY = Path(__file__).resolve().parents[1]
ASKR_VENDOR = Path(
    os.environ.get(
        "SCNSIM_ASKR_VENDOR",
        REPOSITORY / "_extensions" / "arfiligol" / "askr",
    )
)
CATALOG = {
    "schema": "scnsim.pages_versions.v1",
    "development": {"label": "1.0.0 dev", "root": "/scnsim/1.0.0-dev/"},
    "releases": [],
}
SOURCE_SHA = "a" * 40
PUBLICATION_SHA = "b" * 40


class PagesAskrVendorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ASKR_VENDOR.is_dir():
            raise AssertionError(f"official Askr vendor tree is unavailable: {ASKR_VENDOR}")
        if pages_site._vendor_digest(ASKR_VENDOR) != pages_site.ASKR_SHA256:
            raise AssertionError("repository Askr tree does not match the publisher pin")

    def _source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "_quarto.yml").write_text("project:\n  type: website\n", encoding="utf-8")
        (source / "index.qmd").write_text("# Test\n", encoding="utf-8")
        (source / "pyproject.toml").write_text(
            '[project]\nname = "scnsim"\nversion = "1.0.0a3"\n',
            encoding="utf-8",
        )
        return source

    def _install_destination(self, source: Path, *, old_layout: bool = False, subset_size: int | None = None) -> Path:
        destination = source / "_extensions" / "arfiligol" / "askr"
        shutil.copytree(ASKR_VENDOR, destination)
        if old_layout:
            extension = destination / "_extension.yml"
            text = extension.read_text(encoding="utf-8")
            self.assertIn("version: 0.13.3\n", text)
            extension.write_text(
                text.replace("version: 0.13.3\n", "version: 0.12.1\n", 1),
                encoding="utf-8",
            )
            for name in ("chart-view.lua", "chart-viewer.css", "chart-viewer.js", "copy-controls.html"):
                (destination / name).unlink()
            fixture = Path(__file__).parent / "fixtures/pages/askr-0.12.1-math-copy.html"
            shutil.copyfile(fixture, destination / "math-copy.html")
            self.assertEqual(len([path for path in destination.rglob("*") if path.is_file()]), 40)
            self.assertEqual(
                hashlib.sha256((destination / "math-copy.html").read_bytes()).hexdigest(),
                pages_site.RETIRED_MATH_COPY_SHA256,
            )
        elif subset_size is not None:
            files = sorted(path for path in destination.rglob("*") if path.is_file())
            keep = {path.relative_to(destination) for path in files[:subset_size]}
            keep.add(Path("_extension.yml"))
            for path in files:
                if path.relative_to(destination) not in keep:
                    path.unlink()
            self.assertEqual(len([path for path in destination.rglob("*") if path.is_file()]), subset_size)
        return destination

    def _prepare(self, source: Path, vendor: Path = ASKR_VENDOR) -> None:
        pages_site._prepare(
            source, vendor, CATALOG, "1.0.0 dev", SOURCE_SHA, PUBLICATION_SHA
        )

    def test_official_v0133_manifest_is_43_files(self) -> None:
        rows = [path for path in ASKR_VENDOR.rglob("*") if path.is_file()]
        self.assertEqual(len(rows), 43)
        self.assertEqual(pages_site._vendor_digest(ASKR_VENDOR), pages_site.ASKR_SHA256)

    def test_prepare_overlays_empty_exact_older40_and_older25_destinations(self) -> None:
        for destination_kind in ("empty", "exact", "older40", "older25"):
            with self.subTest(destination=destination_kind), tempfile.TemporaryDirectory() as temp:
                source = self._source(Path(temp))
                if destination_kind == "exact":
                    self._install_destination(source)
                elif destination_kind == "older40":
                    self._install_destination(source, old_layout=True)
                elif destination_kind == "older25":
                    self._install_destination(source, subset_size=25)
                self._prepare(source)
                destination = source / "_extensions" / "arfiligol" / "askr"
                self.assertEqual(pages_site._vendor_digest(destination), pages_site.ASKR_SHA256)
                self.assertTrue((destination / "chart-view.lua").is_file())
                self.assertFalse((destination / "math-copy.html").exists())
                self.assertTrue((source / "_quarto-pages.yml").is_file())

    def test_rejects_modified_missing_extra_renamed_and_symlink_vendor_files(self) -> None:
        mutations = ("modified", "missing", "extra", "renamed", "symlink")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                vendor = root / "vendor"
                shutil.copytree(ASKR_VENDOR, vendor)
                member = next(path for path in vendor.rglob("*") if path.is_file())
                if mutation == "modified":
                    member.write_bytes(member.read_bytes() + b"changed")
                elif mutation == "missing":
                    member.unlink()
                elif mutation == "extra":
                    (vendor / "unexpected-extra.txt").write_text("extra", encoding="utf-8")
                elif mutation == "renamed":
                    member.rename(member.with_name(member.name + ".renamed"))
                else:
                    content = member.read_bytes()
                    member.unlink()
                    target = vendor / "symlink-target"
                    target.write_bytes(content)
                    member.symlink_to(target.name)
                source = self._source(root)
                with self.assertRaises(ValueError):
                    self._prepare(source, vendor)
                self.assertFalse((source / "_quarto-pages.yml").exists())

    def test_rejects_unexpected_file_and_preserves_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = self._source(Path(temp))
            destination = self._install_destination(source)
            unexpected = destination / "unexpected-leftover.txt"
            unexpected.write_text("preserve and reject", encoding="utf-8")
            with self.assertRaises(ValueError):
                self._prepare(source)
            self.assertEqual(unexpected.read_text(encoding="utf-8"), "preserve and reject")
            self.assertFalse((source / "_quarto-pages.yml").exists())

    def test_rejects_unrecognized_math_copy_and_preserves_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = self._source(Path(temp))
            destination = self._install_destination(source)
            legacy_name = destination / "math-copy.html"
            legacy_name.write_text("unrecognized bytes", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrecognized math-copy.html"):
                self._prepare(source)
            self.assertEqual(legacy_name.read_text(encoding="utf-8"), "unrecognized bytes")
            self.assertFalse((source / "_quarto-pages.yml").exists())


if __name__ == "__main__":
    unittest.main()
