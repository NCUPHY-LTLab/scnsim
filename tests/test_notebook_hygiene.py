"""Source-bound notebook attachment regressions; no course or backend execution."""

import base64
from pathlib import Path
import runpy
import tempfile
import unittest


validate_cell_attachments = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "check_notebook_attachments.py")
)["validate_cell_attachments"]


class NotebookAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.notebook = self.root / "examples" / "chapter-01" / "chapter.ipynb"
        self.notebook.parent.mkdir(parents=True)
        self.notebook.write_text("{}", encoding="utf-8")
        self.image = self.root / "examples" / "figures" / "circuit.svg"
        self.image.parent.mkdir()
        self.raw = (b'<svg xmlns="http://www.w3.org/2000/svg">'
                    + b'<path d="M0 0L1 1"/>' * 12 + b'</svg>\n')
        if len(self.raw) % 3 == 0:
            self.raw += b" "  # Ensure pad-bit regressions exercise this exact SVG.
        self.image.write_bytes(self.raw)
        self.name = "../figures/circuit.svg"

    def cell(self, *, name=None, source=None, mime="image/svg+xml", raw=None):
        name = self.name if name is None else name
        return {
            "cell_type": "markdown",
            "source": f'<img src="attachment:{name}" alt="circuit" />' if source is None else source,
            "attachments": {name: {mime: base64.b64encode(self.raw if raw is None else raw).decode("ascii")}},
        }

    def validate(self, cell):
        return validate_cell_attachments(cell, self.notebook, self.root)

    def test_public_relative_svg_is_exact_and_retained_for_comparison(self):
        cell = self.cell()
        self.assertEqual(self.validate(cell), cell["attachments"])
        cell["source"] = ['![circuit](attachment:../figures/circuit.svg)\n']
        self.assertEqual(self.validate(cell), cell["attachments"])
        cell["source"] = '![circuit][diagram]\n\n[diagram]: <attachment:../figures/circuit.svg>'
        self.assertEqual(self.validate(cell), cell["attachments"])

    def test_source_byte_drift_is_rejected(self):
        cell = self.cell()
        self.image.write_bytes(self.raw + b"\n")
        with self.assertRaisesRegex(AssertionError, "bytes differ"):
            self.validate(cell)

    def test_invalid_and_noncanonical_base64_are_rejected(self):
        compact = base64.b64encode(self.raw).decode("ascii")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        position = len(compact.rstrip("=")) - 1
        pad_drift = compact[:position] + alphabet[alphabet.index(compact[position]) + 1] + compact[position + 1:]
        # Invalid pad bits decode to the same SVG if canonicality is unchecked.
        self.assertEqual(base64.b64decode(pad_drift, validate=True), self.raw)
        invalid = ["!invalid!", "\u00e9", compact + "=", pad_drift]
        invalid += [compact[:12] + char + compact[12:] for char in (" ", "\t", "\v", "\f", "!")]
        for encoded in invalid:
            with self.subTest(encoded=encoded):
                cell = self.cell()
                cell["attachments"][self.name]["image/svg+xml"] = encoded
                with self.assertRaises(AssertionError):
                    self.validate(cell)

    def test_mime_lf_and_crlf_wrapping_normalize_to_compact(self):
        compact = base64.b64encode(self.raw).decode("ascii")
        self.assertGreater(len(compact), 76)
        expected = {self.name: {"image/svg+xml": compact}}
        for separator in ("\n", "\r\n"):
            wrapped = separator.join(compact[index:index + 76] for index in range(0, len(compact), 76)) + separator
            cell = self.cell()
            cell["attachments"][self.name]["image/svg+xml"] = wrapped
            with self.subTest(separator=repr(separator)):
                self.assertEqual(self.validate(cell), expected)
                self.assertEqual(cell["attachments"][self.name]["image/svg+xml"], wrapped)

    def test_raw_html_text_cannot_authorize_markdown_attachments(self):
        image = f"![x](attachment:{self.name})"
        for tag in ("script", "style", "pre", "textarea", "title"):
            for source in (f"<{tag}>{image}</{tag}>",
                           f"<{tag}>\n\n{image}\n</{tag}>",
                           f"prefix <{tag}>{image}</{tag}>",
                           f"<{tag}>{image}"):
                with self.subTest(tag=tag, source=source):
                    with self.assertRaises(AssertionError):
                        self.validate(self.cell(source=source))
        script = f"<script>const x='{image}'</script>"
        with self.assertRaises(AssertionError):
            self.validate(self.cell(source=script))
        # A fake HTML img inside textarea is text, too.
        with self.assertRaises(AssertionError):
            self.validate(self.cell(source=f'<textarea><img src="attachment:{self.name}"></textarea>'))

    def test_raw_html_block_blank_line_boundary_and_real_img(self):
        image = f"![x](attachment:{self.name})"
        for tag in ("div", "figure", "title", "custom-block"):
            with self.subTest(tag=tag):
                with self.assertRaises(AssertionError):
                    self.validate(self.cell(source=f"<{tag}>\n{image}\n</{tag}>"))
                # A blank line terminates this kind of HTML block.
                cell = self.cell(source=f"<{tag}>\ntext\n</{tag}>\n\n{image}")
                self.assertEqual(self.validate(cell), cell["attachments"])
        with self.assertRaises(AssertionError):
            self.validate(self.cell(source=f"<div>\n<script>fake</script>\n{image}\n</div>"))
        with self.assertRaises(AssertionError):
            self.validate(self.cell(source=f"<title>fake</title>\n{image}"))
        for source in (f'<figure>\n<img src="attachment:{self.name}">\n</figure>',
                       f'<pre><img src="attachment:{self.name}"></pre>',
                       f"<script>fake</script>\n{image}"):
            with self.subTest(source=source):
                cell = self.cell(source=source)
                self.assertEqual(self.validate(cell), cell["attachments"])

    def test_missing_and_extra_attachments_are_rejected(self):
        missing = self.cell()
        missing["attachments"] = {}
        extra = self.cell()
        extra["attachments"]["unused.svg"] = extra["attachments"][self.name]
        for cell in (missing, extra, self.cell(source="plain prose")):
            with self.subTest(cell=cell):
                with self.assertRaises(AssertionError):
                    self.validate(cell)

    def test_code_examples_and_comments_do_not_reference_an_attachment(self):
        uri = f'<img src="attachment:{self.name}">'
        for source in (f"```html\n{uri}\n```", f"`{uri}`", f"<!-- {uri} -->", "    " + uri,
                       f"![circuit](elsewhere.svg)\n[circuit]: attachment:{self.name}"):
            with self.subTest(source=source):
                with self.assertRaises(AssertionError):
                    self.validate(self.cell(source=source))

    def test_non_markdown_cells_cannot_carry_attachments(self):
        for kind in ("code", "raw"):
            cell = self.cell()
            cell["cell_type"] = kind
            with self.subTest(kind=kind):
                with self.assertRaises(AssertionError):
                    self.validate(cell)
        self.assertEqual(self.validate({"cell_type": "code", "source": "print(1)"}), {})

    def test_non_image_mime_and_extensions_are_rejected(self):
        for cell in (self.cell(mime="text/plain"), self.cell(mime="image/png"),
                     self.cell(name="../figures/circuit.txt")):
            with self.subTest(cell=cell):
                with self.assertRaises(AssertionError):
                    self.validate(cell)
        cell = self.cell()
        cell["attachments"][self.name]["text/plain"] = "private payload"
        with self.assertRaises(AssertionError):
            self.validate(cell)

    def test_non_svg_bytes_are_rejected_even_when_source_matches(self):
        raw = b"<html>not an image</html>"
        self.image.write_bytes(raw)
        with self.assertRaises(AssertionError):
            self.validate(self.cell(raw=raw))

    def test_missing_source_and_corrupt_svg_are_rejected(self):
        self.image.unlink()
        with self.assertRaises(AssertionError):
            self.validate(self.cell())
        self.image.write_bytes(b"<svg")
        with self.assertRaises(AssertionError):
            self.validate(self.cell(raw=b"<svg"))

    def test_escaping_absolute_uri_and_windows_paths_are_rejected(self):
        outside = self.root.parent / "outside.svg"
        outside.write_bytes(self.raw)
        names = ("../../../outside.svg", str(outside), "C:/private.svg", "C:private.svg",
                 r"C:\private.svg", r"\\server\private.svg", r"..\figures\circuit.svg",
                 "//server/private.svg", "file:///private.svg", "https://example.org/x.svg",
                 "../figures/circuit.svg?x=1", "../figures/circuit.svg#x", "%2e%2e/private.svg")
        for name in names:
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    self.validate(self.cell(name=name))

    def test_attachment_schema_and_artifact_path_are_checked(self):
        for attachments in (None, [], {self.name: {"image/svg+xml": []}}):
            cell = self.cell()
            cell["attachments"] = attachments
            with self.subTest(attachments=attachments):
                with self.assertRaises(AssertionError):
                    self.validate(cell)
        with self.assertRaises(AssertionError):
            validate_cell_attachments(self.cell(), "fresh render of chapter.qmd", self.root)
        with self.assertRaises(AssertionError):
            validate_cell_attachments(self.cell(), self.root.parent / "chapter.ipynb", self.root)


if __name__ == "__main__":
    unittest.main()
