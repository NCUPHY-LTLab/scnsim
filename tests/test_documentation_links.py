"""Pure-source and rendered-file checks; no Quarto or circuit execution."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/check_documentation_links.py"
spec = importlib.util.spec_from_file_location("documentation_links", SCRIPT)
links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(links)


class DocumentationLinkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "_site"

    def write(self, relative, text="fixture\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self, *sources):
        return links.check_links(self.root, sources, site_root=Path("_site"))

    def test_root_url_checks_actual_rendered_file_with_fragment(self):
        self.write("docs/tutorials/inputs.qmd", "[Contract](/en/docs/contracts/parameter-system.html#parameters)")
        self.write("_site/en/docs/contracts/parameter-system.html", "<html>rendered</html>")
        self.assertEqual(self.check("docs/tutorials/inputs.qmd"), 1)
        self.assertFalse((self.root / "en/docs/contracts/parameter-system.html").exists())

    def test_source_and_repo_decoy_do_not_replace_missing_rendered_page(self):
        self.write("guide.qmd", "[Contract](/en/docs/contracts/parameter-system.html)")
        self.write("docs/contracts/parameter-system.qmd", "# Existing source")
        self.write("en/docs/contracts/parameter-system.html", "source-tree decoy")
        with self.assertRaisesRegex(AssertionError, "broken rendered local link"):
            self.check("guide.qmd")
        # A directory with an HTML filename is also not a rendered page.
        (self.site / "en/docs/contracts/parameter-system.html").mkdir(parents=True)
        with self.assertRaisesRegex(AssertionError, "broken rendered local link"):
            self.check("guide.qmd")

    def test_root_notebook_download_requires_published_file(self):
        self.write("guide.qmd", "[Notebook](/en/examples/chapter.ipynb)")
        self.write("examples/chapter.ipynb", "{}")
        with self.assertRaisesRegex(AssertionError, "broken rendered local link"):
            self.check("guide.qmd")
        self.write("_site/en/examples/chapter.ipynb", "{}")
        self.assertEqual(self.check("guide.qmd"), 1)

    def test_relative_links_keep_source_file_checks(self):
        self.write("docs/guide.qmd", "[Next](next.qmd#section)")
        target = self.write("docs/next.qmd", "# Next")
        self.assertEqual(self.check("docs/guide.qmd"), 1)
        target.unlink()
        self.write("_site/en/docs/next.html", "<html>not a source</html>")
        with self.assertRaisesRegex(AssertionError, "broken local link"):
            self.check("docs/guide.qmd")

    def test_include_fragment_links_use_every_consumer_context(self):
        self.write("a/page.qmd", '{{< include "../shared/body.qmd" >}}')
        self.write("b/page.qmd", "{{< include ../shared/body.qmd >}}")
        self.write("shared/body.qmd", "[Figure](figure.svg)")
        self.write("a/figure.svg")
        target = self.write("b/figure.svg")
        sources = ("a/page.qmd", "b/page.qmd", "shared/body.qmd")
        self.assertEqual(self.check(*sources), 3)
        self.assertFalse((self.root / "shared/figure.svg").exists())
        target.unlink()
        with self.assertRaisesRegex(AssertionError, "broken local link"):
            self.check(*sources)

    def test_missing_include_is_rejected(self):
        self.write("guide.qmd", "{{< include missing.qmd >}}")
        with self.assertRaisesRegex(AssertionError, "broken local include"):
            self.check("guide.qmd")

    def test_fenced_examples_external_and_fragment_links_stay_excluded(self):
        self.write("guide.qmd", """# Guide
```python
[Missing](/en/not-rendered.html)
{{< include missing.qmd >}}
```
~~~~markdown
[Missing](missing.qmd)
{{< include also-missing.qmd >}}
~~~~
[External](https://example.org/missing.html)
[Email](mailto:reader@example.org)
[Heading](#guide)
""")
        self.assertEqual(self.check("guide.qmd"), 1)

    def test_root_directory_requires_index_and_cannot_escape_site(self):
        self.write("guide.qmd", "[English](/en/)")
        (self.site / "en").mkdir(parents=True)
        with self.assertRaisesRegex(AssertionError, "broken rendered local link"):
            self.check("guide.qmd")
        self.write("_site/en/index.html", "<html>index</html>")
        self.assertEqual(self.check("guide.qmd"), 1)
        self.write("guide.qmd", "[Outside](/../outside.html)")
        self.write("outside.html", "outside web root")
        with self.assertRaisesRegex(AssertionError, "broken rendered local link"):
            self.check("guide.qmd")

    def test_root_directory_symlink_index_cannot_escape_site(self):
        self.write("guide.qmd", "[English](/en/)")
        outside = self.write("outside.html", "outside web root")
        index = self.site / "en/index.html"
        index.parent.mkdir(parents=True)
        try:
            index.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(AssertionError, "broken rendered local link"):
            self.check("guide.qmd")

    def test_cli_uses_tracked_prose_and_shared_checker(self):
        self.write("guide.qmd", "[Page](/en/page.html)")
        self.write("_site/en/page.html", "<html>page</html>")
        argv = [str(SCRIPT), "--root", str(self.root), "--site-root", "_site"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(
            links.subprocess, "check_output", return_value="guide.qmd\n"
        ) as git_read, redirect_stdout(output):
            links.main()
        git_read.assert_called_once_with(
            ["git", "ls-files", "*.md", "*.qmd"], cwd=self.root, text=True,
        )
        self.assertIn("verified local links in 1", output.getvalue())

    def test_both_ci_steps_call_shared_script_after_render(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        bodies = re.findall(
            r"      - name: Render and inspect documentation\n(.*?)(?=\n      -|\n  [a-z])",
            workflow, re.S,
        )
        self.assertEqual(len(bodies), 2)
        calls = []
        for body in bodies:
            self.assertIn("quarto render --no-execute --no-clean", body)
            match = re.search(
                r"(?m)^          (uv run --locked(?: --no-sync)? python scripts/check_documentation_links.py --site-root _site)$",
                body,
            )
            self.assertIsNotNone(match)
            self.assertLess(body.index("quarto render"), match.start())
            self.assertIn('echo "generated output is tracked"', body)
            self.assertNotIn("def prose_only", body)
            self.assertNotIn("include_contexts", body)
            calls.append(match.group(1))
        self.assertEqual(calls, [
            "uv run --locked python scripts/check_documentation_links.py --site-root _site",
            "uv run --locked --no-sync python scripts/check_documentation_links.py --site-root _site",
        ])


if __name__ == "__main__":
    unittest.main()
