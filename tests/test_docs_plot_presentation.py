"""Exercise the real saved-plot output router without Plotly or rendering."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts/docs_plot_presentation.py"


class DocsPlotPresentationRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()
        self.source_reads = []
        self.html_reads = []
        self.writes = []
        owner = self

        class Figure:
            def __init__(self):
                self.data = [SimpleNamespace(x=["x"], y=["y"])]
                self.layout = SimpleNamespace(height=340, showlegend=False)
            def write_html(self, path, **kwargs):
                owner.writes.append(path)
                path.write_bytes(b"saved html\n")
            def write_image(self, path, **kwargs):
                owner.writes.append(path)
                path.write_bytes(b"saved vector\n")

        def build_case(name, source):
            self.assertEqual(source, self.source)
            path = source / (name + ".csv")
            self.source_reads.append(path)
            path.read_bytes()
            return [(name, Figure(), [(["x"], ["y"])])], [path], "saved fixture"

        def saved_figure(path):
            self.html_reads.append(path)
            self.assertEqual(path.read_bytes(), b"saved html\n")
            return Figure()

        # Compile the actual generate/main bodies, not a second router. Only
        # scientific plotting and file export are replaced with bounded fixtures.
        tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
        selected = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in {"generate", "main"}
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id in {"CASES", "PRESENTATION_CASES"} for target in node.targets))]
        self.namespace = {"__file__": str(SCRIPT), "Path": Path, "argparse": argparse,
                          "json": json, "np": SimpleNamespace(array_equal=lambda a, b, **kw: a == b),
                          "values": lambda value: value, "build_case": build_case,
                          "saved_figure": saved_figure,
                          "sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest()}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), "exec"), self.namespace)
        for name in self.namespace["CASES"]:
            (self.source / (name + ".csv")).write_bytes(b"saved fixture\n")

    def test_explicit_chapter4_cases_route_every_writer_to_presentation(self):
        result = self.namespace["generate"](self.source, self.output,
                                               ("chapter4-raw", "chapter4-ptc"), png=True)
        expected = {self.output / "presentation" / (name + "." + extension)
                    for name in ("chapter4-raw", "chapter4-ptc")
                    for extension in ("html", "svg", "png")}
        self.assertEqual(set(self.writes), expected)
        self.assertEqual(set(result["assets"]), {
            "presentation/" + name + "." + extension
            for name in ("chapter4-raw", "chapter4-ptc") for extension in ("html", "svg")})
        self.assertEqual({path.parent for path in self.source_reads}, {self.source})

    def test_all_cli_routes_only_the_approved_cases(self):
        argv = [str(SCRIPT), "--source-dir", str(self.source), "--output-dir", str(self.output),
                "--cases", "all"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            self.namespace["main"]()
        routed = {"chapter4-raw", "chapter4-ptc", "chapter7", "chapter8"}
        for case in self.namespace["CASES"]:
            for extension in ("html", "svg"):
                parent = self.output / "presentation" if case in routed else self.output
                self.assertIn(parent / (case + "." + extension), self.writes)
                self.assertTrue((parent / (case + "." + extension)).is_file())
        manifest = json.loads((self.output / "docs-plot-assets.json").read_text())
        self.assertEqual(set(manifest["assets"]), {
            ("presentation/" if case in routed else "") + case + "." + extension
            for case in self.namespace["CASES"] for extension in ("html", "svg")})
        self.assertEqual({path.parent for path in self.source_reads}, {self.source})

    def test_verify_only_reads_the_same_routed_html_without_export(self):
        cases = ("overview", "chapter4-raw", "chapter7", "chapter8")
        self.namespace["generate"](self.source, self.output, cases)
        self.writes.clear()
        self.namespace["generate"](self.source, self.output, cases, verify_only=True)
        self.assertEqual(self.writes, [])
        self.assertEqual(set(self.html_reads), {
            self.output / "overview.html",
            *(self.output / "presentation" / (case + ".html") for case in cases[1:]),
        })


if __name__ == "__main__":
    unittest.main()
