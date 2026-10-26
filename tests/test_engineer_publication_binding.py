"""Non-native regressions for the engineer course's companion/kernel ownership."""
import ast
import hashlib
from importlib.machinery import PathFinder
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/engineer_publication_binding.py'


def helper_function(name):
    tree = ast.parse(HELPER.read_text(), filename=str(HELPER))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(Path=Path)
    exec(compile(module, str(HELPER) + ':' + name, 'exec'), namespace)
    return namespace[name]


class EngineerPublicationBindingTests(unittest.TestCase):
    def test_companion_spec_and_same_source_binding(self):
        present = 'examples' in sys.modules
        original = sys.modules.pop('examples', None)
        path_before = list(sys.path)
        producer_before = {k: v for k, v in sys.modules.items()
                           if k == 'scnsim' or k.startswith('scnsim.')}
        try:
            bind = helper_function('_bind_course_examples')
            bind(ROOT)
            namespace = sys.modules['examples']
            bind(ROOT)
            self.assertIs(sys.modules['examples'], namespace)
            engineer = PathFinder.find_spec('examples.engineer', namespace.__path__)
            diagram = PathFinder.find_spec('examples.engineer.diagram_layouts',
                                          engineer.submodule_search_locations)
            self.assertEqual(Path(diagram.origin).resolve(),
                             (ROOT / 'examples/engineer/diagram_layouts.py').resolve())
            self.assertEqual(sys.path, path_before)
            self.assertEqual({k: v for k, v in sys.modules.items()
                              if k == 'scnsim' or k.startswith('scnsim.')}, producer_before)
        finally:
            sys.modules.pop('examples', None)
            if present:
                sys.modules['examples'] = original

    def test_foreign_companion_namespace_is_preserved(self):
        present = 'examples' in sys.modules
        original = sys.modules.pop('examples', None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                foreign = ModuleType('examples')
                foreign.__path__ = [directory]
                sys.modules['examples'] = foreign
                with self.assertRaisesRegex(RuntimeError, 'existing examples namespace differs'):
                    helper_function('_bind_course_examples')(ROOT)
                self.assertIs(sys.modules['examples'], foreign)
        finally:
            sys.modules.pop('examples', None)
            if present:
                sys.modules['examples'] = original

    def test_kernel_temp_uses_inherited_root_and_native_cleanup(self):
        source = ROOT / "scripts/generate_engineer_chapter8.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_kernel_temporary_directory")
        namespace = {"tempfile": tempfile}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(source) + ":exact-temp-owner", "exec"), namespace)
        create = namespace["_kernel_temporary_directory"]
        with tempfile.TemporaryDirectory() as inherited:
            with create({"TMPDIR": inherited}) as first, create({"TMPDIR": inherited}) as second:
                self.assertEqual(Path(first).parent, Path(inherited))
                self.assertEqual(Path(second).parent, Path(inherited))
                self.assertNotEqual(first, second)
                self.assertTrue(Path(first).is_dir())
            self.assertFalse(Path(first).exists())
            self.assertFalse(Path(second).exists())
        with create({}) as default:
            self.assertEqual(Path(default).parent, Path(tempfile.gettempdir()))
        self.assertFalse(Path(default).exists())

    def test_kernel_poll_error_preserves_primary_and_later_cleanup(self):
        source = ROOT / 'scripts/generate_engineer_chapter8.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_kernel')
        owned_try = next(n for n in function.body if isinstance(n, ast.Try) and n.finalbody)
        skeleton = ast.parse('''def exercise(client, process, manager, cwd, primary):
    private_facts = {}
    identity = {}
    cleanup_error = None
    kernel_tmp = None
    try:
        raise primary
    finally:
        pass
''')
        skeleton.body[0].body[-1].finalbody = owned_try.finalbody
        ast.fix_missing_locations(skeleton)
        namespace = dict(os=os, Path=Path, json=json,
                         _hash=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
        exec(compile(skeleton, str(source) + ':exact-finally', 'exec'), namespace)
        events = []
        primary = RuntimeError('original kernel operation')
        child = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        class OwnedProcess:
            def poll(self):
                events.append('poll')
                raise OSError('liveness readback')
            def wait(self):
                events.append('wait')
                return child.wait()
        client = SimpleNamespace(stop_channels=lambda: events.append('channels'))
        manager = SimpleNamespace(kernel_id='regression',
                                  shutdown_kernel=lambda **kw: events.append('shutdown'))
        try:
            with tempfile.TemporaryDirectory() as directory:
                cwd = Path(directory) / 'execution'
                cwd.mkdir()
                with self.assertRaises(RuntimeError) as caught:
                    namespace['exercise'](client, OwnedProcess(), manager, cwd, primary)
                self.assertIs(caught.exception, primary)
                self.assertEqual(events, ['poll', 'channels', 'shutdown', 'wait'])
                facts = json.loads((cwd.parent / 'kernel-regression-runtime.json').read_text())
                self.assertTrue(facts['waited'])
                self.assertEqual(facts['primary_error'], repr(primary))
        finally:
            child.wait()


class EngineerPresentationTransitionTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("publication_fixture", HELPER)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "chapter-01-artifacts.json"
        self.raw_files = {"summary.md": b"| sample | value |\n| a | unchanged |\n",
                          "response.csv": b"sample,value\na,unchanged\n",
                          "diagram.svg": b'<svg xmlns="http://www.w3.org/2000/svg"/>'}
        for name, raw in self.raw_files.items():
            (self.directory / name).write_bytes(raw)
        old_map = {name: self.sha(raw) for name, raw in self.raw_files.items()}
        tree = {"files": {"src/scnsim/core.py": self.sha(b"unchanged package")}}
        tree["sha256"] = self.helper._canonical_sha256(tree["files"])
        self.prior = {"generator_sha256": self.sha(b"unchanged generator"),
                      "sources": {"scripts/engineer_publication_binding.py": self.sha(b"old helper")},
                      "source_tree": tree, "cell_ids": ["display"],
                      "cells": {"display": self.sha(b"saved display")}}
        self.current = json.loads(json.dumps(self.prior))
        self.current["sources"]["scripts/engineer_publication_binding.py"] = self.sha(b"new helper")
        self.before = {"binding": self.prior, "artifacts": old_map,
                       "execution": {"generator_sha256": self.prior["generator_sha256"],
                                     "source_tree_sha256": tree["sha256"], "cells": self.prior["cells"]},
                       "environment": {"identity": "unchanged"},
                       "observations": {"identity": "unchanged"},
                       "results": {"identity": "unchanged"}}
        self.path.write_text(json.dumps(self.before))
        (self.directory / "summary.md").write_bytes(self.raw_files["summary.md"] + b"\n[Download](response.csv)\n")
        self.published = {name: self.sha((self.directory / name).read_bytes()) for name in old_map}
        self.transition = self.closed_transition(self.published)

    @staticmethod
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()

    def closed_transition(self, published):
        h = self.helper
        return {"prior_manifest_sha256": self.sha(self.path.read_bytes()),
                "prior_publication_binding_sha256": h._canonical_sha256(self.prior),
                "publication_binding_sha256": h._canonical_sha256(self.current),
                "deltas": h._transition_deltas(self.prior, self.current),
                "artifacts": dict(published),
                "historical_execution": h._historical_execution(
                    self.before, self.prior, prior_publication_artifacts=published),
                "prior_publication_rebind_sha256": None}

    def prepare(self, **kwargs):
        with patch.object(self.helper, "_reviewed_transitions", return_value={self.path.name: self.transition}):
            return json.loads(self.helper._prepare_rebind(self.path, self.current, **kwargs))

    def test_closed_markdown_transition_retains_execution_payload(self):
        validated = []
        after = self.prepare(validate_historical=lambda manifest, **kw: validated.append((json.loads(json.dumps(manifest)), kw)))
        h = self.helper
        historical = after["publication_rebind"]["historical_execution"]
        self.assertEqual(after["artifacts"], self.published)
        self.assertEqual(historical["artifacts"], self.before["artifacts"])
        self.assertEqual(historical["prior_publication_artifacts"], self.published)
        self.assertEqual(historical["prior_redrawn_artifacts"], {
            "summary.md": {"execution_sha256": self.before["artifacts"]["summary.md"],
                           "prior_publication_sha256": self.published["summary.md"]}})
        self.assertEqual(after["publication_rebind"]["redrawn_artifacts"], {})
        self.assertEqual(h._execution_payload(after, historical["artifacts"]), h._payload(self.before))
        self.assertEqual(h._canonical_sha256(h._execution_payload(after, historical["artifacts"])),
                         historical["payload_sha256"])
        for key in ("binding", "execution", "environment", "observations", "results"):
            self.assertEqual(after[key], self.before[key])
        self.assertEqual(validated, [(self.before, {"artifact_sha256": self.before["artifacts"]})])

    def test_default_historical_publication_map_is_unchanged(self):
        historical = self.helper._historical_execution(self.before, self.prior)
        self.assertEqual(historical["prior_publication_artifacts"], self.before["artifacts"])
        self.assertEqual(historical["prior_redrawn_artifacts"], {})

    def test_csv_and_svg_changes_are_rejected_even_when_closed(self):
        for name in ("response.csv", "diagram.svg"):
            with self.subTest(name=name):
                (self.directory / name).write_bytes(self.raw_files[name] + b"changed")
                published = dict(self.published)
                published[name] = self.sha((self.directory / name).read_bytes())
                self.transition = self.closed_transition(published)
                with self.assertRaisesRegex(RuntimeError, "not Markdown-only"):
                    self.prepare()
                (self.directory / name).write_bytes(self.raw_files[name])

    def test_changed_artifact_keysets_are_rejected(self):
        for remove in (True, False):
            published = dict(self.published)
            if remove:
                published.pop("summary.md")
            else:
                published["extra.md"] = self.sha(b"extra")
            self.transition = dict(self.closed_transition(self.published), artifacts=published)
            with self.subTest(remove=remove):
                with self.assertRaisesRegex(RuntimeError, "not Markdown-only"):
                    self.prepare()

    def test_published_bytes_must_match_closed_hashes(self):
        (self.directory / "summary.md").write_bytes(b"unexpected wrapper drift")
        with self.assertRaisesRegex(RuntimeError, "artifact bytes changed"):
            self.prepare()

    def test_prior_manifest_and_historical_map_cannot_drift(self):
        self.transition["historical_execution"]["artifacts"] = dict(self.published)
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.transition = self.closed_transition(self.published)
        self.path.write_text(json.dumps(self.before) + "\n")
        with self.assertRaisesRegex(RuntimeError, "exact trusted prior"):
            self.prepare()


if __name__ == '__main__':
    unittest.main(verbosity=2)
