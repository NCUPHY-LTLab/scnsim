"""Build the sole ordered compensated CPU kernel; runtime owns its buffers.

Prebuilt wheels require no end-user compiler. Source builds require C++17.
The extension uses CPython's buffer API, not NumPy's ABI, and has no Python
numerical fallback. Strict arithmetic flags preserve the EFT operation order.
Knowledge copies are generated from the producer inventory before wheel/sdist
file discovery, without importing SCNSim or executing documentation directives.
"""
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

# PEP 517 run_setup may execute this file without its directory on sys.path.
_knowledge_spec = spec_from_file_location(
    '_scnsim_build_agent_knowledge',
    Path(__file__).resolve().parent / 'build_support' / 'agent_knowledge.py',
)
_knowledge_builder = module_from_spec(_knowledge_spec)
_knowledge_spec.loader.exec_module(_knowledge_builder)
generate = _knowledge_builder.generate


class KnowledgeBuildPy(build_py):
    def run(self):
        generate(Path(__file__).resolve().parent)
        super().run()


class KnowledgeSdist(sdist):
    def run(self):
        generate(Path(__file__).resolve().parent)
        super().run()


class OrderedBuildExt(build_ext):
    """Choose strict arithmetic flags from the actual extension compiler."""

    def build_extensions(self):
        flags = (['/std:c++17', '/fp:strict'] if self.compiler.compiler_type == 'msvc'
                 else ['-std=c++17', '-fno-fast-math', '-ffp-contract=off'])
        for extension in self.extensions:
            extension.extra_compile_args = list(flags)
        print(f'SCNSim ordered kernel compiler={self.compiler.compiler_type} flags={flags!r}', flush=True)
        super().build_extensions()

setup(cmdclass={'build_ext': OrderedBuildExt, 'build_py': KnowledgeBuildPy,
                'sdist': KnowledgeSdist}, ext_modules=[Extension(
    'scnsim._native._compensated_cpu',
    sources=['src/scnsim/_native/compensated_cpu.cpp'],
    language='c++',
)])
