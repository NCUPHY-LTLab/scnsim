"""Build the sole ordered compensated CPU kernel; runtime owns its buffers.

Prebuilt wheels require no end-user compiler. Source builds require C++17.
The extension uses CPython's buffer API, not NumPy's ABI, and has no Python
numerical fallback. Strict arithmetic flags preserve the EFT operation order.
"""
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OrderedBuildExt(build_ext):
    """Choose strict arithmetic flags from the actual extension compiler."""

    def build_extensions(self):
        flags = (['/std:c++17', '/fp:strict'] if self.compiler.compiler_type == 'msvc'
                 else ['-std=c++17', '-fno-fast-math', '-ffp-contract=off'])
        for extension in self.extensions:
            extension.extra_compile_args = list(flags)
        self.announce(f'SCNSim ordered kernel compiler={self.compiler.compiler_type} flags={flags!r}', level=2)
        super().build_extensions()

setup(cmdclass={'build_ext': OrderedBuildExt}, ext_modules=[Extension(
    'scnsim._native._compensated_cpu',
    sources=['src/scnsim/_native/compensated_cpu.cpp'],
    language='c++',
)])
