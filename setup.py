"""Build the sole ordered compensated CPU kernel; runtime owns its buffers.

Prebuilt wheels require no end-user compiler. Source builds require C++17.
The extension uses CPython's buffer API, not NumPy's ABI, and has no Python
numerical fallback. Strict arithmetic flags preserve the EFT operation order.
"""
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    'scnsim._native._compensated_cpu',
    sources=['src/scnsim/_native/compensated_cpu.cpp'],
    language='c++',
    extra_compile_args=['-std=c++17', '-fno-fast-math', '-ffp-contract=off'],
)])
