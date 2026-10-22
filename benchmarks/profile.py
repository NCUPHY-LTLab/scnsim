"""Run one benchmark worker under a focused profiler."""

from __future__ import annotations

import argparse
import importlib
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


_RUNNER = Path(__file__).resolve().with_name("run.py")


def _load_cprofile():
    """Import stdlib cProfile without resolving this script as stdlib profile."""
    script_directory = Path(__file__).resolve().parent
    original_path = sys.path[:]
    try:
        sys.path[:] = [
            entry
            for entry in original_path
            if Path(entry or ".").resolve() != script_directory
        ]
        return importlib.import_module("cProfile")
    finally:
        sys.path[:] = original_path


def _worker_argv(arguments: Sequence[str], warm_repetitions: int) -> list[str]:
    if not _RUNNER.is_file():
        raise FileNotFoundError(f"benchmark runner not found: {_RUNNER}")
    if any(
        item == "--worker"
        or item == "--warm-repetitions"
        or item.startswith("--warm-repetitions=")
        for item in arguments
    ):
        raise ValueError("profile.py owns --warm-repetitions; do not pass it after --")
    return [
        str(_RUNNER),
        "--worker",
        "--warm-repetitions",
        str(warm_repetitions),
        *arguments,
    ]


def _run_in_process(arguments: Sequence[str], warm_repetitions: int) -> None:
    previous_argv = sys.argv
    sys.argv = _worker_argv(arguments, warm_repetitions)
    try:
        runpy.run_path(str(_RUNNER), run_name="__main__")
    finally:
        sys.argv = previous_argv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile one benchmarks/run.py worker invocation."
    )
    parser.add_argument(
        "mode", choices=("cprofile", "jax-trace", "py-spy")
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        required=True,
        help="pstats file, JAX trace directory, or py-spy SVG, according to mode",
    )
    parser.add_argument(
        "--warm-repetitions",
        type=int,
        default=0,
        help="warm calls in the same worker after its initial cold call",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        separator = arguments.index("--")
    except ValueError:
        parser.error("separate profile options from run.py options with --")
    args = parser.parse_args(arguments[:separator])
    worker_args = arguments[separator + 1 :]
    if not worker_args:
        parser.error("pass the benchmarks/run.py worker options after --")

    output = args.profile_output.expanduser().resolve()
    try:
        command = _worker_argv(worker_args, args.warm_repetitions)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "cprofile":
        profiler = _load_cprofile().Profile()
        try:
            profiler.runcall(_run_in_process, worker_args, args.warm_repetitions)
        finally:
            profiler.dump_stats(str(output))
        return 0

    if args.mode == "jax-trace":
        if any(
            item == "--jax-trace-output"
            or item.startswith("--jax-trace-output=")
            for item in worker_args
        ):
            parser.error("profile.py owns --jax-trace-output; omit it after --")
        _run_in_process(
            [*worker_args, "--jax-trace-output", str(output)],
            args.warm_repetitions,
        )
        return 0

    py_spy = shutil.which("py-spy")
    if py_spy is None:
        parser.error("py-spy is not installed in this environment")
    traced_command = [
        py_spy,
        "record",
        "--threads",
        "--output",
        str(output),
        "--",
        sys.executable,
        *command,
    ]
    return subprocess.run(traced_command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
