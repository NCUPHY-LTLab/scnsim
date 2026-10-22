# Profiling the M0 benchmark worker

Use `run.py` for timing measurements and `profile.py` for a single diagnostic
worker. A profile adds overhead, so keep its output separate from the repeated
measurements used for cold and warm summaries.

Run commands from the repository root in the same isolated environment used
for the benchmark. The commands below profile the C1 direct sweep in memory;
replace the case and operation with an existing pair from `circuits.py` when
profiling another workload.

## Repeated timing measurements

The regular harness starts five fresh workers for cold samples. Its first
worker also performs five warm repetitions on the same Run. The summary reports
the samples, median, and interquartile range for each group. Golden capture runs
separately and is excluded from the timing summary.

```sh
python benchmarks/run.py \
  --case C1 --operation direct --storage memory \
  --points 401 --chunk-size 64 --repetitions 5 \
  --output /tmp/scnsim-m0-c1-direct
```

Use a new output directory for each invocation. The worker records machine and
load readings before and after the operation, Python and package versions,
requested `cpu_threads`, process affinity, compilation events, call timings,
and peak RSS. The normal run writes `summary.json`, per-worker `worker.json`,
and separate stdout and stderr files. `cpu_threads` is the declared SCNSim
setting; affinity is the process's allowed CPU set. These fields do not report
the discovered size of a native thread pool.

## One profiled operation

`profile.py` forwards the normal worker options after `--`, adds `--worker`,
and runs exactly one worker. `--warm-repetitions` defaults to `0`; set it to
`5` to profile the initial cold call and five warm calls on the same Run. The
resulting profile includes both cold and warm work; it is not a warm-only
profile. Keep its `--profile-output` path separate from the worker's `--output`
directory. Put profiler options before the literal `--` and `run.py` options
after it.

### Python main thread with cProfile

```sh
python benchmarks/profile.py cprofile \
  --profile-output /tmp/scnsim-m0-profiles/c1-direct.pstats \
  --warm-repetitions 5 -- \
  --case C1 --operation direct --storage memory \
  --points 401 --chunk-size 64 \
  --output /tmp/scnsim-m0-profile-workers/c1-direct
```

The `.pstats` file covers Python calls on the main thread. It does not attribute
work running on XLA execution threads. Inspect it with Python's `pstats` module.

### Python and XLA threads with py-spy

```sh
python benchmarks/profile.py py-spy \
  --profile-output /tmp/scnsim-m0-profiles/c1-direct-threads.svg \
  --warm-repetitions 5 -- \
  --case C1 --operation direct --storage memory \
  --points 401 --chunk-size 64 \
  --output /tmp/scnsim-m0-profile-workers/c1-direct-py-spy
```

This mode invokes `py-spy record --threads`. `py-spy` must already be available
in the environment; the helper does not install it. Record `py-spy --version`
with the profile because it is an external executable, not one of the packages
listed in `worker.json`.

### JAX profiler trace

```sh
python benchmarks/profile.py jax-trace \
  --profile-output /tmp/scnsim-m0-profiles/c1-direct-jax-trace \
  --warm-repetitions 5 -- \
  --case C1 --operation direct --storage memory \
  --points 401 --chunk-size 64 \
  --output /tmp/scnsim-m0-profile-workers/c1-direct-jax-trace
```

The wrapper forwards the resolved `--profile-output` path to the worker as
`--jax-trace-output`. The worker applies its runtime settings, including any
`--cpu-threads` option, and completes initial host/backend setup before the
trace begins. The trace excludes that initial setup and includes the original
compile and benchmark calls, warm repetitions, and `Run.close()`. The worker
also writes `worker.json` under its `--output` directory with machine,
version, call, compilation, and peak-RSS metadata.

Use the ordinary `run.py` summary for cold and warm timing comparisons. A
single profiled worker is an attribution sample, not a replacement timing
measurement or a performance pass/fail result.
