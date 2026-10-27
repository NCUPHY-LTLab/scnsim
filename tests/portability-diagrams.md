---
title: "Paths, diagrams, and teaching outputs"
lang: en
sidebar: testing
---


This group tests portable paths and authored presentation artifacts around declared circuits. A diagram or notebook-preparation test does not by itself validate CAD geometry or run a physical solver.

## File paths across platforms

The path fixture creates and reads temporary files whose directory names include spaces, Traditional Chinese characters, a literal percent sign, and the text `%20`. It checks a native path and a `file:` URI with an empty or `localhost` authority. URI path decoding happens once: a literal `%20` in the name must remain literal, and the resolved path and file contents must match.

1. Create and read a temporary file named `內容 %20.txt`; compare the native path with its `file:` URI.
2. On Windows, check drive-letter and UNC path conversion. Other platforms skip these platform-specific cases; the test checks path strings and does not access a network share.
3. Make a required path unreadable and confirm that the original failure cause is retained. Source: [`test_path_portability.py`](test_path_portability.py).

## Drawings must represent the declared circuit

A single-resonator drawing fixture uses 110 fF, 5.8 nH, 6 fF coupling, and a 50-ohm port. It renders a selected 120 fF point with automatic layout and dark presentation. The schematic audit checks that parameter and presentation changes do not alter the plan’s declared connectivity; the `show` operation must use the current declaration.

1. Draw the declared resonator at 120 fF, then confirm the parameter and presentation choices keep the plan’s declared connectivity and the `show` operation uses the current declaration.
2. For the 1.6 mm, two-section readout/filter multiconductor transmission line (MTL), preserve ordered head and tail ports. Swap a physical capacitor terminal or connect an MTL pin to the wrong endpoint; the schematic audit rejects the drawing. A missing crossing jumper is also rejected, while a previously declared drawing remains available when a later layout is invalid.

These checks concern connectivity, not conductor dimensions. Source: [`test_manual_diagrams.py`](test_manual_diagrams.py).

## Notebook and course preparation

The notebook portability converter removes only the machine-local `kernelspec.path`. It preserves cell content, portable kernel identity, and all other parsed fields. The course-preparation cases execute the declared chapter inputs and Direct trace where specified, then stop before solving; authored circuit drawings can export their audit certificates.

Publication-binding cases preserve foreign namespaces and make cleanup errors secondary to the original error. Together, these tests catch a notebook tied to one machine, documentation preparation that unexpectedly starts a solver, or cleanup that hides the actual failure. Sources: [`test_portable_notebook.py`](test_portable_notebook.py), [`test_engineer_chapter3_generation.py`](test_engineer_chapter3_generation.py), [`test_engineer_chapter4_generation.py`](test_engineer_chapter4_generation.py), and [`test_engineer_publication_binding.py`](test_engineer_publication_binding.py).
