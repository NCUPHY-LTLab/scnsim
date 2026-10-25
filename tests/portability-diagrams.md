---
title: "Paths, diagrams, and teaching outputs"
lang: en
sidebar: testing
---

[English](portability-diagrams.md) · [繁體中文](zh-TW/portability-diagrams.md)

## Paths across platforms and storage modes

Path cases cover native filesystem paths, file URIs, Windows drive and UNC conversion, package-source provenance, and Memory/Workspace path lifetimes. Unreadable required paths retain their strict failure cause, catching double decoding and platform-specific path interpretation errors.

Source: [test_path_portability.py](test_path_portability.py).

## Manual circuit diagrams

Manual drawings are checked against current circuit declarations. Presentation settings must not change connectivity, `show` requires a current declaration, and wrong wire endpoints, MTL pins, missing crossing jumps, or layouts from another plan are rejected; these checks catch drawings that do not represent the declared circuit.

Source: [test_manual_diagrams.py](test_manual_diagrams.py).

## Notebooks and course helpers

Notebook and course-helper cases preserve cell and kernel identity while removing a machine-local kernel path, prepare declared chapter requests without solving, and export certificates for authored diagrams. Publication-binding cases preserve foreign namespaces and keep cleanup failures from replacing the primary error; together these checks catch nonportable notebooks, accidental solves during documentation preparation, and unsafe cleanup.

Sources: [test_portable_notebook.py](test_portable_notebook.py), [test_engineer_chapter3_generation.py](test_engineer_chapter3_generation.py), [test_engineer_chapter4_generation.py](test_engineer_chapter4_generation.py), and [test_engineer_publication_binding.py](test_engineer_publication_binding.py).
