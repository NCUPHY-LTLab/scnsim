---
title: "Circuit inputs and components"
lang: en
sidebar: testing
---

[English](circuit-data.md) · [繁體中文](zh-TW/circuit-data.md)

## RLGC and Q2D inputs

Tests build ordered RLGC matrix inputs and load Q2D source data. Values must remain immutable and ordered, matrices must satisfy physical requirements, and Q2D mappings must be bijective and match their content digest; these checks catch mutation, nonphysical matrices, incomplete mappings, and mismatched source files.

Source: [test_rlgc.py](test_rlgc.py).

## Catalog components and coupling

Component fixtures assemble junctions, capacitors, SQUIDs, resonators, and coupled inductors from catalog declarations. Provenance is captured once, expansions remain explicit, non-positive-definite inductance graphs are rejected, and valid coupling is compared with an independent stiffness-matrix stamp; these checks catch factory reinvocation and expansion errors.

Source: [test_component_models.py](test_component_models.py).
