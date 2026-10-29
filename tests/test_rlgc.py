"""Retained physical RLGC and strict Q2D source regressions.

Inputs and assertions migrate the original dev5 physical-value cases; these
checks do not accept new authoring or parameter identity semantics.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from scnsim import RLGC, SCNSimValidationError, load_q2d_rlgc, units as u

def _coupled_rlgc() -> RLGC:
    return RLGC(
        conductors=("readout", "filter"),
        reference_conductor="ground",
        resistance_per_length=[[0.18, 0.0], [0.0, 0.22]] * u.ohm / u.meter,
        inductance_per_length=[[420.0, 75.0], [75.0, 395.0]] * u.nH / u.meter,
        conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.siemens / u.meter,
        capacitance_per_length=[[175.0, -22.0], [-22.0, 168.0]] * u.pF / u.meter,
        extraction_frequency=5.0 * u.GHz,
    )

def _q2d_bytes() -> bytes:
    blocks = (
        ("Capacitance Matrix", ((175.0, -22.0), (-22.0, 168.0))),
        ("Conductance Matrix", ((0.0, 0.0), (0.0, 0.0))),
        ("Inductance Matrix", ((420.0, 75.0), (75.0, 395.0))),
        ("Resistance Matrix", ((0.18, 0.0), (0.0, 0.22))),
    )
    lines = [
        "Setup1:LastAdaptive",
        "Problem Type:  CG, RL",
        "C Units:pF/meter, G Units:mho/meter, R Units:ohm/meter, L Units:nH/meter",
        "Reduce Matrix:  Original",
        "Frequency:  5.0GHz",
        "",
    ]
    for title, matrix in blocks:
        lines.extend(
            (
                title,
                ",trace_a,trace_b",
                f"trace_a,{matrix[0][0]},{matrix[0][1]}",
                f"trace_b,{matrix[1][0]},{matrix[1][1]}",
                "",
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


class RLGCPhysicalValueTests(unittest.TestCase):
    def test_manual_rlgc_is_ordered_immutable_and_defensively_copied(self) -> None:
        rlgc = _coupled_rlgc()
        self.assertEqual(rlgc.conductors, ("readout", "filter"))
        self.assertEqual(rlgc.reference_conductor, "ground")
        self.assertEqual(rlgc.source, {"source_kind": "manual"})
        self.assertEqual(rlgc.extraction_frequency.to("gigahertz").magnitude, 5.0)
        fields = ("resistance_per_length", "inductance_per_length",
                  "conductance_per_length", "capacitance_per_length")
        inputs = {name: getattr(rlgc, name) for name in fields}
        expected = {name: np.array(value.magnitude, copy=True)
                    for name, value in inputs.items()}
        copied = RLGC(conductors=rlgc.conductors,
                      reference_conductor=rlgc.reference_conductor,
                      extraction_frequency=rlgc.extraction_frequency, **inputs)
        # Either readonly access or fresh writable copies can isolate the
        # immutable carrier; the public contract does not mandate array flags.
        for name, value in inputs.items():
            if np.asarray(value.magnitude).flags.writeable:
                value.magnitude[0, 0] = 0.0
            np.testing.assert_array_equal(getattr(copied, name).magnitude,
                                          expected[name])
        for name in fields:
            matrix = getattr(rlgc, name)
            if np.asarray(matrix.magnitude).flags.writeable:
                matrix.magnitude[0, 0] = 0.0
            np.testing.assert_array_equal(getattr(rlgc, name).magnitude,
                                          expected[name])
        with self.assertRaises(AttributeError):
            rlgc.other = object()

    def test_rlgc_rejects_nonphysical_or_noncanonical_matrices(self) -> None:
        kwargs = dict(
            conductors=("a", "b"),
            reference_conductor="ground",
            resistance_per_length=[[1.0, 0.0], [0.0, 1.0]] * u.ohm / u.meter,
            inductance_per_length=[[2.0, 0.2], [0.2, 2.0]] * u.nH / u.meter,
            conductance_per_length=[[1.0, -0.2], [-0.2, 1.0]] * u.siemens / u.meter,
            capacitance_per_length=[[2.0, -0.2], [-0.2, 2.0]] * u.pF / u.meter,
        )
        for field, value in (
            ("resistance_per_length", [[1.0, 0.1], [0.0, 1.0]] * u.ohm / u.meter),
            ("inductance_per_length", [[1.0, 2.0], [2.0, 1.0]] * u.nH / u.meter),
            ("conductance_per_length", [[1.0, 0.1], [0.1, 1.0]] * u.siemens / u.meter),
            ("capacitance_per_length", [[1.0, -2.0], [-2.0, 1.0]] * u.pF / u.meter),
        ):
            with self.subTest(field=field):
                invalid = dict(kwargs)
                invalid[field] = value
                with self.assertRaises(SCNSimValidationError):
                    RLGC(**invalid)
        with self.assertRaises(SCNSimValidationError):
            RLGC(**{**kwargs, "conductors": ("a", "a")})

    def test_q2d_loader_is_strict_bijective_and_digest_bound(self) -> None:
        raw = _q2d_bytes()
        with TemporaryDirectory() as temporary:
            path = Path(temporary, "rlgc.csv")
            path.write_bytes(raw)
            rlgc = load_q2d_rlgc(
                path,
                reference_conductor="ground",
                conductor_map={"trace_a": "readout", "trace_b": "filter"},
            )
            source = rlgc.source
            self.assertEqual(source["content_sha256"], sha256(raw).hexdigest())
            self.assertEqual(tuple(source["native_conductor_order"]), ("trace_a", "trace_b"))
            self.assertEqual(rlgc.conductors, ("readout", "filter"))
            self.assertEqual(rlgc.capacitance_per_length.to("pF / meter").magnitude[0, 1], -22.0)

            with self.assertRaises(SCNSimValidationError):
                load_q2d_rlgc(path, reference_conductor="ground", conductor_map={"trace_a": "only"})
            path.write_bytes(raw.replace(b"Problem Type:  CG, RL", b"Problem Type: CG, RL"))
            with self.assertRaises(SCNSimValidationError):
                load_q2d_rlgc(path, reference_conductor="ground")
            path.write_bytes(b"\xff")
            with self.assertRaises(SCNSimValidationError):
                load_q2d_rlgc(path, reference_conductor="ground")
