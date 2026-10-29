"""Human-accepted contract tests for explicitly declared circuit diagrams.

The Plans and geometry recipes below follow the current Engineer Chapter 1,
Chapter 4, and Chapter 7 source examples.  These tests cover manual authoring
and detached presentation only; they do not execute a numerical backend.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

from examples.engineer.diagram_layouts import engineer_layout
from scnsim import (
    CircuitDiagramSpec,
    CircuitPlan,
    DiagramEndpoint,
    DiagramJump,
    ParameterDefinitions,
    ParameterSet,
    ParameterSpec,
    RLGC,
    RLGCParameterSpec,
    SCNSimValidationError,
    Theme,
    components,
    units as u,
)
from scnsim.schematic import SchematicLayout


def _chapter_one_plan(*, plan_id="engineer_coupled_lc"):
    """The parameterized Chapter 1 coupled grounded-LC Plan."""
    inputs = ParameterDefinitions(id="engineer_lc_design")
    capacitance = inputs.parameter(
        id="capacitance",
        baseline=110.0 * u.fF,
        spec=ParameterSpec(unit=u.fF),
    )
    plan = CircuitPlan(id=plan_id)
    resonator = plan.subsystem(id="resonator")
    capacitor = resonator.add(
        components.capacitor(id="capacitor", capacitance=capacitance)
    )
    inductor = resonator.add(
        components.inductor(id="inductor", inductance=5.8 * u.nH)
    )
    signal = resonator.bus(id="signal")
    resonator.parallel(
        id="parallel_lc",
        start=signal,
        branches=((capacitor,), (inductor,)),
        end=resonator.ground,
    )
    terminal = resonator.expose_pin(id="terminal", at=signal)
    resonator.expose_coordinate(id="signal", at=signal)
    root_signal = plan.bus(id="signal_boundary")
    coupler = plan.add(
        components.capacitor(id="coupling_capacitor", capacitance=6.0 * u.fF)
    )
    plan.series(
        id="coupling", start=root_signal, elements=(coupler,), end=terminal
    )
    plan.add_port(
        id="signal_in",
        at=root_signal,
        role="terminated",
        reference_impedance=50.0 * u.ohm,
    )
    return plan, capacitance


def _chapter_seven_plan():
    """The ordered N=2 RLGC line and four Chapter 7 signal Ports."""
    inputs = ParameterDefinitions(id="coupled_line_design")
    baseline_rlgc = RLGC(
        conductors=("readout", "filter"),
        reference_conductor="ground",
        resistance_per_length=[[0.18, 0.0], [0.0, 0.22]] * u.ohm / u.m,
        inductance_per_length=[[420.0, 75.0], [75.0, 395.0]] * u.nH / u.m,
        conductance_per_length=[[0.0, 0.0], [0.0, 0.0]] * u.S / u.m,
        capacitance_per_length=[[175.0, -22.0], [-22.0, 168.0]] * u.pF / u.m,
    )
    length = inputs.parameter(
        id="line_length", baseline=1.6 * u.mm, spec=ParameterSpec(unit=u.mm)
    )
    rlgc = inputs.parameter(
        id="line_rlgc",
        baseline=baseline_rlgc,
        spec=RLGCParameterSpec(
            conductors=("readout", "filter"), reference_conductor="ground"
        ),
    )
    plan = CircuitPlan(id="ordered_n2_line")
    line = plan.add(
        components.transmission_line(
            id="coupled", length=length, rlgc=rlgc, n_sections=8
        )
    )
    buses = {
        name: plan.bus(id=name)
        for name in ("readout_head", "readout_tail", "filter_head", "filter_tail")
    }
    plan.series(
        id="readout_conductor",
        start=buses["readout_head"],
        elements=(
            line.between(
                line.pin("head", conductor="readout"),
                line.pin("tail", conductor="readout"),
            ),
        ),
        end=buses["readout_tail"],
    )
    plan.link(
        id="filter_head_binding",
        endpoints=(buses["filter_head"], line.pin("head", conductor="filter")),
    )
    plan.link(
        id="filter_tail_binding",
        endpoints=(buses["filter_tail"], line.pin("tail", conductor="filter")),
    )
    for name in ("readout_head", "readout_tail", "filter_head", "filter_tail"):
        plan.add_port(
            id=name,
            at=buses[name],
            role="terminated",
            reference_impedance=50.0 * u.ohm,
        )
    return plan


def _chapter_four_plan():
    """The existing Chapter 4 feedline/readout/floating capstone Plan."""
    plan = CircuitPlan(id="floating_probe_course")
    inputs = ParameterDefinitions(id="floating_probe_design")
    feedline_rlgc = RLGC(
        conductors=("signal",),
        reference_conductor="ground",
        resistance_per_length=[[0.0]] * u.ohm / u.m,
        inductance_per_length=[[420.0]] * u.nH / u.m,
        conductance_per_length=[[0.0]] * u.S / u.m,
        capacitance_per_length=[[175.0]] * u.pF / u.m,
    )
    line_rlgc = inputs.parameter(
        id="line_rlgc",
        baseline=feedline_rlgc,
        spec=RLGCParameterSpec(conductors=("signal",), reference_conductor="ground"),
    )
    readout_capacitance = inputs.parameter(
        id="readout_capacitance", baseline=110.0 * u.fF,
        spec=ParameterSpec(unit=u.fF),
    )
    readout_inductance = inputs.parameter(
        id="readout_inductance", baseline=5.8 * u.nH,
        spec=ParameterSpec(unit=u.nH),
    )
    mutual_capacitance = inputs.parameter(
        id="mutual_capacitance", baseline=16.0 * u.fF,
        spec=ParameterSpec(unit=u.fF),
    )
    mutual_inductance = inputs.parameter(
        id="mutual_inductance", baseline=7.0 * u.nH,
        spec=ParameterSpec(unit=u.nH),
    )
    plus_shunt_capacitance = inputs.parameter(
        id="plus_shunt_capacitance", baseline=45.0 * u.fF,
        spec=ParameterSpec(unit=u.fF),
    )
    minus_shunt_capacitance = inputs.parameter(
        id="minus_shunt_capacitance", baseline=42.0 * u.fF,
        spec=ParameterSpec(unit=u.fF),
    )

    feedline = plan.subsystem(id="feedline")
    left = feedline.add(components.transmission_line(
        id="left", length=1.0 * u.mm, rlgc=line_rlgc, n_sections=1
    ))
    right = feedline.add(components.transmission_line(
        id="right", length=1.0 * u.mm, rlgc=line_rlgc, n_sections=1
    ))
    input_bus, middle_bus, output_bus = (
        feedline.bus(id=name) for name in ("input", "middle", "output")
    )
    feedline.series(
        id="left_section", start=input_bus,
        elements=(left.between(left.pin("head", conductor="signal"), left.pin("tail", conductor="signal")),),
        end=middle_bus,
    )
    feedline.series(
        id="right_section", start=middle_bus,
        elements=(right.between(right.pin("head", conductor="signal"), right.pin("tail", conductor="signal")),),
        end=output_bus,
    )
    feedline_input_pin = feedline.expose_pin(id="input", at=input_bus)
    feedline_coupling_pin = feedline.expose_pin(id="tap", at=middle_bus)
    feedline_output_pin = feedline.expose_pin(id="output", at=output_bus)

    readout = plan.subsystem(id="readout")
    readout_capacitor = readout.add(components.capacitor(
        id="capacitor", capacitance=readout_capacitance
    ))
    readout_inductor = readout.add(components.inductor(
        id="inductor", inductance=readout_inductance
    ))
    readout_bus = readout.bus(id="node")
    readout.parallel(
        id="parallel_lc", start=readout_bus,
        branches=((readout_capacitor,), (readout_inductor,)), end=readout.ground,
    )
    readout_terminal = readout.expose_pin(id="readout_node", at=readout_bus)

    floating = plan.subsystem(id="floating")
    plus_bus, minus_bus = floating.bus(id="plus"), floating.bus(id="minus")
    mutual_cap = floating.add(components.capacitor(
        id="mutual_cap", capacitance=mutual_capacitance
    ))
    mutual_ind = floating.add(components.inductor(
        id="mutual_ind", inductance=mutual_inductance
    ))
    plus_shunt = floating.add(components.capacitor(
        id="plus_shunt", capacitance=plus_shunt_capacitance
    ))
    minus_shunt = floating.add(components.capacitor(
        id="minus_shunt", capacitance=minus_shunt_capacitance
    ))
    floating.parallel(
        id="mutual_network", start=plus_bus,
        branches=((mutual_cap,), (mutual_ind,)), end=minus_bus,
    )
    floating.branch(
        id="plus_shunt", at=plus_bus, elements=(plus_shunt,), end=floating.ground
    )
    floating.branch(
        id="minus_shunt", at=minus_bus, elements=(minus_shunt,), end=floating.ground
    )
    plus_1 = floating.expose_pin(id="plus_1", at=plus_bus)
    plus_2 = floating.expose_pin(id="plus_2", at=plus_bus)
    minus_1 = floating.expose_pin(id="minus_1", at=minus_bus)
    minus_2 = floating.expose_pin(id="minus_2", at=minus_bus)

    feedline_in_bus = plan.bus(id="feedline_in")
    feedline_out_bus = plan.bus(id="feedline_out")
    readout_root_bus = plan.bus(id="readout_node")
    floating_plus_bus = plan.bus(id="floating_plus")
    floating_minus_bus = plan.bus(id="floating_minus")
    plan.link(id="feedline_input_child", endpoints=(feedline_in_bus, feedline_input_pin))
    plan.link(id="feedline_output_child", endpoints=(feedline_out_bus, feedline_output_pin))
    plan.link(id="readout_child", endpoints=(readout_root_bus, readout_terminal))
    plan.link(id="floating_plus_child", endpoints=(floating_plus_bus, plus_2))
    plan.link(id="floating_minus_child", endpoints=(floating_minus_bus, minus_2))
    feedline_coupler = plan.add(components.capacitor(
        id="feedline_readout_coupler", capacitance=6.0 * u.fF
    ))
    plus_coupler = plan.add(components.capacitor(
        id="readout_to_floating_plus", capacitance=4.0 * u.fF
    ))
    minus_coupler = plan.add(components.capacitor(
        id="readout_to_floating_minus", capacitance=3.0 * u.fF
    ))
    plan.series(
        id="feedline_readout", start=feedline_coupling_pin,
        elements=(feedline_coupler,), end=readout_root_bus,
    )
    plan.series(
        id="readout_floating_plus", start=readout_root_bus,
        elements=(plus_coupler,), end=plus_1,
    )
    plan.series(
        id="readout_floating_minus", start=readout_root_bus,
        elements=(minus_coupler,), end=minus_1,
    )
    for name, bus, role in (
        ("feedline_in", feedline_in_bus, "terminated"),
        ("feedline_out", feedline_out_bus, "terminated"),
        ("floating_probe_plus", floating_plus_bus, "nonloading_probe"),
        ("floating_probe_minus", floating_minus_bus, "nonloading_probe"),
    ):
        plan.add_port(
            id=name, at=bus, role=role, reference_impedance=50.0 * u.ohm
        )
    return plan


def _physical_terminal(preparation, scope_path, component_path, pin):
    scope = next(
        row for row in preparation.scopes
        if row.ref.scope_path == tuple(scope_path)
    )
    return next(
        row["ref"]
        for row in scope.contacts
        if row["source"]["kind"] == "physical_terminal"
        and tuple(row["source"]["path"]) == tuple(component_path)
        and row["source"]["pin"] == pin
    )


def _measured_contact(preparation, layout, target):
    """Return a captured physical contact in the declared root placement."""
    for (body, rotation), fragment in preparation.native.items():
        pose = layout.root.poses.get(body)
        if pose is None or pose.rotation != rotation:
            continue
        row = fragment.contacts.get(target)
        if row is not None:
            point = row["point"]
            return (
                (point.x + pose.origin[0], point.y + pose.origin[1]),
                row["side"],
            )
    raise AssertionError("selected contact has no measured placed native body")


def _outward_from_route(route, *, start, distance):
    point = route.waypoints[0] if start else route.waypoints[-1]
    neighbor = route.waypoints[1] if start else route.waypoints[-2]
    dx, dy = neighbor[0] - point[0], neighbor[1] - point[1]
    length = abs(dx) + abs(dy)
    if not length or (dx and dy):
        raise AssertionError("authored route endpoint must have a cardinal segment")
    return (
        point[0] + distance * (dx / length),
        point[1] + distance * (dy / length),
    )


def _outward_from_contact(point, side, distance):
    vectors = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "top": (0.0, 1.0),
        "bottom": (0.0, -1.0),
    }
    dx, dy = vectors[side]
    return (point[0] + distance * dx, point[1] + distance * dy)


def _replace_routes(layout, updates, *, jumps=()):
    routes = tuple(
        replace(route, **updates[route.id]) if route.id in updates else route
        for route in layout.root.routes
    )
    root = replace(
        layout.root,
        routes=routes,
        jumps=(*layout.root.jumps, *jumps),
    )
    return SchematicLayout(layout.preparation, root)


class ManualDiagramContractTests(unittest.TestCase):
    def test_parameter_point_and_presentation_options_keep_plan_connectivity(self):
        plan, capacitance = _chapter_one_plan()
        selected = ParameterSet({capacitance: 120.0 * u.fF})
        auto_spec = CircuitDiagramSpec(theme=Theme.AUTO)
        auto_preparation = plan.prepare_schematic(auto_spec, parameters=selected)
        auto_layout = engineer_layout(auto_preparation, "coupled_lc")
        auto = plan.render_schematic(
            CircuitDiagramSpec(layout=auto_layout, theme=Theme.AUTO),
            parameters=selected,
        )

        dark_spec = CircuitDiagramSpec(theme=Theme.DARK)
        dark_preparation = plan.prepare_schematic(dark_spec, parameters=selected)
        dark_layout = engineer_layout(dark_preparation, "coupled_lc")
        dark = plan.render_schematic(
            CircuitDiagramSpec(layout=dark_layout, theme=Theme.DARK),
            parameters=selected,
        )

        self.assertEqual(auto.audit.representation, "authoring")
        self.assertEqual(auto.audit.plan_sha256, dark.audit.plan_sha256)
        self.assertEqual(auto.audit.connectivity_sha256, dark.audit.connectivity_sha256)
        self.assertEqual(
            auto_preparation.identity["parameters_sha256"],
            dark_preparation.identity["parameters_sha256"],
        )
        self.assertNotEqual(
            auto_preparation.identity["presentation_options_sha256"],
            dark_preparation.identity["presentation_options_sha256"],
        )

        baseline_preparation = plan.prepare_schematic(auto_spec)
        baseline_layout = engineer_layout(baseline_preparation, "coupled_lc")
        baseline = plan.render_schematic(
            CircuitDiagramSpec(layout=baseline_layout, theme=Theme.AUTO)
        )
        self.assertEqual(auto.audit.connectivity_sha256, baseline.audit.connectivity_sha256)
        self.assertNotEqual(
            auto_preparation.identity["parameters_sha256"],
            baseline_preparation.identity["parameters_sha256"],
        )

        held_audit = auto.audit.show().html
        plan.bus(id="later_source_change")
        self.assertEqual(auto.audit.show().html, held_audit)
        self.assertEqual(auto.layout.preparation.identity, auto_preparation.identity)

    def test_render_does_not_declare_and_show_requires_a_current_declaration(self):
        plan, _ = _chapter_one_plan()
        with self.assertRaises(SCNSimValidationError):
            plan.render_schematic(CircuitDiagramSpec())
        with self.assertRaises(SCNSimValidationError):
            plan.show_schematic()

        preparation = plan.prepare_schematic(CircuitDiagramSpec())
        layout = engineer_layout(preparation, "coupled_lc")
        rendered = plan.render_schematic(CircuitDiagramSpec(layout=layout))
        self.assertFalse(hasattr(plan, "_declared_schematic"))
        self.assertIsNotNone(rendered.audit)
        with self.assertRaises(SCNSimValidationError):
            plan.show_schematic()

        first_declaration = plan.declare_schematic(CircuitDiagramSpec(layout=layout))
        dark_preparation = plan.prepare_schematic(CircuitDiagramSpec(theme=Theme.DARK))
        dark_layout = engineer_layout(dark_preparation, "coupled_lc")
        replacement = plan.declare_schematic(
            CircuitDiagramSpec(layout=dark_layout, theme=Theme.DARK)
        )
        self.assertIsNot(first_declaration, replacement)
        self.assertIs(plan._declared_schematic[0], replacement)
        plan.show_schematic()
        extra = plan.add(components.capacitor(id="later_capacitor", capacitance=2.0 * u.fF))
        later_bus = plan.bus(id="later_bus")
        plan.series(id="later_branch", start=later_bus, elements=(extra,), end=plan.ground)
        with self.assertRaises(SCNSimValidationError):
            plan.show_schematic()

    def test_chapter_one_wrong_physical_wire_endpoint_is_rejected(self):
        plan, _ = _chapter_one_plan()
        preparation = plan.prepare_schematic(CircuitDiagramSpec())
        layout = engineer_layout(preparation, "coupled_lc")
        terminal_1 = _physical_terminal(
            preparation, (), ("coupling_capacitor",), "terminal_1"
        )
        terminal_2 = _physical_terminal(
            preparation, (), ("coupling_capacitor",), "terminal_2"
        )
        terminal_1_point, terminal_1_side = _measured_contact(
            preparation, layout, terminal_1
        )
        terminal_2_point, terminal_2_side = _measured_contact(
            preparation, layout, terminal_2
        )
        incoming = next(
            route for route in layout.root.routes
            if route.id == "measurement_to_coupler"
        )
        outgoing = next(
            route for route in layout.root.routes
            if route.id == "coupler_to_child"
        )
        access = preparation.metrics.terminal_stub
        source = incoming.waypoints[0]
        source_out = _outward_from_route(incoming, start=True, distance=access)
        target_out = _outward_from_contact(
            terminal_2_point, terminal_2_side, access
        )
        lower = layout.root.frame[1] + 2.0
        incoming_points = (
            source,
            source_out,
            (source_out[0], lower),
            (target_out[0], lower),
            target_out,
            terminal_2_point,
        )

        child_end = outgoing.waypoints[-1]
        child_end_out = _outward_from_route(
            outgoing, start=False, distance=access
        )
        terminal_1_out = _outward_from_contact(
            terminal_1_point, terminal_1_side, access
        )
        upper = layout.root.frame[3] - 2.0
        outgoing_points = (
            terminal_1_point,
            terminal_1_out,
            (terminal_1_out[0], upper),
            (child_end_out[0], upper),
            child_end_out,
            child_end,
        )
        malformed = _replace_routes(
            layout,
            {
                "measurement_to_coupler": {
                    "end": DiagramEndpoint(terminal_2),
                    "waypoints": incoming_points,
                },
                "coupler_to_child": {
                    "start": DiagramEndpoint(terminal_1),
                    "waypoints": outgoing_points,
                },
            },
        )
        with self.assertRaises(SCNSimValidationError) as raised:
            plan.render_schematic(CircuitDiagramSpec(layout=malformed))
        self.assertEqual(raised.exception.stage, "schematic_audit")

    def test_chapter_seven_ordered_mtl_wrong_pin_endpoint_is_rejected(self):
        plan = _chapter_seven_plan()
        preparation = plan.prepare_schematic(CircuitDiagramSpec())
        layout = engineer_layout(preparation, "ordered_mtl")
        readout_pin = _physical_terminal(
            preparation, (), ("coupled",), "head.readout"
        )
        filter_pin = _physical_terminal(
            preparation, (), ("coupled",), "head.filter"
        )
        readout_point, _ = _measured_contact(preparation, layout, readout_pin)
        filter_point, _ = _measured_contact(preparation, layout, filter_pin)
        readout_route = next(
            route for route in layout.root.routes
            if route.id == "readout_head_attachment"
        )
        filter_route = next(
            route for route in layout.root.routes
            if route.id == "filter_head_attachment"
        )
        readout_lane = (readout_point[0] - 4.0, readout_point[1])
        filter_lane = (filter_point[0] - 2.0, filter_point[1])
        crossed = _replace_routes(
            layout,
            {
                "readout_head_attachment": {
                    "end": DiagramEndpoint(filter_pin),
                    "waypoints": (
                        readout_route.waypoints[0],
                        (readout_lane[0], readout_route.waypoints[0][1]),
                        (readout_lane[0], filter_point[1]),
                        filter_point,
                    ),
                },
                "filter_head_attachment": {
                    "end": DiagramEndpoint(readout_pin),
                    "waypoints": (
                        filter_route.waypoints[0],
                        (filter_lane[0], filter_route.waypoints[0][1]),
                        (filter_lane[0], readout_point[1]),
                        readout_point,
                    ),
                },
            },
            jumps=(
                DiagramJump(
                    "readout_head_attachment",
                    (filter_lane[0], filter_point[1]),
                    "filter_head_attachment",
                ),
            ),
        )
        with self.assertRaises(SCNSimValidationError) as raised:
            plan.render_schematic(CircuitDiagramSpec(layout=crossed))
        self.assertEqual(raised.exception.stage, "schematic_audit")

    def test_chapter_four_missing_crossing_jump_fails_and_keeps_declared_drawing(self):
        plan = _chapter_four_plan()
        preparation = plan.prepare_schematic(CircuitDiagramSpec())
        layout = engineer_layout(preparation, "four_port_capstone")
        self.assertTrue(any(jump.route_id == "minus_coupler_child" for jump in layout.root.jumps))

        spec = CircuitDiagramSpec(layout=layout)
        declared = plan.declare_schematic(spec)
        without_crossing_jump = SchematicLayout(
            preparation,
            replace(
                layout.root,
                jumps=tuple(
                    jump for jump in layout.root.jumps
                    if jump.route_id != "minus_coupler_child"
                ),
            ),
        )
        with self.assertRaises(SCNSimValidationError):
            plan.declare_schematic(CircuitDiagramSpec(layout=without_crossing_jump))

        self.assertIs(plan._declared_schematic[0], declared)
        plan.show_schematic()

    def test_layout_from_another_plan_cannot_be_rendered(self):
        first, _ = _chapter_one_plan()
        second, _ = _chapter_one_plan(plan_id="different_plan")
        preparation = second.prepare_schematic(CircuitDiagramSpec())
        layout = engineer_layout(preparation, "coupled_lc")
        with self.assertRaises(SCNSimValidationError):
            first.render_schematic(CircuitDiagramSpec(layout=layout))


if __name__ == "__main__":
    unittest.main()
