"""Explicit drawing recipes for the engineer course's named physical Plans.

This is ordinary example code, not a package layout service. Each recipe owns
its frames, individual body poses, wiring bends and junction arms. Shared code
only resolves fresh captured refs and translates measured native contacts;
there is no packing, route search, unknown-Plan completion or renderer fallback.
"""
from __future__ import annotations

import json

from scnsim import (
    DiagramCaption, DiagramEndpoint, DiagramJunction, DiagramLeader,
    DiagramPose, DiagramRoute, DiagramSide, DiagramJump, SchematicLayout,
    SchematicScopeLayout,
)


class _Scope:
    """Mechanical construction of one authored scope declaration."""

    def __init__(self, preparation, path, frame):
        self.preparation = preparation
        self.inventory = next(row for row in preparation.scopes
                              if row.ref.scope_path == tuple(path))
        self.frame = frame
        self.poses = {}
        self.boundaries = {}
        self.ports = {}
        self.grounds = {}
        self.children = {}
        self.points = {}
        self.junctions = []
        self.routes = []
        self.jumps = []
        self.label_positions = {}

    def ref(self, kind, path=(), id=None):
        return next(ref for ref in self.preparation.refs
                    if (record := json.loads(ref.key))["kind"] == kind
                    and tuple(record.get("path", ())) == tuple(path)
                    and (id is None or record.get("id") == id))

    def pin(self, path, id):
        return next(row["ref"] for row in self.inventory.contacts
                    if row["source"]["kind"] == "physical_terminal"
                    and tuple(row["source"]["path"]) == tuple(path)
                    and row["source"]["pin"] == id)

    def boundary(self, path, id):
        return self.ref("contact", path, id)

    def native(self, target, at, rotation=0, *, contact=None):
        fragment = self.preparation.native[target, rotation]
        # The recipe chooses the contact location; measured body extent is
        # never stretched to reach it. This translation creates no new geometry.
        anchor = fragment.contacts[target if contact is None and target.kind in
                                   ("port", "ground") else contact] if contact is not None or target.kind in ("port", "ground") else None
        origin = at if anchor is None else (at[0] - anchor["point"].x,
                                            at[1] - anchor["point"].y)
        pose = DiagramPose(origin, rotation)
        collection = self.ports if target.kind == "port" else self.grounds if target.kind == "ground" else self.poses
        collection[target] = pose
        for ref, row in fragment.contacts.items():
            self.points[ref] = (row["point"].x + origin[0], row["point"].y + origin[1])
        return target

    def child(self, child_scope, layout, origin):
        measured = self.preparation.measure_scope(child_scope, layout)
        self.children[child_scope] = layout
        self.poses[child_scope] = DiagramPose(origin)
        for ref, row in measured.contacts.items():
            self.points[ref] = (row["point"].x + origin[0], row["point"].y + origin[1])

    def expose(self, ref, at):
        self.boundaries[ref] = at
        self.points[ref] = at

    def junction(self, id, at, *arms):
        self.junctions.append(DiagramJunction(id, at, tuple(DiagramSide(arm) for arm in arms)))
        # Current intrinsic junction arm length; this is an explicit course
        # coordinate, not a tunable package clearance or geometry search.
        for arm, delta in {"left": (-.55, 0), "right": (.55, 0),
                           "top": (0, .55), "bottom": (0, -.55)}.items():
            if arm in arms:
                self.points[id, arm] = (at[0] + delta[0], at[1] + delta[1])

    def wire(self, id, start, end, *bends):
        endpoint = lambda ref: DiagramEndpoint(*ref) if isinstance(ref, tuple) else DiagramEndpoint(ref)
        self.routes.append(DiagramRoute(id, endpoint(start), endpoint(end),
                                       (self.points[start], *bends, self.points[end])))

    def labels(self, **positions):
        # Entries are explicit label text -> (actual contact, knee, end).
        self.label_positions.update(positions)

    def finish(self, *, caption_positions=None):
        captions = []
        for row in self.inventory.captions:
            target = row["target"]
            if caption_positions is not None and row["ref"] in caption_positions:
                at = caption_positions[row["ref"]]
            elif target in self.poses or target in self.ports or target in self.grounds:
                pose = {**self.poses, **self.ports, **self.grounds}[target]
                value = self.preparation.native[target, pose.rotation].scene
                for part in row["text_path"]:
                    value = value[part] if isinstance(part, int) else getattr(value, part)
                at = (value.origin.x + pose.origin[0], value.origin.y + pose.origin[1])
            elif json.loads(row["ref"].key)["source_kind"] == "scope_title":
                at = (self.frame[0] + 2, self.frame[3] - 2)
            else:
                # Every exposed label has its own explicitly declared boundary.
                x, y = self.points[target]
                at = (x - (row["run"].bounds.xmax - row["run"].bounds.xmin) - .8 if x == self.frame[2] else x + .8, y - 1 if x == self.frame[2] else y + .8)
            captions.append(DiagramCaption(row["ref"], at))
        leaders = []
        for row in self.inventory.analysis_labels:
            contact, knee, end = self.label_positions[row["text"]]
            leaders.append(DiagramLeader(row["ref"], contact, knee, end))
        return SchematicScopeLayout(
            scope=self.inventory.ref, frame=self.frame, poses=self.poses,
            boundaries=self.boundaries, ports=self.ports, grounds=self.grounds,
            junctions=tuple(self.junctions), routes=tuple(self.routes), jumps=tuple(self.jumps),
            captions=tuple(captions), leaders=tuple(leaders), couplings=(),
            children=self.children,
        )


def _lc(preparation, path, *, boundary_names=(), bus_label=None):
    """The course's two explicit parallel LC branches, with a left boundary."""
    s = _Scope(preparation, path, (0, 0, 11.5, 9))
    c = s.ref("occurrence", (*path, "capacitor"))
    l = s.ref("occurrence", (*path, "inductor"))
    c1, c2 = s.pin((*path, "capacitor"), "terminal_1"), s.pin((*path, "capacitor"), "terminal_2")
    l1, l2 = s.pin((*path, "inductor"), "terminal_1"), s.pin((*path, "inductor"), "terminal_2")
    # These named source declarations connect terminal_1 to their signal bus
    # and terminal_2 to ground. Captured refs retain that physical authority.
    cs, cg, ls, lg = c1, c2, l1, l2
    s.native(c, (3, 4.8), 270, contact=cs)
    s.native(l, (8, 4.8), 270, contact=ls)
    if boundary_names:
        s.junction("lc_signal", (3, 6), "left", "right", "bottom")
        s.wire("capacitor_signal", cs, ("lc_signal", "bottom"))
        s.wire("inductor_signal", ls, ("lc_signal", "right"), (8, 6))
        first = s.boundary(path, boundary_names[0]); s.expose(first, (0, 6))
        s.wire("public_signal", first, ("lc_signal", "left"))
        if len(boundary_names) == 2:
            # The second published alias is distinct visible boundary incidence.
            second = s.boundary(path, boundary_names[1]); s.expose(second, (11.5, 6))
            # Its upper route leaves the scope-title band unobstructed.
            s.junctions[-1] = DiagramJunction("lc_signal", (3, 6),
                                             tuple(DiagramSide(a) for a in ("left", "right", "bottom", "top")))
            s.points["lc_signal", "top"] = (3, 6.55)
            s.wire("alternate_public_signal", second, ("lc_signal", "top"),
                   (10, 6), (10, 8), (5, 8), (5, 6.8), (3, 6.8))
    else:
        s.wire("parallel_signal", cs, ls, (3, 6), (8, 6))
    for name, pin, at in (("capacitor", cg, (3, 1.7)),
                          ("inductor", lg, (8, 1.7))):
        ground = s.ref("ground", (*path, name), "terminal_2")
        s.native(ground, at, 270, contact=ground)
        s.wire(name + "_ground", pin, ground)
    if bus_label is not None:
        run = next(row["run"] for row in s.inventory.analysis_labels if row["text"] == bus_label)
        # The leader ends at the text's left baseline, so approach from below.
        label_x = 3 - .4 - (run.bounds.xmax - run.origin.x)
        s.labels(**{bus_label: (cs, (label_x, 4.8), (label_x, 5.35))})
    return s.finish()


def _coupled_lc(preparation, *, coupler_name, alternate=False, bus_label="signal"):
    s = _Scope(preparation, (), (0, 0, 23, 13.5))
    child = s.ref("scope", ("resonator",))
    child_layout = _lc(preparation, ("resonator",),
                       boundary_names=("terminal", "alternate_terminal") if alternate else ("terminal",),
                       bus_label="terminal_node" if alternate else bus_label)
    s.child(child, child_layout, (10, 1))
    coupling = s.ref("occurrence", (coupler_name,))
    a, b = s.pin((coupler_name,), "terminal_1"), s.pin((coupler_name,), "terminal_2")
    s.native(coupling, (6, 7))
    port = s.ref("port", (), "signal_in"); s.native(port, (3.5, 7), 180)
    terminal = s.boundary(("resonator",), "terminal")
    s.wire("measurement_to_coupler", port, a)
    s.wire("coupler_to_child", b, terminal)
    label_contacts = {"signal_boundary": a, "resonator_node": terminal}
    for row in s.inventory.analysis_labels:
        name, run = row["text"], row["run"]
        contact = label_contacts[name]
        x, y = s.points[contact]
        label_x = x + .4 - (run.bounds.xmin - run.origin.x)
        label_y = y - .6 - (run.bounds.ymax - run.origin.y)
        s.labels(**{name: (contact, (x, label_y), (label_x, label_y))})
    return SchematicLayout(preparation, s.finish())


def _four_arm(preparation):
    s = _Scope(preparation, (), (0, 0, 60, 60))
    s.junction("central_cross", (30, 30), "left", "right", "top", "bottom")
    # All four actual capacitor occurrences and all four native Port bodies have
    # authored cardinal poses; the native load shape is not independently bent.
    table = {
        "a": ((20, 30), 180, (12, 30), 180, "left"),
        "b": ((40, 30), 0, (48, 30), 0, "right"),
        "c": ((30, 40), 90, (30, 48), 90, "top"),
        "d": ((30, 20), 270, (30, 12), 270, "bottom"),
    }
    starts = {}
    for name, (origin, rotation, port_at, port_rotation, arm) in table.items():
        body = s.ref("occurrence", (f"capacitor_{name}",))
        a = s.pin((f"capacitor_{name}",), "terminal_1")
        b = s.pin((f"capacitor_{name}",), "terminal_2")
        s.native(body, origin, rotation)
        port = s.ref("port", (), f"port_{name}")
        s.native(port, port_at, port_rotation)
        s.wire(f"central_{name}", ("central_cross", arm), a)
        s.wire(f"outer_{name}", b, port)
        starts[name] = a
        x,y = s.points[b]
        s.labels(**{name: (b, (x, y + 2) if arm in ("left", "right") else (x + 2, y),
                             (x + 2, y + 2))})
    s.labels(central=(starts["a"], (20, 28), (22, 28)))
    return SchematicLayout(preparation, s.finish())



def _feedline(preparation, path, *, tap_name="tap", terminated=False):
    # This named recipe places two measured native CPW bodies on one short row.
    s = _Scope(preparation, path, (0, -2, 19, 6.8) if terminated else (0, 0, 18, 5))
    left, right = (s.ref("occurrence", (*path, name)) for name in ("left", "right"))
    lh, lt = (s.pin((*path, "left"), name) for name in ("head.signal", "tail.signal"))
    rh, rt = (s.pin((*path, "right"), name) for name in ("head.signal", "tail.signal"))
    s.native(left, (2, 2.2), contact=lh)
    s.native(right, (10.5, 2.2), contact=rh)
    arms = ("left", "right") if terminated else ("left", "right", "bottom")
    s.junction("tap_t", (9, 2.2), *arms)
    s.wire("left_to_tap", lt, ("tap_t", "left"))
    s.wire("tap_to_right", ("tap_t", "right"), rh)
    if terminated:
        input_port = s.ref("port", path, "input")
        output_port = s.ref("port", path, "output")
        s.native(input_port, (.8, 2.2), 180); s.native(output_port, (17.2, 2.2))
        s.wire("input_to_left", input_port, lh)
        s.wire("right_to_output", rt, output_port)
        run = next(row["run"] for row in s.inventory.analysis_labels if row["text"] == "output")
        label_x = 17.2 + .4 - (run.bounds.xmin - run.origin.x)
        label_y = 2.2 + .55 - (run.bounds.ymin - run.origin.y)
        s.labels(input=(input_port, (.8, .6), (2, .6)),
                 middle=(lt, (s.points[lt][0], .6), (s.points[lt][0]+1, .6)),
                 output=(output_port, (17.2, label_y), (label_x, label_y)))
    else:
        input_pin = s.boundary(path, "input")
        output_pin = s.boundary(path, "output")
        tap_pin = s.boundary(path, tap_name)
        s.expose(input_pin, (0, 2.2)); s.expose(output_pin, (18, 2.2)); s.expose(tap_pin, (9, 0))
        s.wire("input_to_left", input_pin, lh)
        s.wire("right_to_output", rt, output_pin)
        s.wire("tap_to_boundary", ("tap_t", "bottom"), tap_pin)
    header = next(row["ref"] for row in s.inventory.captions
                  if json.loads(row["ref"].key)["source_kind"] == "scope_title")
    return s.finish(caption_positions={header: (.8, 6 if terminated else 4.2)})


def _feedline_readout(preparation):
    from dataclasses import replace

    s = _Scope(preparation, (), (0, 1.1, 28, 19.5))
    feedline = s.ref("scope", ("feedline",))
    readout = s.ref("scope", ("readout",))
    s.child(feedline, _feedline(preparation, ("feedline",)), (4, 11))
    readout_layout = _lc(preparation, ("readout",), boundary_names=("readout_node",))
    title = next(caption.target for caption in readout_layout.captions
                 if json.loads(caption.target.key)["source_kind"] == "scope_title")
    # Only this instance's envelope/title change; its measured electrical ink stays intact.
    readout_layout = replace(
        readout_layout, frame=(0, .6, 9.4, 7.3),
        captions=tuple(DiagramCaption(caption.target, (.8, 6.65))
                       if caption.target == title else caption
                       for caption in readout_layout.captions),
    )
    s.child(readout, readout_layout, (13.5, 1.3))
    coupling = s.ref("occurrence", ("feedline_readout_coupler",))
    a, b = (s.pin(("feedline_readout_coupler",), name) for name in ("terminal_1", "terminal_2"))
    s.native(coupling, (13, 10.1), 270)
    tap = s.boundary(("feedline",), "tap")
    node = s.boundary(("readout",), "readout_node")
    s.wire("feedline_to_coupler", tap, a)
    s.wire("coupler_to_readout", b, node, (13, 7.3))
    left = s.ref("port", (), "feedline_in"); s.native(left, (1.5, 13.2), 180)
    right = s.ref("port", (), "feedline_out"); s.native(right, (24.5, 13.2))
    s.wire("input_boundary", left, s.boundary(("feedline",), "input"))
    s.wire("output_boundary", s.boundary(("feedline",), "output"), right)
    run = next(row["run"] for row in s.inventory.analysis_labels if row["text"] == "feedline_out")
    label_x = 24.5 + .4 - (run.bounds.xmin - run.origin.x)
    label_y = 13.2 + .55 - (run.bounds.ymin - run.origin.y)
    s.labels(feedline_in=(left, (1.5, 11.5), (3, 11.5)),
             feedline_out=(right, (24.5, label_y), (label_x, label_y)))
    return SchematicLayout(preparation, s.finish())


def _tapped(preparation):
    s = _Scope(preparation, (), (0, 0, 26, 11))
    child = s.ref("scope", ("feedline",))
    s.child(child, _feedline(preparation, ("feedline",)), (4, 2))
    return SchematicLayout(preparation, s.finish())


def _mtl(preparation):
    s = _Scope(preparation, (), (0, 0, 34, 19))
    line = s.ref("occurrence", ("coupled",)); s.native(line, (13, 9))
    # Four explicitly selected signal contacts, not a split into scalar lines.
    # This course recipe places upper/lower Ports outside their respective
    # conductor rows. The explicit elbows leave the native Port shunt before
    # returning to the measured line contact; no geometry search is performed.
    for name, pin, rotation, dx, port_dy, label_dy, elbow_dx in (
        ("readout_head", "head.readout", 180, -6, 2.8, 2, 2),
        ("filter_head", "head.filter", 180, -6, -2.8, -2, 2),
        ("readout_tail", "tail.readout", 0, 6, 2.8, 2, -2),
        ("filter_tail", "tail.filter", 0, 6, -2.8, -2, -2),
    ):
        contact = s.pin(("coupled",), pin)
        x, y = s.points[contact]
        port = s.ref("port", (), name)
        s.native(port, (x+dx, y+port_dy), rotation)
        s.wire(name+"_attachment", port, contact,
               (x+dx+elbow_dx, y+port_dy), (x+dx+elbow_dx, y))
        s.labels(**{name: (port, (x+dx, y+port_dy+label_dy),
                          (x+dx+3, y+port_dy+label_dy))})
    return SchematicLayout(preparation, s.finish())


def _floating(preparation):
    s = _Scope(preparation, ("floating",), (0, 0, 24, 21))
    pins = {}
    for name, origin, rotation in (
        ("mutual_cap", (9, 12), 0), ("mutual_ind", (9, 7.5), 0),
        ("plus_shunt", (5, 5), 270), ("minus_shunt", (17, 5), 270),
    ):
        target = s.ref("occurrence", ("floating", name))
        s.native(target, origin, rotation)
        pins[name] = tuple(s.pin(("floating", name), id) for id in ("terminal_1", "terminal_2"))
    s.junction("plus_upper", (5, 12), "left", "right", "top", "bottom")
    s.junction("plus_lower", (5, 7.5), "right", "top", "bottom")
    s.junction("minus_upper", (17, 12), "left", "right", "top", "bottom")
    s.junction("minus_lower", (17, 7.5), "left", "top", "bottom")
    s.wire("plus_capacitor", ("plus_upper", "right"), pins["mutual_cap"][0])
    s.wire("plus_inductor", ("plus_lower", "right"), pins["mutual_ind"][0])
    s.wire("plus_junctions", ("plus_upper", "bottom"), ("plus_lower", "top"))
    s.wire("plus_shunt_signal", ("plus_lower", "bottom"), pins["plus_shunt"][0])
    s.wire("minus_capacitor", pins["mutual_cap"][1], ("minus_upper", "left"))
    s.wire("minus_inductor", pins["mutual_ind"][1], ("minus_lower", "left"))
    s.wire("minus_junctions", ("minus_upper", "bottom"), ("minus_lower", "top"))
    s.wire("minus_shunt_signal", ("minus_lower", "bottom"), pins["minus_shunt"][0])
    boundary = {name: s.boundary(("floating",), name)
                for name in ("plus_1", "plus_2", "minus_1", "minus_2")}
    for name, at in {"plus_1": (0,12), "plus_2": (24,16),
                     "minus_1": (0,18), "minus_2": (24,12)}.items():
        s.expose(boundary[name], at)
    s.wire("plus_1_boundary", boundary["plus_1"], ("plus_upper", "left"))
    s.wire("plus_2_boundary", boundary["plus_2"], ("plus_upper", "top"), (5,16))
    s.wire("minus_1_boundary", boundary["minus_1"], ("minus_upper", "top"), (17,18))
    s.wire("minus_2_boundary", ("minus_upper", "right"), boundary["minus_2"])
    s.jumps.append(DiagramJump("minus_1_boundary", (17,16), "plus_2_boundary"))
    # Each actual source-grounded shunt has its own declared return glyph.
    ground_poses = {("floating", "plus_shunt"): (5,1.8),
                    ("floating", "minus_shunt"): (17,1.8)}
    for ground in s.inventory.grounds:
        path = tuple(json.loads(ground.key)["path"])
        s.native(ground, ground_poses[path], 270, contact=ground)
        s.wire("ground_"+path[-1], pins[path[-1]][1], ground)
    return s.finish()


def _capstone(preparation):
    s = _Scope(preparation, (), (0,0,70,35))
    feedline = s.ref("scope", ("feedline",))
    readout = s.ref("scope", ("readout",))
    floating = s.ref("scope", ("floating",))
    s.child(feedline, _feedline(preparation, ("feedline",)), (4,25))
    s.child(readout, _lc(preparation, ("readout",), boundary_names=("readout_node",)), (11,0))
    s.child(floating, _floating(preparation), (36,0))
    pins = {}
    for name, origin, rotation in (
        ("feedline_readout_coupler", (13,23), 270),
        ("readout_to_floating_plus", (28,17), 0),
        ("readout_to_floating_minus", (28,13), 0),
    ):
        target = s.ref("occurrence", (name,)); s.native(target, origin, rotation)
        pins[name] = tuple(s.pin((name,), id) for id in ("terminal_1", "terminal_2"))
    s.junction("readout_attachment", (8,17), "left", "right", "bottom")
    s.junction("floating_split", (24,17), "left", "right", "bottom")
    s.junction("floating_minus_turn", (24,13), "top", "right")
    s.wire("feedline_coupler", s.boundary(("feedline",), "tap"), pins["feedline_readout_coupler"][0])
    s.wire("readout_attachment_left", pins["feedline_readout_coupler"][1],
           ("readout_attachment", "left"), (13,20), (6,20), (6,17))
    node = s.boundary(("readout",), "readout_node")
    s.wire("readout_attachment_bottom", ("readout_attachment", "bottom"), node, (8,6))
    s.wire("readout_attachment_right", ("readout_attachment", "right"), ("floating_split", "left"))
    s.wire("plus_coupler_start", ("floating_split", "right"), pins["readout_to_floating_plus"][0])
    s.wire("minus_turn", ("floating_split", "bottom"), ("floating_minus_turn", "top"))
    s.wire("minus_coupler_start", ("floating_minus_turn", "right"), pins["readout_to_floating_minus"][0])
    s.wire("plus_coupler_child", pins["readout_to_floating_plus"][1],
           s.boundary(("floating",), "plus_1"), (33,17), (33,12))
    s.wire("minus_coupler_child", pins["readout_to_floating_minus"][1],
           s.boundary(("floating",), "minus_1"), (31.5,13), (31.5,10), (34,10), (34,18))
    s.jumps.append(DiagramJump("minus_coupler_child", (34,12), "plus_coupler_child"))
    port_specs = (("feedline_in", (1,27.2), 180, ("feedline",), "input"),
                  ("feedline_out", (25,27.2), 0, ("feedline",), "output"),
                  ("floating_probe_plus", (64,16), 0, ("floating",), "plus_2"),
                  ("floating_probe_minus", (64,12), 0, ("floating",), "minus_2"))
    ports = {}
    for name, at, rotation, child_path, boundary in port_specs:
        port = s.ref("port", (), name); s.native(port, at, rotation); ports[name] = port
        s.wire(name+"_attachment", s.boundary(child_path, boundary), port)
    run = next(row["run"] for row in s.inventory.analysis_labels if row["text"] == "feedline_out")
    label_x = 25 + .4 - (run.bounds.xmin - run.origin.x)
    label_y = 27.2 + .55 - (run.bounds.ymin - run.origin.y)
    s.labels(feedline_in=(ports["feedline_in"], (1,25.2), (3,25.2)),
             feedline_out=(ports["feedline_out"], (25,label_y), (label_x,label_y)),
             readout_node=(node, (9,6), (9,4.5)),
             floating_plus=(ports["floating_probe_plus"], (64,18), (66,18)),
             floating_minus=(ports["floating_probe_minus"], (64,10), (66,10)))
    return SchematicLayout(preparation, s.finish())


def engineer_layout(preparation, example):
    """Build only a named course declaration from this preparation's refs."""
    recipes = {
        "primitive_resonator": lambda: _coupled_lc(preparation, coupler_name="coupling_cap", bus_label=None),
        "coupled_lc": lambda: _coupled_lc(preparation, coupler_name="coupling_capacitor"),
        "reusable_lc": lambda: _coupled_lc(preparation, coupler_name="coupling_cap", alternate=True),
        "readout_illustration": lambda: SchematicLayout(preparation, _lc(preparation, (), bus_label="node")),
        "four_arm": lambda: _four_arm(preparation),
        "feedline_readout": lambda: _feedline_readout(preparation),
        "feedline_illustration": lambda: SchematicLayout(preparation, _feedline(preparation, (), terminated=True)),
        "tapped_feedline": lambda: _tapped(preparation),
        "ordered_mtl": lambda: _mtl(preparation),
        "four_port_capstone": lambda: _capstone(preparation),
    }
    return recipes[example]()
