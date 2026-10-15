"""Independent electrical and semantic witnesses for emitted diagram scenes.

Electrical connections are reconstructed from visible anchors and conductive
strokes.  Captured source bindings are checked separately against the exact
measured native geometry; they never supply nets or connections.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Literal, cast

from ...errors import SCNSimValidationError
from .scene import (
    COORDINATE_TOLERANCE,
    Bounds,
    ElectricalBox,
    GuideMark,
    NativeSymbol,
    NeutralScene,
    Path,
    Point,
    PortBlock,
    SubsystemRegion,
    TextRun,
    scene_digest,
)

_Representation = Literal["authoring"]
_Path = tuple[str, ...]
_NATIVE_KINDS = frozenset({"R", "L", "C", "JJ", "G"})

def _audit_fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_audit", evidence=evidence)

def _layout_fail(message: str, **evidence: object) -> SCNSimValidationError:
    return SCNSimValidationError(message, stage="schematic_layout", evidence=evidence)

def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value

def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value

def _same(left: float, right: float) -> bool:
    return abs(left - right) <= COORDINATE_TOLERANCE

def _same_point(left: Point, right: Point) -> bool:
    return _same(left.x, right.x) and _same(left.y, right.y)

def _finite_point(point: Point) -> bool:
    return math.isfinite(point.x) and math.isfinite(point.y)

def _area(bounds: Bounds) -> float:
    return (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin)

def _point_on_segment(point: Point, left: Point, right: Point) -> bool:
    length = math.hypot(right.x - left.x, right.y - left.y)
    if length <= COORDINATE_TOLERANCE:
        return _same_point(point, left)
    cross = (point.x - left.x) * (right.y - left.y) - (point.y - left.y) * (right.x - left.x)
    if abs(cross) / length > COORDINATE_TOLERANCE:
        return False
    return (
        min(left.x, right.x) - COORDINATE_TOLERANCE <= point.x <= max(left.x, right.x) + COORDINATE_TOLERANCE
        and min(left.y, right.y) - COORDINATE_TOLERANCE <= point.y <= max(left.y, right.y) + COORDINATE_TOLERANCE
    )

def _segment_intersection(
    first: tuple[Point, Point], second: tuple[Point, Point]
) -> tuple[Literal["none", "point", "overlap"], Point | None]:
    a, b = first
    c, d = second
    first_length = math.hypot(b.x - a.x, b.y - a.y)
    second_length = math.hypot(d.x - c.x, d.y - c.y)
    if first_length <= COORDINATE_TOLERANCE:
        return ("point", a) if _point_on_segment(a, c, d) else ("none", None)
    if second_length <= COORDINATE_TOLERANCE:
        return ("point", c) if _point_on_segment(c, a, b) else ("none", None)
    denominator = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x)
    numerator_first = (c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)
    numerator_second = (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
    if abs(denominator) / (first_length * second_length) <= COORDINATE_TOLERANCE:
        if (
            abs(numerator_first) / second_length > COORDINATE_TOLERANCE
            or abs(numerator_second) / first_length > COORDINATE_TOLERANCE
        ):
            return "none", None
        if any(_point_on_segment(point, *first) for point in second) or any(
            _point_on_segment(point, *second) for point in first
        ):
            return "overlap", None
        return "none", None
    first_fraction = numerator_first / denominator
    second_fraction = numerator_second / denominator
    first_tolerance = COORDINATE_TOLERANCE / first_length
    second_tolerance = COORDINATE_TOLERANCE / second_length
    if (
        -first_tolerance <= first_fraction <= 1.0 + first_tolerance
        and -second_tolerance <= second_fraction <= 1.0 + second_tolerance
    ):
        return "point", Point(
            a.x + first_fraction * (b.x - a.x),
            a.y + first_fraction * (b.y - a.y),
        )
    return "none", None

class _PointGraph:
    """Geometric union-find whose keys are scene points, never hidden net IDs."""

    def __init__(self) -> None:
        self.points: list[Point] = []
        self.parent: list[int] = []
        self.ground_roots: set[int] = set()

    def index(self, point: Point) -> int:
        if not _finite_point(point):
            raise _layout_fail("diagram geometry contains a non-finite point")
        for index, existing in enumerate(self.points):
            if _same_point(point, existing):
                return index
        index = len(self.points)
        self.points.append(point)
        self.parent.append(index)
        return index

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, *indexes: int) -> None:
        if not indexes:
            return
        roots = [self.find(index) for index in indexes]
        root = min(roots)
        for item in roots:
            self.parent[item] = root

    def mark_ground(self, point: Point) -> None:
        self.ground_roots.add(self.find(self.index(point)))

    def normalize_ground(self) -> None:
        roots = {self.find(root) for root in self.ground_roots}
        if roots:
            self.union(*roots)
            self.ground_roots = {self.find(next(iter(roots)))}

    def root(self, point: Point) -> int:
        return self.find(self.index(point))

    def is_ground(self, root: int) -> bool:
        return self.find(root) in {self.find(item) for item in self.ground_roots}

@dataclass(frozen=True, slots=True)
class _Region:
    source: SubsystemRegion
    path: _Path
    parent: _Path | None

@dataclass(frozen=True, slots=True)
class DiagramAuditData:
    """Immutable internal certificate data consumed by the public audit facade."""

    representation: _Representation
    plan_id: str
    plan_sha256: str
    connectivity_sha256: str
    semantic_sha256: str
    compiled_graph_sha256: str | None
    expanded_graph_sha256: str | None
    presentation_sha256: str | None
    observed_electrical: Mapping[str, object]
    observed_semantic: Mapping[str, object]
    observed_rows: tuple[Mapping[str, object], ...]
    verified_rows: tuple[Mapping[str, object], ...]

def _visible_bounds(bounds: Iterable[Bounds]) -> Bounds:
    materialized = tuple(bounds)
    if not materialized:
        raise _layout_fail("visible object has no geometric extent")
    return Bounds(
        min(item.xmin for item in materialized),
        min(item.ymin for item in materialized),
        max(item.xmax for item in materialized),
        max(item.ymax for item in materialized),
    )

def _verify_rectangular_frame(path: Path, bounds: Bounds, *, role: str) -> None:
    perimeter = (
        Point(bounds.xmin, bounds.ymin),
        Point(bounds.xmax, bounds.ymin),
        Point(bounds.xmax, bounds.ymax),
        Point(bounds.xmin, bounds.ymax),
    )
    if (
        path.role != role
        or not path.closed
        or path.codes is not None
        or len(path.points) != 5
    ):
        raise _layout_fail("visible rectangular frame disagrees with its occupied bounds", role=role)
    # Rigid cardinal adoption preserves perimeter direction but can rotate the
    # starting corner.  Accept only a cyclic shift of the same closed walk.
    for start in range(len(perimeter)):
        wanted = (*perimeter[start:], *perimeter[:start], perimeter[start])
        if all(
            _same_point(actual, expected)
            for actual, expected in zip(path.points, wanted, strict=True)
        ):
            return
    raise _layout_fail("visible rectangular frame disagrees with its occupied bounds", role=role)

def _deepest_region(bounds: Bounds, regions: Sequence[_Region]) -> _Region:
    containing = [region for region in regions if region.source.bounds.contains(bounds)]
    if not containing:
        raise _layout_fail("a complete occupied object lies outside the root envelope")
    return min(containing, key=lambda region: _area(region.source.bounds))

def _box_model(scene: NeutralScene, regions: Sequence[_Region]) -> None:
    paths: set[_Path] = set()
    for box in scene.boxes:
        if box.kind not in {"CPW", "MTL"} or box.orientation not in (0, 90, 180, 270):
            raise _audit_fail("a transmission-line box is not visibly native CPW/MTL geometry")
        _verify_rectangular_frame(box.outline, box.outline.bounds, role="box")
        visible = _visible_bounds(
            (
                box.outline.bounds,
                *(path.bounds for path in box.paths),
                box.title.bounds,
                box.kind_label.bounds,
                *((box.length_label.bounds,) if box.length_label is not None else ()),
                *((box.section_label.bounds,) if box.section_label is not None else ()),
                *(row.bounds for row in box.conductor_rows),
                *(label.bounds for label in box.anchor_labels),
                *((box.reference_label.bounds,) if box.reference_label is not None else ()),
            )
        )
        if not _same_bounds(visible, box.bounds):
            raise _layout_fail("transmission-line occupied bounds do not match its actual visible ink")
        endpoints = tuple(
            point
            for path in box.paths
            for point in (path.points[0], path.points[-1])
        )
        if any(
            sum(_same_point(anchor, endpoint) for endpoint in endpoints) != 1
            for _, anchor in box.anchors
        ):
            raise _audit_fail("transmission-line anchor is not one actual visible lead endpoint")
        physical_bounds = _visible_bounds((box.outline.bounds, *(path.bounds for path in box.paths)))
        owner = _deepest_region(physical_bounds, regions)
        path = (*owner.path, box.title.text)
        if path in paths:
            raise _audit_fail("visible transmission-line IDs are not unique in their owning region")
        paths.add(path)

def _branch_role(symbol: NativeSymbol) -> str:
    if symbol.kind not in _NATIVE_KINDS:
        raise _audit_fail("scene contains a symbol outside the closed native whitelist")
    branch_label = None if symbol.branch_label is None else symbol.branch_label.text
    if symbol.kind == "R":
        if branch_label not in (None, "resistance"):
            raise _audit_fail("resistor carries an unsupported visible branch badge")
        return "resistance"
    if symbol.kind == "C":
        if branch_label is None or branch_label == "capacitance":
            return "capacitance"
        if branch_label == "Cj":
            return "junction_capacitance"
        raise _audit_fail("capacitor carries an unsupported visible branch badge")
    if symbol.kind == "L":
        if branch_label in (None, "self", "inductor:self"):
            return "inductor:self"
        if branch_label.startswith("inductor:") and len(branch_label) > len("inductor:"):
            return branch_label
        raise _audit_fail("inductor carries an unsupported visible branch badge")
    if symbol.kind == "JJ":
        if branch_label not in (None, "josephson_inductance"):
            raise _audit_fail("Josephson symbol carries an unsupported visible branch badge")
        return "josephson_inductance"
    if branch_label is None:
        return "conductance"
    return branch_label

def _same_bounds(left: Bounds, right: Bounds) -> bool:
    return all(
        _same(first, second)
        for first, second in zip(
            (left.xmin, left.ymin, left.xmax, left.ymax),
            (right.xmin, right.ymin, right.xmax, right.ymax),
            strict=True,
        )
    )

def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.role == right.role
        and left.closed == right.closed
        and left.codes == right.codes
        and len(left.points) == len(right.points)
        and all(_same_point(first, second) for first, second in zip(left.points, right.points, strict=True))
    )

def _same_closed_path_cyclic_start(left: Path, right: Path) -> bool:
    """Compare a closed path while permitting only a cyclic start change."""
    if (
        left.role != right.role
        or not left.closed
        or not right.closed
        or left.codes != right.codes
        or len(left.points) != len(right.points)
        or len(left.points) < 2
        or not _same_point(left.points[0], left.points[-1])
        or not _same_point(right.points[0], right.points[-1])
    ):
        return False
    left_walk = left.points[:-1]
    right_walk = right.points[:-1]
    if len(left_walk) != len(right_walk):
        return False
    for start in range(len(right_walk)):
        candidate = (*right_walk[start:], *right_walk[:start])
        if all(
            _same_point(actual, expected)
            for actual, expected in zip(left_walk, candidate, strict=True)
        ):
            return True
    return False

def _verify_text_run(run: object) -> None:
    from .metrics import shape_text
    from .scene import Bounds, Point, TextRun

    if not isinstance(run, TextRun):
        raise _audit_fail("scene text fact is not one immutable TextRun")
    try:
        if run.orientation not in (0, 90, 180, 270):
            raise ValueError("TextRun orientation is not cardinal")
        inverse = (-run.orientation) % 360
        source_origin = run.origin.rotated(inverse)
        source = shape_text(run.text, at=source_origin, size=run.size, role=run.role)

        def orient(point: Point) -> Point:
            return point.rotated(run.orientation)

        def orient_bounds(bounds):
            return Bounds.around(
                orient(Point(x, y))
                for x in (bounds.xmin, bounds.xmax)
                for y in (bounds.ymin, bounds.ymax)
            )

        glyphs = tuple(
            replace(
                glyph,
                vertices=tuple(orient(point) for point in glyph.vertices),
                bounds=orient_bounds(glyph.bounds),
            )
            for glyph in source.glyphs
        )
        reference = replace(
            source,
            origin=orient(source.origin),
            glyphs=glyphs,
            bounds=orient_bounds(source.bounds),
            orientation=run.orientation,
        )
    except Exception as error:
        raise _audit_fail("visible text cannot be reconstructed by the trusted font grammar") from error
    if (
        not _same_bounds(run.bounds, reference.bounds)
        or not _same(run.ascent, reference.ascent)
        or not _same(run.descent, reference.descent)
        or len(run.glyphs) != len(reference.glyphs)
    ):
        raise _audit_fail("visible text bounds/metrics do not match its literal glyphs", text=run.text)
    for actual, wanted in zip(run.glyphs, reference.glyphs, strict=True):
        if (
            actual.character != wanted.character
            or actual.font_sha256 != wanted.font_sha256
            or actual.codes != wanted.codes
            or not _same_bounds(actual.bounds, wanted.bounds)
            or not _same(actual.advance, wanted.advance)
            or len(actual.vertices) != len(wanted.vertices)
            or any(
                not _same_point(first, second)
                for first, second in zip(actual.vertices, wanted.vertices, strict=True)
            )
        ):
            raise _audit_fail("visible glyph outlines do not encode their claimed text", text=run.text)

_GraphicOwner = tuple[str, int]

def _scene_text_groups(scene: NeutralScene) -> tuple[tuple[_GraphicOwner, TextRun], ...]:
    groups: list[tuple[_GraphicOwner, TextRun]] = []

    def extend(owner: _GraphicOwner, runs: Iterable[TextRun | None]) -> None:
        groups.extend((owner, run) for run in runs if run is not None)

    for index, run in enumerate(scene.text):
        extend(("scene_text", index), (run,))
    for index, symbol in enumerate(scene.symbols):
        extend(("symbol", index), (symbol.visible_name, symbol.value, symbol.branch_label))
    for index, port in enumerate(scene.ports):
        extend(
            ("port", index),
            (
                *port.labels,
                port.reference_load.visible_name,
                port.reference_load.value,
                port.reference_load.branch_label,
            ),
        )
    for index, box in enumerate(scene.boxes):
        extend(
            ("box", index),
            (
                box.title,
                box.kind_label,
                box.length_label,
                box.section_label,
                *box.conductor_rows,
                *box.anchor_labels,
                box.reference_label,
            ),
        )
    for index, region in enumerate(scene.regions):
        extend(("region", index), (region.header,))
    for index, site in enumerate(scene.boundary_sites):
        extend(("boundary_site", index), (site.visible_label,))
    for index, mark in enumerate(scene.node_marks):
        extend(("node_mark", index), (mark.label,))
    for index, guide in enumerate(scene.guides):
        extend(("guide", index), (guide.label,))
    if scene.provenance_band is not None:
        extend(("provenance", 0), scene.provenance_band.lines)
    return tuple(groups)

def _verify_scene_text(scene: NeutralScene) -> None:
    for _, run in _scene_text_groups(scene):
        _verify_text_run(run)

def _verify_occupied_bounds_match_ink(scene: NeutralScene) -> None:
    """Check emitted occupied bounds against the literal visible ink."""

    for symbol in scene.symbols:
        visible = _visible_bounds(
            (
                symbol.symbol_bounds,
                symbol.visible_name.bounds,
                *((symbol.value.bounds,) if symbol.value is not None else ()),
                *((symbol.branch_label.bounds,) if symbol.branch_label is not None else ()),
                *((symbol.reference_polarity.bounds,) if symbol.reference_polarity is not None else ()),
            )
        )
        if not _same_bounds(visible, symbol.occupied_bounds):
            raise _layout_fail(
                "native symbol occupied bounds do not match its actual visible ink",
                identity=symbol.visible_name.text,
            )

    for port in scene.ports:
        load = port.reference_load
        load_visible = _visible_bounds(
            (
                load.symbol_bounds,
                load.visible_name.bounds,
                *((load.value.bounds,) if load.value is not None else ()),
                *((load.branch_label.bounds,) if load.branch_label is not None else ()),
                *((load.reference_polarity.bounds,) if load.reference_polarity is not None else ()),
            )
        )
        if not _same_bounds(load_visible, load.occupied_bounds):
            raise _layout_fail("Port load occupied bounds do not match its actual visible ink")
        if any(
            sum(_same_path(ground_path, path) for path in port.paths) != 1
            for ground_path in port.ground_glyph
        ):
            raise _audit_fail("Port ground glyph is not part of its actual emitted stroke set")
        port_visible = _visible_bounds(
            (
                port.circle.bounds,
                load.occupied_bounds,
                *(path.bounds for path in port.paths),
                *(label.bounds for label in port.labels),
            )
        )
        if not _same_bounds(port_visible, port.occupied_bounds):
            raise _layout_fail("Port occupied bounds do not match its actual visible ink")

    for box in scene.boxes:
        visible = _visible_bounds(
            (
                box.outline.bounds,
                *(path.bounds for path in box.paths),
                box.title.bounds,
                box.kind_label.bounds,
                *((box.length_label.bounds,) if box.length_label is not None else ()),
                *((box.section_label.bounds,) if box.section_label is not None else ()),
                *(row.bounds for row in box.conductor_rows),
                *(label.bounds for label in box.anchor_labels),
                *((box.reference_label.bounds,) if box.reference_label is not None else ()),
            )
        )
        if not _same_bounds(visible, box.bounds):
            raise _layout_fail("transmission-line occupied bounds do not match its actual visible ink")
def _verify_native_geometry(symbol: NativeSymbol, metrics) -> None:
    """Match actual ink against the trusted native-template grammar."""

    anchors = tuple(symbol.anchors)
    if len(anchors) != 2 or {name for name, _ in anchors} != {"terminal_1", "terminal_2"}:
        raise _audit_fail("native symbol does not expose the closed terminal_1/terminal_2 grammar")
    start, end = anchors[0][1], anchors[1][1]
    from .native import native_block

    try:
        references = tuple(
            native_block(
                symbol.kind,
                start=start,
                end=end,
                name="native-geometry-reference",
                value=None,
                metrics=metrics,
                label_side=side,
            )
            for start, end in ((start, end), (end, start))
            for side in ("top", "bottom", "left", "right")
        )
    except Exception as error:
        raise _layout_fail("native symbol anchors do not form one exact trusted 1U primitive") from error
    if not any(
        symbol.orientation == reference.orientation
        and len(symbol.paths) == len(reference.paths)
        and all(
            _same_path(actual, wanted)
            for actual, wanted in zip(symbol.paths, reference.paths, strict=True)
        )
        and _same_bounds(symbol.symbol_bounds, reference.symbol_bounds)
        and _same_path(symbol.reference_polarity, reference.reference_polarity)
        for reference in references
    ):
        raise _audit_fail(
            "native symbol kind/anchors do not match its trusted visible glyph geometry",
            visible_name=symbol.visible_name.text,
            native_kind=symbol.kind,
        )

def _conductive_paths(scene: NeutralScene, *, metrics) -> tuple[tuple[Point, ...], ...]:
    paths: list[tuple[Point, ...]] = [wire.points for wire in scene.conductive]
    # Native bodies are branches, never wires through their glyph.  Their
    # endpoint-adjacent lead strokes are nevertheless real conductor ink and
    # must support an intentionally singleton terminal as well as a visible T.
    # Add only the one segment incident to each validated terminal anchor.
    for symbol in scene.symbols:
        if symbol.kind == "JJ":
            # The verified glyph paths contain two endpoint-adjacent lead
            # strokes and the central cross.  Keep only the lead portions
            # outside the cross's actual projection on the terminal axis; a
            # fixed metrics stub can overrun a short native span and short
            # the two component terminals in the point graph.
            anchors = tuple(point for _, point in symbol.anchors)
            if len(anchors) != 2:
                raise _audit_fail("visible Josephson glyph lacks two actual terminals")
            start, end = anchors
            dx, dy = end.x - start.x, end.y - start.y
            span = math.hypot(dx, dy)
            if span == 0.0:
                raise _audit_fail("visible Josephson terminals coincide")
            ux, uy = dx / span, dy / span
            terminal_path_indices: list[int] = []
            for anchor in (start, end):
                matches = [
                    index
                    for index, path in enumerate(symbol.paths)
                    if path.points
                    and (
                        _same_point(anchor, path.points[0])
                        or _same_point(anchor, path.points[-1])
                    )
                ]
                if len(matches) != 1:
                    raise _audit_fail(
                        "visible Josephson terminal lacks one exact native lead path"
                    )
                terminal_path_indices.append(matches[0])
            if terminal_path_indices[0] == terminal_path_indices[1]:
                raise _audit_fail("visible Josephson terminals share one glyph path")
            cross_paths = tuple(
                path
                for index, path in enumerate(symbol.paths)
                if index not in terminal_path_indices
            )
            if not cross_paths:
                raise _audit_fail("visible Josephson glyph omits its central cross paths")
            cross_projection = tuple(
                (point.x - start.x) * ux + (point.y - start.y) * uy
                for path in cross_paths
                for point in (path.points[0], path.points[-1])
            )
            cross_min, cross_max = min(cross_projection), max(cross_projection)
            boundaries = (
                Point(start.x + ux * cross_min, start.y + uy * cross_min),
                Point(start.x + ux * cross_max, start.y + uy * cross_max),
            )
            for anchor, boundary, path_index in zip(
                (start, end), boundaries, terminal_path_indices, strict=True
            ):
                native_lead = symbol.paths[path_index]
                if native_lead.codes is not None:
                    raise _audit_fail("visible Josephson lead uses unsupported coded geometry")
                lead_points = native_lead.points
                if _same_point(lead_points[-1], anchor):
                    lead_points = tuple(reversed(lead_points))
                elif not _same_point(lead_points[0], anchor):
                    raise _audit_fail("visible Josephson lead endpoint changed after validation")
                clipped = [anchor]
                for left, right in zip(lead_points, lead_points[1:]):
                    if _point_on_segment(boundary, left, right):
                        if not _same_point(clipped[-1], boundary):
                            clipped.append(boundary)
                        break
                    clipped.append(right)
                if len(clipped) > 1 and _same_point(clipped[-1], boundary):
                    paths.append(tuple(clipped))
            continue
        for _, anchor in symbol.anchors:
            for path in symbol.paths:
                if path.codes is not None or len(path.points) < 2:
                    continue
                if _same_point(anchor, path.points[0]):
                    paths.append(path.points[:2])
                elif _same_point(anchor, path.points[-1]):
                    paths.append(path.points[-2:])
    for box in scene.boxes:
        paths.extend(path.points for path in box.paths if path.role == "wire")
    for port in scene.ports:
        paths.extend(path.points for path in port.paths if path.role == "wire")
    return tuple(paths)

def _is_ground_paths(paths: Sequence[Path], terminal: Point) -> bool:
    if (
        len(paths) != 4
        or any(path.role not in {"symbol", "glyph"} or path.closed or len(path.points) != 2 for path in paths)
    ):
        return False
    stems = [path for path in paths if any(_same_point(terminal, point) for point in path.points)]
    if len(stems) != 1:
        return False
    stem = stems[0]
    stem_end = next(point for point in stem.points if not _same_point(point, terminal))
    vx, vy = stem_end.x - terminal.x, stem_end.y - terminal.y
    length = math.hypot(vx, vy)
    if length <= COORDINATE_TOLERANCE:
        return False
    ux, uy = vx / length, vy / length
    bars = [path for path in paths if path is not stem]
    if any(
        abs((path.points[1].x - path.points[0].x) * ux + (path.points[1].y - path.points[0].y) * uy)
        > COORDINATE_TOLERANCE
        for path in bars
    ):
        return False
    centers = [
        Point(
            (path.points[0].x + path.points[1].x) / 2.0,
            (path.points[0].y + path.points[1].y) / 2.0,
        )
        for path in bars
    ]
    if any(
        abs((center.x - terminal.x) * uy - (center.y - terminal.y) * ux)
        > COORDINATE_TOLERANCE
        for center in centers
    ):
        return False
    ordered = sorted(
        zip(bars, centers),
        key=lambda item: (item[1].x - terminal.x) * ux + (item[1].y - terminal.y) * uy,
    )
    widths = [
        math.hypot(path.points[1].x - path.points[0].x, path.points[1].y - path.points[0].y)
        for path, _ in ordered
    ]
    distances = [
        (center.x - terminal.x) * ux + (center.y - terminal.y) * uy
        for _, center in ordered
    ]
    return (
        _same_point(ordered[0][1], stem_end)
        and widths[0] > widths[1] > widths[2] > COORDINATE_TOLERANCE
        and 0.0 < distances[0] < distances[1] < distances[2]
    )

def _build_point_graph(
    scene: NeutralScene,
    *,
    metrics,
    normalize_ground: bool = True,
) -> tuple[_PointGraph, tuple[tuple[Point, Point], ...]]:
    graph = _PointGraph()
    conductive = _conductive_paths(scene, metrics=metrics)
    segments: list[tuple[Point, Point]] = []
    for points in conductive:
        if len(points) < 2:
            raise _layout_fail("a conductive path has fewer than two points")
        indexes = [graph.index(point) for point in points]
        graph.union(*indexes)
        segments.extend(zip(points, points[1:]))
    for jump in scene.jumps:
        points = jump.path.points
        if (
            jump.path.role != "jump"
            or jump.path.closed
            or jump.path.codes != (1, 4, 4, 4)
            or len(points) != 4
        ):
            raise _layout_fail("wire-jump geometry is not one visible open cubic path")
        start, end = points[0], points[-1]
        if _same_point(start, end) or not (_same(start.x, end.x) or _same(start.y, end.y)):
            raise _layout_fail("wire-jump endpoints do not span one cardinal conductor gap")
        if not all(any(_same_point(endpoint, tip) for segment in segments for tip in segment) for endpoint in (start, end)):
            raise _layout_fail("wire-jump endpoints do not terminate actual split conductive strokes")
        midpoint = Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
        horizontal = _same(start.y, end.y)
        if any(
            _point_on_segment(midpoint, *segment)
            and ((_same(segment[0].y, segment[1].y)) if horizontal else (_same(segment[0].x, segment[1].x)))
            for segment in segments
        ):
            raise _layout_fail("wire-jump straight interval still contains conductive ink")
        graph.union(graph.index(start), graph.index(end))
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            relation, point = _segment_intersection(first, second)
            if relation == "none":
                continue
            if relation == "overlap":
                shared = [
                    endpoint
                    for endpoint in (*first, *second)
                    if _point_on_segment(endpoint, *first) and _point_on_segment(endpoint, *second)
                ]
                graph.union(*(graph.index(endpoint) for endpoint in shared))
                continue
            assert point is not None
            first_interior = not any(_same_point(point, endpoint) for endpoint in first)
            second_interior = not any(_same_point(point, endpoint) for endpoint in second)
            if first_interior and second_interior:
                raise _layout_fail(
                    "different conductive strokes form an ambiguous implicit crossing",
                    first_segment=[{"x": item.x, "y": item.y} for item in first],
                    second_segment=[{"x": item.x, "y": item.y} for item in second],
                    crossing={"x": point.x, "y": point.y},
                )
            graph.union(graph.index(first[0]), graph.index(second[0]), graph.index(point))
    visible_contact_points = [
        point
        for branch in scene.symbols
        for _, point in branch.anchors
    ]
    visible_contact_points.extend(site.point for site in scene.boundary_sites)
    visible_contact_points.extend(mark.point for mark in scene.node_marks)
    visible_contact_points.extend(
        point
        for box in scene.boxes
        for _, point in box.anchors
    )
    visible_contact_points.extend(
        point
        for port in scene.ports
        for point in (
            port.boundary_anchor,
            port.circuit_anchor,
            port.external_anchor,
            port.load_anchor,
            port.ground_anchor,
        )
    )
    visible_contact_points.extend(
        point
        for port in scene.ports
        for _, point in port.reference_load.anchors
    )
    for port in scene.ports:
        if not _is_ground_paths(port.ground_glyph, port.ground_anchor):
            raise _audit_fail("Port local return is not a recognized visible ground glyph")
    for point in visible_contact_points:
        contacts = [segment for segment in segments if _point_on_segment(point, *segment)]
        ground_contact = any(
            guide.label is None
            and guide.paths
            and all(path.role in {"symbol", "glyph"} for path in guide.paths)
            and any(_same_point(point, terminal) for terminal in guide.terminals)
            for guide in scene.guides
        ) or any(_same_point(point, port.ground_anchor) for port in scene.ports)
        if not contacts and not ground_contact:
            raise _layout_fail("a registered electrical anchor has no actual conductive contact")
        graph.union(graph.index(point), *(graph.index(segment[0]) for segment in contacts))
    ground_guides = [
        guide
        for guide in scene.guides
        if guide.label is None
        and len(guide.terminals) == 1
        and _is_ground_paths(guide.paths, guide.terminals[0])
    ]
    duplicated_port_ground = [
        guide
        for guide in ground_guides
        if any(
            _same_point(guide.terminals[0], port.ground_anchor)
            for port in scene.ports
        )
    ]
    if duplicated_port_ground:
        raise _audit_fail(
            "Port local return duplicates its intrinsic visible ground glyph"
        )
    suspicious_ground_ink = [
        guide
        for guide in scene.guides
        if guide.label is None
        and guide.paths
        and all(path.role in {"symbol", "glyph"} for path in guide.paths)
        and guide not in ground_guides
    ]
    if suspicious_ground_ink:
        raise _audit_fail("an unlabeled native-symbol guide is not a recognized ground glyph")
    for guide in ground_guides:
        terminal = guide.terminals[0]
        anchor_contact = any(
            _same_point(terminal, point)
            for symbol in scene.symbols
            for _, point in symbol.anchors
        ) or any(
            _same_point(terminal, point)
            for port in scene.ports
            for _, point in port.reference_load.anchors
        )
        contacts = [segment for segment in segments if _point_on_segment(terminal, *segment)]
        if not contacts and not anchor_contact:
            raise _layout_fail("a ground glyph does not contact a conductive return")
        graph.union(
            graph.index(terminal),
            *(graph.index(segment[0]) for segment in contacts),
        )
        graph.mark_ground(terminal)
    for port in scene.ports:
        graph.mark_ground(port.ground_anchor)
    if normalize_ground:
        graph.normalize_ground()
    return graph, tuple(segments)

def _port_identity(port: PortBlock) -> tuple[str, str, str, bool]:
    if len(port.labels) not in (1, 2):
        raise _audit_fail("Port block must visibly carry its ID and optional PTC mark")
    port_id = port.labels[0].text
    removable = len(port.labels) == 2
    if removable and port.labels[1].text != "PTC-removable":
        raise _audit_fail("nonloading Port carries an unrecognized PTC-removable mark")
    role = "nonloading_probe" if removable else "terminated"
    if (
        port.reference_load.value is not None
        or not port.reference_load.visible_name.text
    ):
        raise _audit_fail("Port omits its sole visible raw Z0 native load", port_id=port_id)
    return port_id, role, port.reference_load.visible_name.text, removable

def _verify_port_block_geometry(port: PortBlock, metrics) -> None:
    """Match actual Port ink while allowing declared caption placement/rotation."""

    from .native import port_block

    port_id, role, impedance, _ = _port_identity(port)
    dx = port.boundary_anchor.x - port.circuit_anchor.x
    dy = port.boundary_anchor.y - port.circuit_anchor.y
    if _same(dy, 0.0) and not _same(dx, 0.0):
        side = "right" if dx > 0.0 else "left"
    elif _same(dx, 0.0) and not _same(dy, 0.0):
        side = "top" if dy > 0.0 else "bottom"
    else:
        raise _audit_fail("Port boundary-to-circuit incidence is not cardinal")
    load_dx = port.load_anchor.x - port.circuit_anchor.x
    load_dy = port.load_anchor.y - port.circuit_anchor.y
    if _same(load_dy, 0.0) and not _same(load_dx, 0.0):
        load_side = "right" if load_dx > 0.0 else "left"
    elif _same(load_dx, 0.0) and not _same(load_dy, 0.0):
        load_side = "top" if load_dy > 0.0 else "bottom"
    else:
        raise _audit_fail("Port circuit-to-load incidence is not cardinal")
    if (side in {"left", "right"}) == (load_side in {"left", "right"}):
        raise _audit_fail("Port boundary and load incidence are not perpendicular")
    reference = port_block(
        port_id=port_id,
        role=cast(Literal["terminated", "nonloading_probe"], role),
        reference_impedance=impedance,
        boundary_anchor=port.boundary_anchor,
        side=cast(Literal["left", "right", "top", "bottom"], side),
        load_side=cast(Literal["left", "right", "top", "bottom"], load_side),
        metrics=metrics,
    )
    load = port.reference_load
    wanted_load = reference.reference_load
    if (
        not _same_closed_path_cyclic_start(port.circle, reference.circle)
        or not _same_point(port.circle_center, reference.circle_center)
        or any(
            not _same_point(actual, wanted)
            for actual, wanted in zip(
                (
                    port.boundary_anchor,
                    port.circuit_anchor,
                    port.external_anchor,
                    port.load_anchor,
                    port.ground_anchor,
                ),
                (
                    reference.boundary_anchor,
                    reference.circuit_anchor,
                    reference.external_anchor,
                    reference.load_anchor,
                    reference.ground_anchor,
                ),
                strict=True,
            )
        )
        or len(port.paths) != len(reference.paths)
        or any(
            not _same_path(actual, wanted)
            for actual, wanted in zip(port.paths, reference.paths, strict=True)
        )
        or len(port.ground_glyph) != len(reference.ground_glyph)
        or any(
            not _same_path(actual, wanted)
            for actual, wanted in zip(
                port.ground_glyph, reference.ground_glyph, strict=True
            )
        )
        or len(port.labels) != len(reference.labels)
        or any(
            actual.text != wanted.text
            or actual.role != wanted.role
            or not _same(actual.size, wanted.size)
            for actual, wanted in zip(port.labels, reference.labels, strict=True)
        )
        or load.kind != "R"
        or load.branch_label is not None
        or len(load.anchors) != len(wanted_load.anchors)
        or any(
            actual_name != wanted_name or not _same_point(actual_point, wanted_point)
            for (actual_name, actual_point), (wanted_name, wanted_point) in zip(
                load.anchors, wanted_load.anchors, strict=True
            )
        )
        or not _same_bounds(load.symbol_bounds, wanted_load.symbol_bounds)
        or load.visible_name.text != wanted_load.visible_name.text
        or load.visible_name.role != wanted_load.visible_name.role
        or not _same(load.visible_name.size, wanted_load.visible_name.size)
    ):
        raise _audit_fail(
            "Port body/load ink does not match its visible native grammar",
            port_id=port_id,
        )
    _verify_native_geometry(load, metrics)

def _verify_guide_terminal_incidence(guide: GuideMark) -> None:
    """Prove every declared guide terminal belongs to one visible connector."""

    if not guide.paths or not guide.terminals:
        raise _audit_fail("visible semantic guide has no actual connector incidence")
    if len(guide.terminals) != len(set(guide.terminals)):
        raise _audit_fail("visible semantic guide duplicates a connector terminal")
    if any(
        not any(
            _same_point(terminal, path.points[0])
            or _same_point(terminal, path.points[-1])
            for path in guide.paths
        )
        for terminal in guide.terminals
    ):
        raise _audit_fail("visible semantic guide terminal is not an actual stroke endpoint")

    graph = _PointGraph()
    segments: list[tuple[Point, Point]] = []
    for path in guide.paths:
        graph.union(*(graph.index(point) for point in path.points))
        if path.codes is None:
            segments.extend(zip(path.points, path.points[1:]))
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            relation, point = _segment_intersection(first, second)
            if relation == "point":
                assert point is not None
                graph.union(graph.index(first[0]), graph.index(second[0]), graph.index(point))
            elif relation == "overlap":
                shared = [
                    endpoint
                    for endpoint in (*first, *second)
                    if _point_on_segment(endpoint, *first)
                    and _point_on_segment(endpoint, *second)
                ]
                graph.union(*(graph.index(endpoint) for endpoint in shared))
    root = graph.root(guide.terminals[0])
    if any(graph.root(terminal) != root for terminal in guide.terminals) or any(
        graph.root(path.points[0]) != root for path in guide.paths
    ):
        raise _audit_fail("visible semantic guide strokes do not form one connected incidence")

def _point_observed_rows(
    electrical: Mapping[str, object],
    structural: Mapping[str, object],
    detail: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = [*detail]
    collections = (
        ("electrical_net", electrical.get("nets", ())),
        ("physical_body", electrical.get("bodies", ())),
        ("port", electrical.get("ports", ())),
        ("mutual_coupling", electrical.get("couplings", ())),
        ("ownership_region", structural.get("regions", ())),
        ("leaf_ownership", structural.get("leaf_ownership", ())),
        ("port_ownership", structural.get("port_ownership", ())),
        ("boundary_incidence", structural.get("boundary_incidence", ())),
    )
    for kind, values in collections:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise _audit_fail("reconstructed manifest collection is malformed", kind=kind)
        rows.extend(
            {
                "category": "observed_scene",
                "kind": kind,
                "facts": _plain(value),
            }
            for value in values
            if isinstance(value, Mapping)
        )
    return tuple(cast(Mapping[str, object], _freeze(row)) for row in rows)

def _point_verified_rows(expected: object) -> tuple[Mapping[str, object], ...]:
    verified = getattr(expected, "verified", None)
    if not isinstance(verified, Mapping):
        raise _audit_fail("point expectation lacks verified provenance")
    source_rows = verified.get("source_rows")
    if not isinstance(source_rows, Sequence) or isinstance(source_rows, (str, bytes)):
        raise _audit_fail("point expectation lacks complete verified source rows")
    if any(
        not isinstance(row, Mapping)
        or row.get("category") != "verified_source"
        or not isinstance(row.get("kind"), str)
        for row in source_rows
    ):
        raise _audit_fail("point expectation has malformed verified source rows")
    rows: list[dict[str, object]] = [
        {
            "category": "verified_snapshot_provenance",
            "kind": "identity",
            "plan_id": verified.get("plan_id"),
            "plan_sha256": verified.get("plan_sha256"),
            "connectivity_sha256": verified.get("connectivity_sha256"),
            "semantic_sha256": verified.get("semantic_sha256"),
            "parameters_sha256": verified.get("parameters_sha256"),
            "compiled_graph_sha256": None,
            "expanded_graph_sha256": None,
        },
        {
            "category": "verified_snapshot_provenance",
            "kind": "canonical_point_values",
            "values": _plain(verified.get("canonical_values")),
        },
    ]
    rows.extend(dict(row) for row in source_rows)
    return tuple(cast(Mapping[str, object], _freeze(row)) for row in rows)

def certify_scene(
    point: object,
    scene: NeutralScene,
    *,
    show_values: bool,
    layout,
    metrics,
) -> DiagramAuditData:
    """Certify one frozen manually laid-out scene by visible reconstruction."""

    from ...authoring.snapshot import ResolvedPlanPoint
    from .expected import build_point_expected
    from .witness import witness_authoring

    if not isinstance(point, ResolvedPlanPoint):
        raise TypeError("certify_scene() requires an immutable ResolvedPlanPoint")
    if not isinstance(scene, NeutralScene):
        raise TypeError("certify_scene() requires a NeutralScene")
    if not isinstance(show_values, bool):
        raise TypeError("show_values must be boolean")
    _verify_scene_text(scene)
    _verify_occupied_bounds_match_ink(scene)
    expected = build_point_expected(point)
    electrical, structural, detail = witness_authoring(
        point,
        scene,
        show_values=show_values,
        layout=layout,
        metrics=metrics,
    )
    verified = expected.verified
    required = {
        key: verified.get(key)
        for key in (
            "plan_id",
            "plan_sha256",
            "connectivity_sha256",
            "semantic_sha256",
        )
    }
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise _audit_fail("verified point identity is incomplete")
    return DiagramAuditData(
        representation="authoring",
        plan_id=cast(str, required["plan_id"]),
        plan_sha256=cast(str, required["plan_sha256"]),
        connectivity_sha256=cast(str, required["connectivity_sha256"]),
        semantic_sha256=cast(str, required["semantic_sha256"]),
        compiled_graph_sha256=None,
        expanded_graph_sha256=None,
        presentation_sha256=scene_digest(scene),
        observed_electrical=electrical,
        observed_semantic=structural,
        observed_rows=_point_observed_rows(electrical, structural, detail),
        verified_rows=_point_verified_rows(expected),
    )
