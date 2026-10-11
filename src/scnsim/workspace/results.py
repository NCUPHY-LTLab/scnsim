"""Fixed, independently re-openable readers for sealed current results.

The reader keeps only immutable path/binding metadata and a detached selection.
Every operation opens one short SQLite snapshot and verifies the selected
completion, index root, locator, and canonical body before returning it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from ..errors import _freeze
from ..numeric_encoding import record_bytes, record_document
from .operation_store import OperationStore, _reference


@dataclass(frozen=True, slots=True)
class ResultSelection:
    workspace_instance_id: str
    plan_sha256: str
    request_sha256: str
    task_id: str
    attempt_id: str
    attempt_sha256: str
    terminal_attempt_ref: Mapping[str, Any]
    completion_ref: Mapping[str, Any]
    completion_sha256: str
    result_ref: Mapping[str, Any]
    index_root_ref: Mapping[str, Any]
    index_root_sha256: str
    generation_ancestry_ref: Mapping[str, Any] | None
    generation_count: int
    candidate_count: int
    baseline_locator: Mapping[str, Any]
    best_locator: Mapping[str, Any]

    def __post_init__(self):
        for name in ("terminal_attempt_ref", "completion_ref", "result_ref", "index_root_ref",
                     "generation_ancestry_ref", "baseline_locator", "best_locator"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))


@dataclass(frozen=True, slots=True)
class SensitivityResultSelection:
    """Fixed terminal identity for one complete or partial Sensitivity result."""

    workspace_instance_id: str
    plan_sha256: str
    request_sha256: str
    task_id: str
    attempt_id: str
    outcome: str
    attempt_sha256: str
    completion_ref: Mapping[str, Any]
    result_ref: Mapping[str, Any]
    manifest_ref: Mapping[str, Any]
    index_root_ref: Mapping[str, Any]
    block_count: int

    def __post_init__(self):
        for name in ("completion_ref", "result_ref", "manifest_ref", "index_root_ref"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PhysicalOptimizationResultSelection:
    """Detached identity for one exact v7 physical Optimization terminal."""

    workspace_instance_id: str
    plan_sha256: str
    request_sha256: str
    task_id: str
    attempt_id: str
    outcome: str
    terminal_attempt_ref: Mapping[str, Any]
    completion_ref: Mapping[str, Any]
    result_ref: Mapping[str, Any]
    manifest_ref: Mapping[str, Any]
    index_root_ref: Mapping[str, Any]
    comparison_ref: Mapping[str, Any]
    baseline_ref: Mapping[str, Any] | None
    generation_ancestry_ref: Mapping[str, Any] | None
    stage_count: int
    transition_count: int

    def __post_init__(self):
        for name in ("terminal_attempt_ref", "completion_ref", "result_ref", "manifest_ref",
                     "index_root_ref", "comparison_ref", "baseline_ref",
                     "generation_ancestry_ref"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))


def _sensitivity_canonical(snapshot, reference, *, role, schema):
    from .validation.common import _integrity

    if not isinstance(reference, Mapping) or reference.get("role") != role:
        raise _integrity("Sensitivity object reference has the wrong role.", expected=role)
    raw = snapshot.get_object(reference)
    value = record_document(raw)
    if (not isinstance(value, dict) or record_bytes(value) != raw
            or (schema is not None and value.get("schema") != schema)
            or (schema is not None and value.get("schema_version") != 1)):
        raise _integrity("Sensitivity object is not a canonical supported record.",
                         reference=dict(reference), expected_schema=schema)
    return raw, value


def _sensitivity_slice(value: object, *, ordinal: int, axis: str) -> None:
    from .validation.common import _integrity

    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) != {"start", "stop"}:
        raise _integrity("Sensitivity block slice is malformed.", ordinal=ordinal, axis=axis)
    start, stop = value.get("start"), value.get("stop")
    if (not isinstance(start, int) or isinstance(start, bool) or start < 0
            or not isinstance(stop, int) or isinstance(stop, bool) or stop < start):
        raise _integrity("Sensitivity block slice bounds are malformed.", ordinal=ordinal, axis=axis)


def read_sensitivity_success(snapshot, binding_identity: Mapping[str, Any], *, descriptor,
                             attempt_id: str | None, selection: Mapping[str, Any] | None):
    """Select a fixed complete success or explicit terminal attempt in one snapshot."""
    from .validation.common import _integrity

    task_id = descriptor["task_id"]
    request_sha256 = descriptor["request_sha256"]
    if selection is None:
        if not isinstance(attempt_id, str):
            return None
        pointer = snapshot.read_pointer(f"sensitivity_completion/{task_id}/{attempt_id}")
        if pointer is None:
            return None
        if pointer.get("reference", {}).get("role") != "sensitivity_completion":
            raise _integrity("Sensitivity attempt completion pointer has the wrong role.",
                             task_id=task_id, attempt_id=attempt_id)
        completion_ref = pointer["reference"]
        result_ref = None
    else:
        selected_attempt_id = selection.get("attempt_id")
        completion_ref = selection.get("sensitivity_completion")
        result_ref = selection.get("result_ref")
        if (not isinstance(selected_attempt_id, str)
                or selection.get("event") != "request_success_selected"
                or selection.get("request_sha256") != request_sha256
                or selection.get("task_id") != task_id
                or not isinstance(completion_ref, Mapping)
                or completion_ref.get("role") != "sensitivity_completion"
                or not isinstance(result_ref, Mapping)
                or result_ref.get("role") != "operation_result"):
            raise _integrity("Selected Sensitivity success lacks its sealed completion.",
                             task_id=task_id)
        if attempt_id is not None and selected_attempt_id != attempt_id:
            return None
        attempt_id = selected_attempt_id
    if not isinstance(attempt_id, str):
        return None

    completion_pointer = snapshot.read_pointer(f"sensitivity_completion/{task_id}/{attempt_id}")
    if (completion_pointer is None
            or completion_pointer.get("reference") != dict(completion_ref)):
        raise _integrity("Sensitivity completion pointer differs from its selected terminal.",
                         task_id=task_id, attempt_id=attempt_id)
    _, completion = _sensitivity_canonical(
        snapshot, completion_ref, role="sensitivity_completion",
        schema="scnsim.sensitivity_completion",
    )
    if (completion.get("workspace_instance_id") != binding_identity["workspace_instance_id"]
            or completion.get("plan_sha256") != binding_identity["plan_sha256"]
            or completion.get("request_sha256") != request_sha256
            or completion.get("task_id") != task_id
            or completion.get("attempt_id") != attempt_id
            or completion.get("outcome") not in {"success", "partial"}
            or completion_ref.get("sha256") is None):
        raise _integrity("Sensitivity completion differs from its bound request or attempt.",
                         task_id=task_id, attempt_id=attempt_id)
    if result_ref is not None and completion.get("result_ref") != dict(result_ref):
        raise _integrity("Selected Sensitivity result reference differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    result_ref = completion.get("result_ref")
    manifest_ref = completion.get("manifest_ref")
    index_root_ref = completion.get("index_root_ref")
    if (not isinstance(result_ref, Mapping) or not isinstance(manifest_ref, Mapping)
            or not isinstance(index_root_ref, Mapping)):
        raise _integrity("Sensitivity completion references are malformed.",
                         task_id=task_id, attempt_id=attempt_id)
    _, result = _sensitivity_canonical(
        snapshot, result_ref, role="operation_result", schema="scnsim.sensitivity_result",
    )
    _, manifest = _sensitivity_canonical(
        snapshot, manifest_ref, role="sensitivity_manifest", schema="scnsim.sensitivity_manifest",
    )
    _, index_root = _sensitivity_canonical(
        snapshot, index_root_ref, role="sensitivity_index_root",
        schema="scnsim.sensitivity_index_root",
    )
    identity = {
        "workspace_instance_id": binding_identity["workspace_instance_id"],
        "plan_sha256": binding_identity["plan_sha256"], "request_sha256": request_sha256,
        "task_id": task_id, "attempt_id": attempt_id, "outcome": completion["outcome"],
    }
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise _integrity("Sensitivity manifest differs from its completion identity.",
                         task_id=task_id, attempt_id=attempt_id)
    if (result.get("schema") != "scnsim.sensitivity_result"
            or any(result.get(key) != value for key, value in identity.items())
            or result.get("manifest_ref") != dict(manifest_ref)
            or result.get("block_count") != completion.get("block_count")):
        raise _integrity("Sensitivity result summary differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    request_ref = next((item for item in descriptor.get("artifacts", ())
                        if item.get("role") == "operation_request"), None)
    if (request_ref is None or request_ref.get("sha256") != request_sha256
            or manifest.get("request_ref") != dict(request_ref)):
        raise _integrity("Sensitivity manifest does not reference the bound request.",
                         task_id=task_id, attempt_id=attempt_id)
    for key, value in identity.items():
        if index_root.get(key) != value:
            raise _integrity("Sensitivity block index differs from completion identity.",
                             task_id=task_id, attempt_id=attempt_id, field=key)
    if (index_root.get("result_ref") != dict(result_ref)
            or index_root.get("manifest_ref") != dict(manifest_ref)
            or index_root.get("block_count") != completion.get("block_count")
            or index_root.get("request_ref") != dict(request_ref)):
        raise _integrity("Sensitivity block index differs from its sealed terminal.",
                         task_id=task_id, attempt_id=attempt_id)
    blocks = index_root.get("blocks")
    block_count = completion.get("block_count")
    if (not isinstance(block_count, int) or isinstance(block_count, bool) or block_count < 0
            or not isinstance(blocks, list) or len(blocks) != block_count
            or snapshot.sensitivity_block_count(completion_ref["sha256"]) != block_count):
        raise _integrity("Sensitivity block index count differs from its terminal.",
                         task_id=task_id, attempt_id=attempt_id)
    for ordinal, row in enumerate(blocks):
        if (not isinstance(row, Mapping) or row.get("ordinal") != ordinal
                or isinstance(row.get("ordinal"), bool)
                or not isinstance(row.get("block_id"), str)
                or row.get("kind") not in {"primal", "derivative"}
                or (row.get("quantity_id") is not None
                    and not isinstance(row.get("quantity_id"), str))
                or (row.get("output_slice") is not None
                    and not isinstance(row.get("output_slice"), Mapping))
                or (row.get("parameter_slice") is not None
                    and not isinstance(row.get("parameter_slice"), Mapping))
                or not isinstance(row.get("primal_blocks"), list)
                or not isinstance(row.get("summary_json"), Mapping)
                or not isinstance(row.get("block_ref"), Mapping)
                or row["block_ref"].get("role") != "sensitivity_block"):
            raise _integrity("Sensitivity block index row is malformed.", ordinal=ordinal)
        _sensitivity_slice(row.get("output_slice"), ordinal=ordinal, axis="output")
        _sensitivity_slice(row.get("parameter_slice"), ordinal=ordinal, axis="parameter")
        if row["kind"] == "primal":
            if (row.get("quantity_id") is None or row.get("output_slice") is None
                    or row.get("parameter_slice") is not None or row["primal_blocks"]):
                raise _integrity("Primal Sensitivity block axes or links are malformed.", ordinal=ordinal)
        else:
            if (not row["primal_blocks"]
                    or (row.get("quantity_id") is None) != (row.get("output_slice") is None)):
                raise _integrity("Derivative Sensitivity block axes or links are malformed.", ordinal=ordinal)
            seen_primal_ids = set()
            for link in row["primal_blocks"]:
                if (not isinstance(link, Mapping)
                        or set(link) != {"block_id", "block_ref"}
                        or not isinstance(link.get("block_id"), str)
                        or not isinstance(link.get("block_ref"), Mapping)
                        or link["block_ref"].get("role") != "sensitivity_block"
                        or link["block_id"] in seen_primal_ids):
                    raise _integrity("Derivative Sensitivity primal link is malformed.", ordinal=ordinal)
                seen_primal_ids.add(link["block_id"])
                prior = next((candidate for candidate in blocks[:ordinal]
                              if candidate.get("block_id") == link["block_id"]
                              and candidate.get("kind") == "primal"), None)
                if prior is None or prior.get("block_ref") != dict(link["block_ref"]):
                    raise _integrity("Derivative Sensitivity block lacks its sealed primal.",
                                     ordinal=ordinal, primal_block_id=link["block_id"])
    terminal_attempt_ref = completion.get("terminal_attempt_ref")
    if (not isinstance(terminal_attempt_ref, Mapping)
            or terminal_attempt_ref.get("role") != "attempt"):
        raise _integrity("Sensitivity completion lacks its terminal attempt reference.",
                         task_id=task_id, attempt_id=attempt_id)
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    if (attempt_pointer is None or attempt_pointer.get("reference") != dict(terminal_attempt_ref)):
        raise _integrity("Sensitivity terminal attempt pointer differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    _, attempt = _sensitivity_canonical(
        snapshot, terminal_attempt_ref, role="attempt", schema=None,
    )
    if (attempt.get("attempt_id") != attempt_id
            or attempt.get("status") != completion["outcome"]
            or result_ref not in attempt.get("artifacts", ())
            or manifest_ref not in attempt.get("artifacts", ())
            or index_root_ref not in attempt.get("artifacts", ())):
        raise _integrity("Sensitivity terminal attempt differs from its sealed result.",
                         task_id=task_id, attempt_id=attempt_id)
    state_pointer = snapshot.read_pointer(f"attempt_state/{task_id}/{attempt_id}")
    if state_pointer is None or state_pointer["reference"].get("role") != "task_change":
        raise _integrity("Sensitivity terminal lacks committed task state.",
                         task_id=task_id, attempt_id=attempt_id)
    _, state = _sensitivity_canonical(
        snapshot, state_pointer["reference"], role="task_change", schema=None,
    )
    if (state.get("kind") != "attempt_update"
            or state.get("attempt_id") != attempt_id
            or state.get("status") != completion["outcome"]):
        raise _integrity("Sensitivity task state differs from its terminal outcome.",
                         task_id=task_id, attempt_id=attempt_id)
    if completion["outcome"] == "success":
        success_pointer = snapshot.read_pointer("success/" + request_sha256)
        if selection is not None:
            if success_pointer is None or success_pointer.get("reference", {}).get("role") != "success_selection":
                raise _integrity("Sensitivity success lacks its request selection.", request_sha256=request_sha256)
            selected_success = _sensitivity_decode(snapshot, success_pointer["reference"])
            if selected_success != dict(selection):
                raise _integrity("Sensitivity success differs from its request selection.",
                                 request_sha256=request_sha256)
    elif selection is not None:
        raise _integrity("Partial Sensitivity result cannot be a request success selection.",
                         task_id=task_id, attempt_id=attempt_id)

    selection_record = SensitivityResultSelection(
        workspace_instance_id=identity["workspace_instance_id"],
        plan_sha256=identity["plan_sha256"], request_sha256=request_sha256,
        task_id=task_id, attempt_id=attempt_id, outcome=completion["outcome"],
        attempt_sha256=terminal_attempt_ref["sha256"], completion_ref=dict(completion_ref),
        result_ref=dict(result_ref), manifest_ref=dict(manifest_ref),
        index_root_ref=dict(index_root_ref), block_count=block_count,
    )
    reader = SensitivityResultReader(binding_identity, selection_record)
    return {
        "task_id": task_id, "attempt_id": attempt_id,
        "attempt_sha256": terminal_attempt_ref["sha256"],
        "result_sha256": result_ref["sha256"], "result_ref": dict(result_ref),
        "projection": {"terminal": result, "manifest": manifest,
                       "index_root": {key: index_root[key] for key in (
                           "schema", "schema_version", "block_count", "blocks")}},
        "selection": selection_record, "fixed_reader": reader,
    }


def _sensitivity_decode(snapshot, reference):
    from .validation.common import _integrity

    if not isinstance(reference, Mapping) or reference.get("role") != "success_selection":
        raise _integrity("Sensitivity success selection has the wrong role.")
    raw = snapshot.get_object(reference)
    value = record_document(raw)
    if not isinstance(value, dict) or record_bytes(value) != raw:
        raise _integrity("Sensitivity reference body is not canonical.", reference=dict(reference))
    return value


class SensitivityResultReader:
    """Read selected Sensitivity blocks from short, fixed SQLite snapshots."""

    def __init__(self, binding_identity: Mapping[str, Any], selection: SensitivityResultSelection):
        self._operations = Path(binding_identity["leaf"]) / "operations"
        self._plan_sha256 = binding_identity["plan_sha256"]
        self._workspace_instance_id = binding_identity["workspace_instance_id"]
        self.selection = selection

    def _store(self) -> OperationStore:
        return OperationStore(
            self._operations,
            plan_sha256=self._plan_sha256,
            workspace_instance_id=self._workspace_instance_id,
        )

    @staticmethod
    def _canonical(snapshot, reference, role, schema=None):
        return _sensitivity_canonical(snapshot, reference, role=role, schema=schema)[1]

    def _selection_snapshot(self, snapshot):
        from .validation.common import _integrity

        selected = self.selection
        if (selected.workspace_instance_id != self._workspace_instance_id
                or selected.plan_sha256 != self._plan_sha256):
            raise _integrity("Sensitivity selection belongs to another Workspace binding.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        completion = self._canonical(snapshot, selected.completion_ref,
                                     "sensitivity_completion", "scnsim.sensitivity_completion")
        if (selected.completion_ref.get("sha256") is None
                or selected.completion_ref.get("role") != "sensitivity_completion"
                or completion.get("workspace_instance_id") != selected.workspace_instance_id
                or completion.get("plan_sha256") != selected.plan_sha256
                or completion.get("request_sha256") != selected.request_sha256
                or completion.get("task_id") != selected.task_id
                or completion.get("attempt_id") != selected.attempt_id
                or completion.get("outcome") != selected.outcome
                or completion.get("result_ref") != dict(selected.result_ref)
                or completion.get("manifest_ref") != dict(selected.manifest_ref)
                or completion.get("index_root_ref") != dict(selected.index_root_ref)
                or completion.get("block_count") != selected.block_count):
            raise _integrity("Sensitivity selection differs from its sealed completion.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)

        attempt_pointer = snapshot.read_pointer(f"attempt/{selected.task_id}/{selected.attempt_id}")
        if (attempt_pointer is None
                or attempt_pointer.get("reference", {}).get("role") != "attempt"
                or attempt_pointer.get("reference", {}).get("sha256") != selected.attempt_sha256):
            raise _integrity("Sensitivity terminal attempt reference changed.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        completion_pointer = snapshot.read_pointer(
            f"sensitivity_completion/{selected.task_id}/{selected.attempt_id}"
        )
        if (completion_pointer is None
                or completion_pointer.get("reference") != dict(selected.completion_ref)):
            raise _integrity("Sensitivity completion pointer changed after selection.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        state_pointer = snapshot.read_pointer(
            f"attempt_state/{selected.task_id}/{selected.attempt_id}"
        )
        if state_pointer is None or state_pointer.get("reference", {}).get("role") != "task_change":
            raise _integrity("Sensitivity terminal task state disappeared.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        state = self._canonical(snapshot, state_pointer["reference"], "task_change")
        if (state.get("kind") != "attempt_update"
                or state.get("attempt_id") != selected.attempt_id
                or state.get("status") != selected.outcome):
            raise _integrity("Sensitivity terminal task state differs from its selection.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)

        manifest = self._canonical(snapshot, selected.manifest_ref,
                                   "sensitivity_manifest", "scnsim.sensitivity_manifest")
        result = self._canonical(snapshot, selected.result_ref,
                                 "operation_result", "scnsim.sensitivity_result")
        index_root = self._canonical(snapshot, selected.index_root_ref,
                                     "sensitivity_index_root", "scnsim.sensitivity_index_root")
        identity = {
            "workspace_instance_id": selected.workspace_instance_id,
            "plan_sha256": selected.plan_sha256,
            "request_sha256": selected.request_sha256,
            "task_id": selected.task_id,
            "attempt_id": selected.attempt_id,
            "outcome": selected.outcome,
        }
        if (any(manifest.get(key) != value for key, value in identity.items())
                or any(result.get(key) != value for key, value in identity.items())
                or any(index_root.get(key) != value for key, value in identity.items())
                or result.get("manifest_ref") != dict(selected.manifest_ref)
                or result.get("block_count") != selected.block_count
                or index_root.get("manifest_ref") != dict(selected.manifest_ref)
                or index_root.get("result_ref") != dict(selected.result_ref)
                or index_root.get("block_count") != selected.block_count
                or not isinstance(index_root.get("blocks"), list)
                or len(index_root["blocks"]) != selected.block_count
                or snapshot.sensitivity_block_count(selected.completion_ref["sha256"])
                != selected.block_count):
            raise _integrity("Sensitivity fixed selection differs from its sealed bodies.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        request_pointer = snapshot.read_pointer(f"task/{selected.task_id}")
        if request_pointer is None:
            raise _integrity("Sensitivity task descriptor disappeared.", task_id=selected.task_id)
        from .sqlite_storage import _descriptor
        descriptor = _descriptor(snapshot, selected.task_id)
        request_ref = next((item for item in descriptor.get("artifacts", ())
                            if item.get("role") == "operation_request"), None)
        if (descriptor.get("request_sha256") != selected.request_sha256
                or not isinstance(request_ref, Mapping)
                or manifest.get("request_ref") != dict(request_ref)
                or index_root.get("request_ref") != dict(request_ref)):
            raise _integrity("Sensitivity request differs from its fixed selection.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        return completion, manifest, result, index_root

    @staticmethod
    def _row_matches(indexed, sealed):
        keys = ("ordinal", "block_id", "kind", "block_ref", "quantity_id", "output_slice",
                "parameter_slice", "primal_blocks", "primal_status",
                "derivative_status", "summary_json")
        return all(indexed.get(key) == sealed.get(key) for key in keys)

    def _block(self, snapshot, index_root, selector):
        from .validation.common import _integrity

        blocks = index_root["blocks"]
        row = None
        if isinstance(selector, str):
            row = next((item for item in blocks if item.get("block_id") == selector), None)
        elif isinstance(selector, int) and not isinstance(selector, bool):
            row = next((item for item in blocks if item.get("ordinal") == selector), None)
        if row is None:
            raise KeyError(selector)
        ordinal = row["ordinal"]
        indexed = snapshot.read_sensitivity_block_index(self.selection.completion_ref["sha256"], ordinal)
        if (indexed is None or not self._row_matches(indexed, row)
                or row.get("block_ref", {}).get("role") != "sensitivity_block"):
            raise _integrity("Sensitivity block locator differs from its sealed index.",
                             task_id=self.selection.task_id, ordinal=ordinal)
        body = self._canonical(snapshot, row["block_ref"],
                               "sensitivity_block", "scnsim.sensitivity_block")
        identity = {
            "workspace_instance_id": self.selection.workspace_instance_id,
            "plan_sha256": self.selection.plan_sha256,
            "request_sha256": self.selection.request_sha256,
            "task_id": self.selection.task_id,
            "attempt_id": self.selection.attempt_id,
        }
        fields = ("block_id", "kind", "quantity_id", "output_slice", "parameter_slice",
                  "primal_blocks", "primal_status", "derivative_status")
        if (any(body.get(key) != value for key, value in identity.items())
                or any(body.get(key) != row.get(key) for key in fields)
                or not isinstance(body.get("payload"), Mapping)):
            raise _integrity("Sensitivity block differs from its sealed locator.",
                             task_id=self.selection.task_id, block_id=row.get("block_id"))
        if row["kind"] == "primal":
            if row.get("primal_blocks"):
                raise _integrity("Primal Sensitivity block has a derivative reference.",
                                 block_id=row.get("block_id"))
        else:
            for link in row["primal_blocks"]:
                primal_id = link.get("block_id") if isinstance(link, Mapping) else None
                primal_row = next((item for item in blocks
                                   if item.get("block_id") == primal_id
                                   and item.get("kind") == "primal"
                                   and item.get("ordinal", ordinal) < ordinal), None)
                if (primal_row is None or primal_row.get("block_ref") != link.get("block_ref")):
                    raise _integrity("Derivative Sensitivity block has no preceding sealed primal.",
                                     block_id=row.get("block_id"), primal_block_id=primal_id)
        _sensitivity_slice(row.get("output_slice"), ordinal=ordinal, axis="output")
        _sensitivity_slice(row.get("parameter_slice"), ordinal=ordinal, axis="parameter")
        return body

    def read_block(self, block_id: str | int) -> dict[str, Any]:
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, _, _, index_root = self._selection_snapshot(snapshot)
            return self._block(snapshot, index_root, block_id)

    def project(self, kind: str = "blocks", selector=None):
        """Return authenticated compact block rows without loading value bodies."""
        if kind != "blocks":
            raise ValueError(f"unsupported Sensitivity projection: {kind}")
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, _, _, index_root = self._selection_snapshot(snapshot)
            indexed = snapshot.read_sensitivity_block_indexes(
                self.selection.completion_ref["sha256"]
            )
            sealed = index_root["blocks"]
            if len(indexed) != len(sealed) or any(
                    not self._row_matches(actual, expected)
                    for actual, expected in zip(indexed, sealed)):
                from .validation.common import _integrity
                raise _integrity("Sensitivity SQL block index differs from its sealed root.",
                                 task_id=self.selection.task_id)
            if selector is None:
                rows = sealed
            else:
                rows = [row for row in sealed if row.get("quantity_id") == selector]
            return tuple(_freeze(dict(row)) for row in rows)


def select_result(snapshot, binding_identity: Mapping[str, Any], completion_ref: Mapping[str, Any],
                  result_ref: Mapping[str, Any], *, attempt_sha256: str) -> ResultSelection:
    """Resolve a selected success to its sealed, detached metadata in one snapshot."""
    from .validation.common import _integrity

    def read(reference, role, schema=None):
        if reference.get("role") != role:
            raise _integrity("Selected completion reference has the wrong role.",
                             expected=role, actual=reference.get("role"))
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)):
            raise _integrity("Selected completion object is malformed.", reference=dict(reference))
        return value

    completion = read(completion_ref, "workspace_completion", "scnsim.workspace_completion")
    if completion.get("schema_version") != 1:
        raise _integrity("Selected completion version is unsupported.")
    expected_identity = {
        "workspace_instance_id": binding_identity["workspace_instance_id"],
        "plan_sha256": binding_identity["plan_sha256"],
    }
    if any(completion.get(key) != value for key, value in expected_identity.items()):
        raise _integrity("Selected completion differs from its bound Workspace.")
    if (completion.get("result_ref") != dict(result_ref)
            or result_ref.get("role") != "operation_result"):
        raise _integrity("Selected completion differs from its committed result.")
    root_ref = completion.get("index_root_ref")
    root = read(root_ref, "workspace_completion_index_root",
                "scnsim.workspace_completion_index_root")
    for key in ("workspace_instance_id", "plan_sha256", "request_sha256", "task_id", "attempt_id",
                "result_ref", "baseline_ref", "generation_ancestry_ref", "generation_count",
                "candidate_count", "best_ordinal"):
        if root.get(key) != completion.get(key):
            raise _integrity("Selected completion index root differs from completion.", field=key)
    generations = root.get("generations")
    if not isinstance(generations, list) or len(generations) != completion.get("generation_count"):
        raise _integrity("Selected completion generation index count is malformed.")
    sql_generations = snapshot.completion_generation_count(completion_ref["sha256"])
    sql_candidates = snapshot.completion_candidate_count(completion_ref["sha256"])
    if (sql_generations != completion["generation_count"]
            or sql_candidates != completion["candidate_count"]):
        raise _integrity("Selected SQL index counts differ from the sealed completion.")
    best_ordinal = completion.get("best_ordinal")
    if best_ordinal == 0:
        best_locator = {"ordinal": 0, "generation": 0,
                        "block_ref": completion["baseline_ref"]}
    else:
        best_locator = snapshot.read_candidate_index(completion_ref["sha256"], best_ordinal)
        if best_locator is None:
            raise _integrity("Selected best candidate is absent from its sealed SQL index.",
                             best_ordinal=best_ordinal)
        generation_row = next((row for row in generations
                               if row.get("generation") == best_locator.get("generation")), None)
        if generation_row is None:
            raise _integrity("Selected best candidate is outside sealed generation ancestry.",
                             best_ordinal=best_ordinal)
        candidate_index = read(generation_row["candidate_index_ref"], "workspace_candidate_index",
                               "scnsim.workspace_candidate_index")
        sealed_row = next((row for row in candidate_index.get("candidates", ())
                           if row.get("ordinal") == best_ordinal), None)
        if sealed_row != best_locator:
            raise _integrity("Selected best SQL locator differs from its sealed index block.",
                             best_ordinal=best_ordinal)
    terminal_attempt_ref = completion.get("terminal_attempt_ref")
    if (not isinstance(terminal_attempt_ref, Mapping)
            or terminal_attempt_ref.get("role") != "attempt"):
        raise _integrity("Selected completion lacks its terminal attempt reference.",
                         task_id=completion.get("task_id"), attempt_id=completion.get("attempt_id"))
    attempt_pointer = snapshot.read_pointer(
        f"attempt/{completion['task_id']}/{completion['attempt_id']}"
    )
    if (attempt_pointer is None or attempt_pointer.get("reference") != dict(terminal_attempt_ref)):
        raise _integrity("Terminal attempt reference differs from the committed attempt pointer.",
                         task_id=completion.get("task_id"), attempt_id=completion.get("attempt_id"))
    attempt = read(terminal_attempt_ref, "attempt")
    if (terminal_attempt_ref.get("sha256") != attempt_sha256
            or attempt.get("attempt_id") != completion.get("attempt_id")
            or attempt.get("status") != "success"
            or result_ref not in attempt.get("artifacts", ())):
        raise _integrity("Terminal attempt differs from the selected completed result.",
                         task_id=completion.get("task_id"), attempt_id=completion.get("attempt_id"))
    return ResultSelection(
        workspace_instance_id=completion["workspace_instance_id"],
        plan_sha256=completion["plan_sha256"],
        request_sha256=completion["request_sha256"],
        task_id=completion["task_id"], attempt_id=completion["attempt_id"],
        attempt_sha256=attempt_sha256, terminal_attempt_ref=dict(terminal_attempt_ref),
        completion_ref=dict(completion_ref), completion_sha256=completion_ref["sha256"],
        result_ref=dict(result_ref), index_root_ref=dict(root_ref),
        index_root_sha256=root_ref["sha256"],
        generation_ancestry_ref=completion.get("generation_ancestry_ref"),
        generation_count=completion["generation_count"], candidate_count=completion["candidate_count"],
        baseline_locator={"block_ref": completion["baseline_ref"], "ordinal": 0, "generation": 0},
        best_locator=best_locator,
    )


def read_optimization_success(snapshot, binding_identity: Mapping[str, Any], *,
                              descriptor: Mapping[str, Any], attempt_id: str,
                              completion_ref: Mapping[str, Any],
                              result_ref: Mapping[str, Any] | None = None,
                              selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the fixed reader for one exact successful v6 Optimization attempt.

    The SQL success selection and attempt-state pointers are read in the same
    snapshot by the caller. This function verifies their canonical objects,
    then retains only detached result-selection metadata and paths.
    """
    from .validation.common import _integrity

    def read(reference, role, schema=None):
        if not isinstance(reference, Mapping) or reference.get("role") != role:
            raise _integrity("Selected Optimization reference has the wrong role.",
                             expected=role, actual=None if not isinstance(reference, Mapping)
                             else reference.get("role"))
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)):
            raise _integrity("Selected Optimization object is malformed or noncanonical.",
                             reference=dict(reference), expected_schema=schema)
        return raw, value

    task_id = descriptor.get("task_id")
    request_sha256 = descriptor.get("request_sha256")
    if (not isinstance(task_id, str) or not isinstance(request_sha256, str)
            or descriptor.get("environment", {}).get("environment_sha256") is None):
        raise _integrity("Selected Optimization task identity is malformed.")
    _, completion = read(completion_ref, "workspace_completion", "scnsim.workspace_completion")
    if result_ref is None:
        result_ref = completion.get("result_ref")
    if not isinstance(result_ref, Mapping) or result_ref.get("role") != "operation_result":
        raise _integrity("Selected Optimization result has the wrong role.",
                         actual=None if not isinstance(result_ref, Mapping) else result_ref.get("role"))
    _, terminal = read(result_ref, "operation_result")
    if terminal.get("type") != "optimization":
        raise _integrity("Selected Optimization result has the wrong result kind.", task_id=task_id)
    if (completion.get("workspace_instance_id") != binding_identity["workspace_instance_id"]
            or completion.get("plan_sha256") != binding_identity["plan_sha256"]
            or completion.get("request_sha256") != request_sha256
            or completion.get("task_id") != task_id
            or completion.get("attempt_id") != attempt_id
            or completion.get("result_ref") != dict(result_ref)):
        raise _integrity("Selected Optimization completion differs from its bound attempt.",
                         task_id=task_id, attempt_id=attempt_id)

    completion_pointer = snapshot.read_pointer(f"completion/{task_id}/{attempt_id}")
    if (completion_pointer is None
            or completion_pointer.get("reference") != dict(completion_ref)):
        raise _integrity("Selected Optimization completion differs from its committed pointer.",
                         task_id=task_id, attempt_id=attempt_id)

    terminal_attempt_ref = completion.get("terminal_attempt_ref")
    if not isinstance(terminal_attempt_ref, Mapping) or terminal_attempt_ref.get("role") != "attempt":
        raise _integrity("Selected Optimization completion lacks its terminal attempt.",
                         task_id=task_id, attempt_id=attempt_id)
    attempt_raw, attempt = read(terminal_attempt_ref, "attempt")
    attempt_sha256 = sha256(attempt_raw).hexdigest()
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    if (attempt_pointer is None or attempt_pointer.get("reference") != dict(terminal_attempt_ref)
            or attempt.get("attempt_id") != attempt_id or attempt.get("status") != "success"
            or result_ref not in attempt.get("artifacts", ())):
        raise _integrity("Selected Optimization terminal attempt differs from its sealed completion.",
                         task_id=task_id, attempt_id=attempt_id)

    if selection is not None and (
            selection.get("attempt_id") != attempt_id
            or selection.get("task_id") != task_id
            or selection.get("request_sha256") != request_sha256
            or selection.get("result_ref") != dict(result_ref)
            or selection.get("workspace_completion") != dict(completion_ref)):
        raise _integrity("Request success selection differs from its completed Optimization.",
                         task_id=task_id, attempt_id=attempt_id)

    state_pointer = snapshot.read_pointer(f"attempt_state/{task_id}/{attempt_id}")
    if state_pointer is None or state_pointer["reference"].get("role") != "task_change":
        raise _integrity("Selected Optimization has no committed final task state.",
                         task_id=task_id, attempt_id=attempt_id)
    _, state = read(state_pointer["reference"], "task_change")
    if (state.get("kind") != "attempt_update" or state.get("attempt_id") != attempt_id
            or state.get("status") != "success"):
        raise _integrity("Selected Optimization attempt is not durably successful.",
                         task_id=task_id, attempt_id=attempt_id)

    if (attempt.get("attempt_id") != attempt_id
            or attempt.get("resume_from") is not None and not isinstance(attempt.get("resume_from"), Mapping)):
        raise _integrity("Selected Optimization attempt allocation is malformed.",
                         task_id=task_id, attempt_id=attempt_id)

    result_selection = select_result(
        snapshot, binding_identity, completion_ref, result_ref, attempt_sha256=attempt_sha256,
    )
    fixed_reader = FixedResultReader(binding_identity, result_selection)
    return {
        "task_id": task_id,
        "environment_sha256": descriptor["environment"]["environment_sha256"],
        "attempt_sha256": attempt_sha256,
        "result_sha256": result_ref["sha256"],
        "result_ref": dict(result_ref),
        "projection": {"terminal": terminal},
        "fixed_reader": fixed_reader,
        "selection": result_selection,
        "checkpoint_available": (
            snapshot.read_pointer(f"checkpoint/{task_id}/{attempt_id}") is not None
            or isinstance(attempt.get("resume_from"), Mapping)
        ),
        "attempt_id": attempt_id,
    }


def read_physical_optimization_success(
    snapshot, binding_identity: Mapping[str, Any], *, descriptor: Mapping[str, Any],
    attempt_id: str, completion_ref: Mapping[str, Any],
    result_ref: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one exact v7 physical Optimization success or partial terminal."""
    from .validation.common import _integrity

    def read(reference, role, schema=None):
        if not isinstance(reference, Mapping) or reference.get("role") != role:
            raise _integrity("Physical Optimization reference has the wrong role.", expected=role)
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)
                or (schema is not None and value.get("schema_version") != 1)):
            raise _integrity("Physical Optimization object is not canonical.",
                             reference=dict(reference), expected_schema=schema)
        return raw, value

    task_id = descriptor.get("task_id")
    request_sha256 = descriptor.get("request_sha256")
    if not isinstance(task_id, str) or not isinstance(request_sha256, str):
        raise _integrity("Physical Optimization task identity is malformed.")
    if selection is not None:
        if (selection.get("event") != "request_success_selected"
                or selection.get("task_id") != task_id
                or selection.get("request_sha256") != request_sha256
                or selection.get("attempt_id") != attempt_id
                or selection.get("workspace_completion") != dict(completion_ref)):
            raise _integrity("Physical Optimization success selector differs from its completion.",
                             task_id=task_id, attempt_id=attempt_id)
        selected_result_ref = selection.get("result_ref")
        if not isinstance(selected_result_ref, Mapping):
            raise _integrity("Physical Optimization success selector lacks its result reference.",
                             task_id=task_id, attempt_id=attempt_id)
        if result_ref is not None and dict(result_ref) != dict(selected_result_ref):
            raise _integrity("Physical Optimization result reference differs from its selector.",
                             task_id=task_id, attempt_id=attempt_id)
        result_ref = selected_result_ref

    _, completion = read(completion_ref, "workspace_completion", "scnsim.workspace_completion")
    if result_ref is None:
        result_ref = completion.get("result_ref")
    if not isinstance(result_ref, Mapping) or result_ref.get("role") != "operation_result":
        raise _integrity("Physical Optimization completion lacks its result reference.",
                         task_id=task_id, attempt_id=attempt_id)
    outcome = completion.get("outcome")
    if outcome not in {"success", "partial"}:
        raise _integrity("Physical Optimization terminal outcome is malformed.",
                         task_id=task_id, attempt_id=attempt_id)
    if (completion.get("workspace_instance_id") != binding_identity["workspace_instance_id"]
            or completion.get("plan_sha256") != binding_identity["plan_sha256"]
            or completion.get("request_sha256") != request_sha256
            or completion.get("task_id") != task_id
            or completion.get("attempt_id") != attempt_id
            or completion.get("result_ref") != dict(result_ref)):
        raise _integrity("Physical Optimization completion differs from its bound attempt.",
                         task_id=task_id, attempt_id=attempt_id)
    completion_pointer = snapshot.read_pointer(f"completion/{task_id}/{attempt_id}")
    if completion_pointer is None or completion_pointer.get("reference") != dict(completion_ref):
        raise _integrity("Physical Optimization completion differs from its committed pointer.",
                         task_id=task_id, attempt_id=attempt_id)

    _, terminal = read(result_ref, "operation_result", "scnsim.optimization_result")
    if (terminal.get("type") != "optimization"
            or terminal.get("outcome") != outcome
            or terminal.get("manifest_ref") != completion.get("manifest_ref")):
        raise _integrity("Physical Optimization result differs from its completion.",
                         task_id=task_id, attempt_id=attempt_id)
    manifest_ref = completion.get("manifest_ref")
    index_root_ref = completion.get("index_root_ref")
    if (not isinstance(manifest_ref, Mapping) or manifest_ref.get("role") != "optimization_manifest"
            or not isinstance(index_root_ref, Mapping)
            or index_root_ref.get("role") != "optimization_index_root"):
        raise _integrity("Physical Optimization completion references are malformed.",
                         task_id=task_id, attempt_id=attempt_id)
    _, manifest = read(manifest_ref, "optimization_manifest", "scnsim.optimization_manifest")
    _, index_root = read(index_root_ref, "optimization_index_root", "scnsim.optimization_index_root")
    identity = {
        "workspace_instance_id": binding_identity["workspace_instance_id"],
        "plan_sha256": binding_identity["plan_sha256"],
        "request_sha256": request_sha256, "task_id": task_id,
        "attempt_id": attempt_id, "outcome": outcome,
    }
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise _integrity("Physical Optimization manifest differs from its terminal identity.",
                         task_id=task_id, attempt_id=attempt_id)
    request_ref = next((item for item in descriptor.get("artifacts", ())
                        if item.get("role") == "operation_request"), None)
    if (not isinstance(request_ref, Mapping) or request_ref.get("sha256") != request_sha256
            or manifest.get("request_ref") != dict(request_ref)):
        raise _integrity("Physical Optimization manifest differs from its bound request.",
                         task_id=task_id, attempt_id=attempt_id)
    if (manifest.get("declarations") is None
            or not isinstance(manifest.get("declarations"), Mapping)
            or not isinstance(manifest.get("stage_ids"), list)):
        raise _integrity("Physical Optimization manifest declarations are malformed.",
                         task_id=task_id, attempt_id=attempt_id)
    for key, value in identity.items():
        if index_root.get(key) != value:
            raise _integrity("Physical Optimization index root differs from its terminal identity.",
                             task_id=task_id, attempt_id=attempt_id, field=key)
    for key, reference in (("result_ref", result_ref), ("manifest_ref", manifest_ref),
                           ("comparison_ref", completion.get("comparison_ref"))):
        if index_root.get(key) != reference:
            raise _integrity("Physical Optimization index root differs from its completion.",
                             task_id=task_id, attempt_id=attempt_id, field=key)
    generation_ancestry_ref = completion.get("generation_ancestry_ref")
    if index_root.get("generation_ancestry_ref") != generation_ancestry_ref:
        raise _integrity("Physical Optimization generation ancestry differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    if generation_ancestry_ref is not None:
        if (not isinstance(generation_ancestry_ref, Mapping)
                or generation_ancestry_ref.get("role") != "optimization_generation_ancestry"):
            raise _integrity("Physical Optimization generation ancestry reference is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
        _, ancestry = read(generation_ancestry_ref, "optimization_generation_ancestry",
                           "scnsim.optimization_generation_ancestry")
        if (any(ancestry.get(key) != value for key, value in identity.items())
                or not isinstance(ancestry.get("sources"), list)):
            raise _integrity("Physical Optimization generation ancestry is misbound.",
                             task_id=task_id, attempt_id=attempt_id)
        seen_sources = set()
        for source in ancestry["sources"]:
            if (not isinstance(source, Mapping)
                    or not isinstance(source.get("task_id"), str)
                    or not isinstance(source.get("attempt_id"), str)
                    or not isinstance(source.get("checkpoint_ref"), Mapping)
                    or source["checkpoint_ref"].get("role") != "optimization_checkpoint"
                    or not isinstance(source.get("batch_refs"), list)
                    or not source["batch_refs"]
                    or any(not isinstance(reference, Mapping)
                           or reference.get("role") != "optimization_stage_batch"
                           for reference in source["batch_refs"])):
                raise _integrity("Physical Optimization source-prefix entry is malformed.",
                                 task_id=task_id, attempt_id=attempt_id)
            source_identity = (source["task_id"], source["attempt_id"])
            if source_identity in seen_sources:
                raise _integrity("Physical Optimization source-prefix attempt is repeated.",
                                 task_id=task_id, attempt_id=attempt_id)
            seen_sources.add(source_identity)
    stages = index_root.get("stages")
    transitions = index_root.get("transitions")
    if (not isinstance(stages, list) or not isinstance(transitions, list)
            or completion.get("stage_count") != len(stages)
            or completion.get("transition_count") != len(transitions)
            or manifest.get("stage_ids") != [row.get("stage_id") for row in stages]):
        raise _integrity("Physical Optimization index counts or order differ from the completion.",
                         task_id=task_id, attempt_id=attempt_id)
    sql_stages = snapshot.read_optimization_stage_indexes(completion_ref["sha256"])
    sql_transitions = snapshot.read_optimization_transitions(completion_ref["sha256"])
    if len(sql_stages) != len(stages) or len(sql_transitions) != len(transitions):
        raise _integrity("Physical Optimization SQL index counts differ from the sealed root.",
                         task_id=task_id, attempt_id=attempt_id)
    for ordinal, (root_row, sql_row) in enumerate(zip(stages, sql_stages, strict=True)):
        if (not isinstance(root_row, Mapping) or sql_row.get("stage_ordinal") != ordinal
                or sql_row.get("stage_id") != root_row.get("stage_id")
                or sql_row.get("model_identity") != root_row.get("model_identity")
                or sql_row.get("summary_ref") != root_row.get("summary_ref")
                or sql_row.get("history_ref") != root_row.get("history_ref")):
            raise _integrity("Physical Optimization SQL stage locator differs from its sealed root.",
                             stage_ordinal=ordinal)
    for ordinal, (root_row, sql_row) in enumerate(zip(transitions, sql_transitions, strict=True)):
        if (not isinstance(root_row, Mapping) or sql_row.get("ordinal") != ordinal
                or sql_row.get("transition_id") != root_row.get("transition_id")
                or sql_row.get("transition_ref") != root_row.get("transition_ref")):
            raise _integrity("Physical Optimization transition locator differs from its sealed root.",
                             transition_ordinal=ordinal)

    terminal_attempt_ref = completion.get("terminal_attempt_ref")
    if not isinstance(terminal_attempt_ref, Mapping) or terminal_attempt_ref.get("role") != "attempt":
        raise _integrity("Physical Optimization completion lacks its terminal attempt.",
                         task_id=task_id, attempt_id=attempt_id)
    comparison_ref = completion.get("comparison_ref")
    if not isinstance(comparison_ref, Mapping) or comparison_ref.get("role") != "optimization_comparison":
        raise _integrity("Physical Optimization completion lacks its comparison index.",
                         task_id=task_id, attempt_id=attempt_id)
    attempt_pointer = snapshot.read_pointer(f"attempt/{task_id}/{attempt_id}")
    if attempt_pointer is None or attempt_pointer.get("reference") != dict(terminal_attempt_ref):
        raise _integrity("Physical Optimization terminal attempt pointer differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    attempt_raw, attempt = read(terminal_attempt_ref, "attempt")
    attempt_sha256 = sha256(attempt_raw).hexdigest()
    if (attempt.get("attempt_id") != attempt_id or attempt.get("status") != outcome
            or any(ref not in attempt.get("artifacts", ()) for ref in
                   (result_ref, manifest_ref, comparison_ref, index_root_ref))):
        raise _integrity("Physical Optimization terminal attempt differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    state_pointer = snapshot.read_pointer(f"attempt_state/{task_id}/{attempt_id}")
    if state_pointer is None or state_pointer.get("reference", {}).get("role") != "task_change":
        raise _integrity("Physical Optimization terminal task state is absent.",
                         task_id=task_id, attempt_id=attempt_id)
    _, state = read(state_pointer["reference"], "task_change")
    if (state.get("kind") != "attempt_update" or state.get("attempt_id") != attempt_id
            or state.get("status") != outcome):
        raise _integrity("Physical Optimization terminal task state differs from completion.",
                         task_id=task_id, attempt_id=attempt_id)
    if outcome == "success":
        if selection is not None:
            success_pointer = snapshot.read_pointer("success/" + request_sha256)
            if success_pointer is None or success_pointer.get("reference", {}).get("role") != "success_selection":
                raise _integrity("Physical Optimization success pointer is absent.", task_id=task_id)
            _, selected = read(success_pointer["reference"], "success_selection")
            if selected != dict(selection):
                raise _integrity("Physical Optimization request selection differs from its completion.",
                                 task_id=task_id, attempt_id=attempt_id)
    elif selection is not None:
        raise _integrity("Partial Physical Optimization cannot be request-selected.",
                         task_id=task_id, attempt_id=attempt_id)

    selection_record = PhysicalOptimizationResultSelection(
        workspace_instance_id=identity["workspace_instance_id"],
        plan_sha256=identity["plan_sha256"], request_sha256=request_sha256,
        task_id=task_id, attempt_id=attempt_id, outcome=outcome,
        terminal_attempt_ref=dict(terminal_attempt_ref), completion_ref=dict(completion_ref),
        result_ref=dict(result_ref), manifest_ref=dict(manifest_ref),
        index_root_ref=dict(index_root_ref), comparison_ref=dict(comparison_ref),
        baseline_ref=None if completion.get("baseline_ref") is None else dict(completion["baseline_ref"]),
        generation_ancestry_ref=(None if generation_ancestry_ref is None
                                  else dict(generation_ancestry_ref)),
        stage_count=len(stages), transition_count=len(transitions),
    )
    reader = FixedPhysicalOptimizationReader(binding_identity, selection_record)
    from .sqlite_storage import _physical_checkpoint_from_snapshot

    checkpoint_store = OperationStore(
        Path(binding_identity["leaf"]) / "operations",
        plan_sha256=binding_identity["plan_sha256"],
        workspace_instance_id=binding_identity["workspace_instance_id"],
    )
    committed_checkpoint = _physical_checkpoint_from_snapshot(
        snapshot, checkpoint_store, task_id=task_id, attempt_id=attempt_id,
        expected_request_sha256=request_sha256,
    )
    if committed_checkpoint is None and attempt.get("resume_from") is not None:
        resume_from = attempt["resume_from"]
        if (not isinstance(resume_from, Mapping)
                or set(resume_from) != {"task_id", "attempt_id", "checkpoint_ref"}
                or resume_from.get("task_id") != task_id
                or not isinstance(resume_from.get("attempt_id"), str)
                or not isinstance(resume_from.get("checkpoint_ref"), Mapping)):
            raise _integrity("Physical Optimization resume source is malformed.",
                             task_id=task_id, attempt_id=attempt_id)
        committed_checkpoint = _physical_checkpoint_from_snapshot(
            snapshot, checkpoint_store, task_id=task_id,
            attempt_id=resume_from["attempt_id"],
            expected_request_sha256=request_sha256,
            reference=resume_from["checkpoint_ref"],
        )
        if committed_checkpoint is None:
            raise _integrity("Physical Optimization resume source has no committed checkpoint.",
                             task_id=task_id, attempt_id=attempt_id)
    checkpoint_available = committed_checkpoint is not None
    return {
        "task_id": task_id, "attempt_id": attempt_id,
        "attempt_sha256": attempt_sha256, "result_sha256": result_ref["sha256"],
        "result_ref": dict(result_ref), "projection": {"terminal": terminal},
        "selection": selection_record, "fixed_reader": reader,
        "checkpoint_available": checkpoint_available,
    }


class FixedPhysicalOptimizationReader:
    """Short-snapshot projections from one fixed v7 Optimization completion."""

    def __init__(self, binding_identity: Mapping[str, Any], selection: PhysicalOptimizationResultSelection):
        self._operations = Path(binding_identity["leaf"]) / "operations"
        self._plan_sha256 = binding_identity["plan_sha256"]
        self._workspace_instance_id = binding_identity["workspace_instance_id"]
        self.selection = selection

    def _store(self) -> OperationStore:
        return OperationStore(self._operations, plan_sha256=self._plan_sha256,
                              workspace_instance_id=self._workspace_instance_id)

    def _canonical(self, snapshot, reference, role, schema, *, source_identity=None):
        from .validation.common import _integrity
        if not isinstance(reference, Mapping) or reference.get("role") != role:
            raise _integrity("Physical Optimization reference has the wrong role.", expected=role)
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or value.get("schema") != schema or value.get("schema_version") != 1):
            raise _integrity("Physical Optimization object is not canonical.",
                             reference=dict(reference), expected_schema=schema)
        if source_identity is None:
            identity = {
                key: getattr(self.selection, key) for key in (
                    "workspace_instance_id", "plan_sha256", "request_sha256", "task_id", "attempt_id",
                )
            }
        else:
            identity = dict(source_identity)
        if any(value.get(key) != expected for key, expected in identity.items()):
            raise _integrity("Physical Optimization object differs from its fixed selection.",
                             reference=dict(reference))
        return value

    def _selection_snapshot(self, snapshot):
        from .validation.common import _integrity
        selected = self.selection
        completion = self._canonical(snapshot, selected.completion_ref, "workspace_completion",
                                     "scnsim.workspace_completion")
        if (completion.get("outcome") != selected.outcome
                or completion.get("result_ref") != dict(selected.result_ref)
                or completion.get("manifest_ref") != dict(selected.manifest_ref)
                or completion.get("index_root_ref") != dict(selected.index_root_ref)
                or completion.get("terminal_attempt_ref") != dict(selected.terminal_attempt_ref)
                or completion.get("stage_count") != selected.stage_count
                or completion.get("transition_count") != selected.transition_count):
            raise _integrity("Physical Optimization fixed selection differs from completion.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        completion_pointer = snapshot.read_pointer(
            f"completion/{selected.task_id}/{selected.attempt_id}"
        )
        attempt_pointer = snapshot.read_pointer(
            f"attempt/{selected.task_id}/{selected.attempt_id}"
        )
        if (completion_pointer is None or completion_pointer.get("reference") != dict(selected.completion_ref)
                or attempt_pointer is None or attempt_pointer.get("reference") != dict(selected.terminal_attempt_ref)):
            raise _integrity("Physical Optimization fixed pointers changed.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        root = self._canonical(snapshot, selected.index_root_ref,
                               "optimization_index_root", "scnsim.optimization_index_root")
        if (root.get("result_ref") != dict(selected.result_ref)
                or root.get("manifest_ref") != dict(selected.manifest_ref)
                or root.get("comparison_ref") != dict(selected.comparison_ref)
                or root.get("generation_ancestry_ref") != (
                    None if selected.generation_ancestry_ref is None
                    else dict(selected.generation_ancestry_ref))
                or len(root.get("stages", ())) != selected.stage_count
                or len(root.get("transitions", ())) != selected.transition_count):
            raise _integrity("Physical Optimization fixed index root changed.",
                             task_id=selected.task_id, attempt_id=selected.attempt_id)
        ancestry_ref = selected.generation_ancestry_ref
        if ancestry_ref is not None:
            ancestry = self._canonical(
                snapshot, ancestry_ref, "optimization_generation_ancestry",
                "scnsim.optimization_generation_ancestry",
            )
            if (any(ancestry.get(key) != getattr(selected, key) for key in (
                    "workspace_instance_id", "plan_sha256", "request_sha256", "task_id", "attempt_id"))
                    or not isinstance(ancestry.get("sources"), list)):
                raise _integrity("Physical Optimization fixed source ancestry is malformed.",
                                 task_id=selected.task_id, attempt_id=selected.attempt_id)
            root = dict(root)
            root["_verified_generation_ancestry"] = ancestry
        return completion, root

    def _candidate(self, snapshot, reference, *, model_identity=None, stage_id=None, root=None):
        from .validation.common import _integrity
        locator = dict(reference) if isinstance(reference, Mapping) and "candidate_ref" in reference else {
            "kind": "evaluation", "stage_id": stage_id,
            "evaluation_ordinal": None, "candidate_ref": dict(reference),
        }
        candidate_ref = locator.get("candidate_ref")
        source_task_id = locator.get("source_task_id", self.selection.task_id)
        source_attempt_id = locator.get("source_attempt_id", self.selection.attempt_id)
        if (not isinstance(candidate_ref, Mapping)
                or not isinstance(source_task_id, str) or not isinstance(source_attempt_id, str)):
            raise _integrity("Physical Optimization candidate source locator is malformed.")
        if (source_task_id, source_attempt_id) == (self.selection.task_id, self.selection.attempt_id):
            value = self._canonical(snapshot, candidate_ref, "optimization_candidate",
                                    "scnsim.optimization_candidate")
            if locator.get("source_batch_ref") is not None:
                batch_ref = locator["source_batch_ref"]
                batch = self._canonical(snapshot, batch_ref, "optimization_stage_batch",
                                         "scnsim.optimization_stage_batch")
                inherited_baseline = (
                    value.get("kind") == "baseline"
                    and isinstance(value.get("source_locator"), Mapping)
                )
                if (value.get("kind") == "baseline"
                        and value.get("source_locator") is not None
                        and not inherited_baseline):
                    raise _integrity("Physical Optimization inherited baseline provenance is malformed.")
                if not any(isinstance(row, Mapping)
                           and row.get("candidate_ref") == dict(candidate_ref)
                           and row.get("kind") == value.get("kind")
                           and row.get("evaluation_ordinal") == value.get("evaluation_ordinal")
                           and (inherited_baseline
                                or batch.get("stage_id") == value.get("stage_id"))
                           for row in batch.get("rows", ())):
                    raise _integrity("Physical Optimization candidate is absent from its committed batch.",
                                     stage_id=stage_id)
        else:
            if root is None or not isinstance(locator.get("source_batch_ref"), Mapping):
                raise _integrity("Physical Optimization source candidate lacks fixed ancestry.",
                                 task_id=source_task_id, attempt_id=source_attempt_id)
            ancestry = root.get("_verified_generation_ancestry")
            source = next((item for item in ancestry.get("sources", ())
                           if item.get("task_id") == source_task_id
                           and item.get("attempt_id") == source_attempt_id), None) if isinstance(ancestry, Mapping) else None
            batch_ref = locator["source_batch_ref"]
            if (source is None or dict(batch_ref) not in source.get("batch_refs", ())):
                raise _integrity("Physical Optimization candidate is outside fixed source ancestry.",
                                 task_id=source_task_id, attempt_id=source_attempt_id)
            from .sqlite_storage import (
                _descriptor, _physical_candidate_pointer, _physical_request,
            )
            store = self._store()
            descriptor = _descriptor(snapshot, source_task_id)
            request_sha256, _, _ = _physical_request(
                snapshot, store, descriptor, task_id=source_task_id,
                attempt_id=source_attempt_id,
            )
            if request_sha256 != self.selection.request_sha256:
                raise _integrity("Physical Optimization source candidate belongs to another request.",
                                 task_id=source_task_id, attempt_id=source_attempt_id)
            attempt_pointer = snapshot.read_pointer(f"attempt/{source_task_id}/{source_attempt_id}")
            from .sqlite_storage import _physical_baseline_pointer
            candidate_pointer_name = (
                _physical_baseline_pointer(source_task_id, source_attempt_id)
                if locator.get("kind") == "baseline" else
                _physical_candidate_pointer(source_task_id, source_attempt_id,
                                            locator.get("stage_id"),
                                            locator.get("evaluation_ordinal"))
            )
            candidate_pointer = snapshot.read_pointer(candidate_pointer_name)
            if (attempt_pointer is None or attempt_pointer.get("reference", {}).get("role") != "attempt"
                    or candidate_pointer is None
                    or candidate_pointer.get("reference") != dict(candidate_ref)):
                raise _integrity("Physical Optimization source candidate pointer changed.",
                                 task_id=source_task_id, attempt_id=source_attempt_id)
            batch = self._canonical(
                snapshot, batch_ref, "optimization_stage_batch",
                "scnsim.optimization_stage_batch",
                source_identity={
                    "workspace_instance_id": self.selection.workspace_instance_id,
                    "plan_sha256": self.selection.plan_sha256,
                    "request_sha256": request_sha256,
                    "task_id": source_task_id,
                    "attempt_id": source_attempt_id,
                },
            )
            candidate_kind = locator.get("kind", "evaluation")
            if (batch.get("task_id") != source_task_id
                    or batch.get("attempt_id") != source_attempt_id
                    or batch.get("request_sha256") != request_sha256
                    or (candidate_kind != "baseline"
                        and batch.get("stage_id") != locator.get("stage_id"))
                    or not any(isinstance(row, Mapping)
                               and row.get("candidate_ref") == dict(candidate_ref)
                               and row.get("kind") == candidate_kind
                               and row.get("evaluation_ordinal") == locator.get("evaluation_ordinal")
                               for row in batch.get("rows", ()) )):
                raise _integrity("Physical Optimization source candidate differs from its exact batch row.",
                                 task_id=source_task_id, attempt_id=source_attempt_id)
            raw = snapshot.get_object(candidate_ref)
            value = record_document(raw)
            if (not isinstance(value, dict) or record_bytes(value) != raw
                    or value.get("schema") != "scnsim.optimization_candidate"
                    or value.get("schema_version") != 1
                    or value.get("workspace_instance_id") != self.selection.workspace_instance_id
                    or value.get("plan_sha256") != self.selection.plan_sha256
                    or value.get("request_sha256") != self.selection.request_sha256
                    or value.get("task_id") != source_task_id
                    or value.get("attempt_id") != source_attempt_id
                    or value.get("stage_id") != locator.get("stage_id")
                    or value.get("evaluation_ordinal") != locator.get("evaluation_ordinal")
                    or value.get("kind") != locator.get("kind", "evaluation")):
                raise _integrity("Physical Optimization source candidate differs from its fixed locator.",
                                 task_id=source_task_id, attempt_id=source_attempt_id)
        if (stage_id is not None and value.get("stage_id") != stage_id
                or model_identity is not None and value.get("model_identity") != dict(model_identity)):
            raise _integrity("Physical Optimization candidate differs from its model locator.",
                             stage_id=stage_id)
        dependency_refs = value.get("dependencies")
        if not isinstance(dependency_refs, Mapping):
            raise _integrity("Physical Optimization candidate dependency references are malformed.")
        dependencies = {}
        for body_id, body_ref in dependency_refs.items():
            if (not isinstance(body_id, str) or not isinstance(body_ref, Mapping)
                    or body_ref.get("role") != "benchmark_value"):
                raise _integrity("Physical Optimization candidate dependency reference is malformed.",
                                 body_id=body_id)
            body_raw = snapshot.get_object(body_ref)
            body = record_document(body_raw)
            if (not isinstance(body, dict) or record_bytes(body) != body_raw
                    or sha256(body_raw).hexdigest() != body_id):
                raise _integrity("Physical Optimization candidate dependency body differs from its identity.",
                                 body_id=body_id)
            dependencies[body_id] = body
        value["dependencies"] = dependencies
        value["locator"] = {
            "kind": value.get("kind", "evaluation"),
            "stage_id": value.get("stage_id"),
            "evaluation_ordinal": value.get("evaluation_ordinal"),
            "candidate_ref": dict(candidate_ref),
            "source_task_id": source_task_id,
            "source_attempt_id": source_attempt_id,
            **({"source_batch_ref": dict(locator["source_batch_ref"])}
               if locator.get("source_batch_ref") is not None else {}),
        }
        return value

    def _stage_records(self, snapshot, root):
        from .validation.common import _integrity
        completion_sha = self.selection.completion_ref["sha256"]
        indexed = snapshot.read_optimization_stage_indexes(completion_sha)
        stages = root.get("stages")
        if len(indexed) != len(stages):
            raise _integrity("Physical Optimization stage SQL index count changed.")
        records = []
        for ordinal, (sealed, sql) in enumerate(zip(stages, indexed, strict=True)):
            stage_id = sealed.get("stage_id") if isinstance(sealed, Mapping) else None
            model = sealed.get("model_identity") if isinstance(sealed, Mapping) else None
            if (not isinstance(stage_id, str) or not isinstance(model, Mapping)
                    or sql.get("stage_ordinal") != ordinal or sql.get("stage_id") != stage_id
                    or sql.get("model_identity") != dict(model)
                    or sql.get("summary_ref") != sealed.get("summary_ref")
                    or sql.get("history_ref") != sealed.get("history_ref")):
                raise _integrity("Physical Optimization stage row differs from sealed SQL index.",
                                 stage_ordinal=ordinal)
            summary_body = self._canonical(snapshot, sealed["summary_ref"],
                                           "optimization_stage_summary",
                                           "scnsim.optimization_stage_summary")
            history_body = self._canonical(snapshot, sealed["history_ref"],
                                           "optimization_stage_history",
                                           "scnsim.optimization_stage_history")
            if (summary_body.get("stage_id") != stage_id
                    or summary_body.get("model_identity") != dict(model)
                    or history_body.get("stage_id") != stage_id
                    or history_body.get("model_identity") != dict(model)):
                raise _integrity("Physical Optimization stage bodies differ from sealed identities.",
                                 stage_id=stage_id)
            summary = summary_body.get("summary")
            if not isinstance(summary, Mapping):
                raise _integrity("Physical Optimization stage summary is malformed.", stage_id=stage_id)
            observed_ref = summary_body.get("observed_best_ref")
            accepted_ref = summary_body.get("accepted_iterate_ref")
            observed_locator = summary_body.get("observed_best_locator")
            accepted_locator = summary_body.get("accepted_iterate_locator")
            if ((observed_ref is None) != (observed_locator is None)
                    or (accepted_ref is None) != (accepted_locator is None)
                    or observed_ref is not None and (
                        not isinstance(observed_locator, Mapping)
                        or observed_locator.get("candidate_ref") != observed_ref)
                    or accepted_ref is not None and (
                        not isinstance(accepted_locator, Mapping)
                        or accepted_locator.get("candidate_ref") != accepted_ref)):
                raise _integrity("Physical Optimization stage candidate locators are malformed.",
                                 stage_id=stage_id)
            records.append({
                "stage_id": stage_id, "model_identity": dict(model), **dict(summary),
                "observed_best": None if observed_ref is None else self._candidate(
                    snapshot, observed_locator, model_identity=model, stage_id=stage_id, root=root),
                "accepted_iterate": None if accepted_ref is None else self._candidate(
                    snapshot, accepted_locator, model_identity=model, stage_id=stage_id, root=root),
            })
        return records

    def read_candidate(self, locator: Mapping[str, Any]) -> dict[str, Any]:
        from .validation.common import _integrity
        if not isinstance(locator, Mapping) or not isinstance(locator.get("candidate_ref"), Mapping):
            raise _integrity("Physical Optimization candidate locator is malformed.")
        with self._store().reader() as snapshot:
            if snapshot is None:
                raise _integrity("Selected Workspace database disappeared.")
            _, root = self._selection_snapshot(snapshot)
            candidate_ref = locator["candidate_ref"]
            permitted = False
            for stage in root.get("stages", ()):
                if stage.get("stage_id") != locator.get("stage_id"):
                    continue
                history = self._canonical(snapshot, stage["history_ref"],
                                          "optimization_stage_history",
                                          "scnsim.optimization_stage_history")
                refs = history.get("candidate_refs")
                sources = history.get("candidate_sources")
                if not isinstance(refs, list) or not isinstance(sources, list) or len(refs) != len(sources):
                    raise _integrity("Physical Optimization history source index is malformed.",
                                     stage_id=locator.get("stage_id"))
                for reference, source in zip(refs, sources, strict=True):
                    if (reference == dict(candidate_ref)
                            and isinstance(source, Mapping)
                            and source.get("task_id") == locator.get("source_task_id", self.selection.task_id)
                            and source.get("attempt_id") == locator.get("source_attempt_id", self.selection.attempt_id)
                            and source.get("batch_ref") == locator.get("source_batch_ref")):
                        permitted = True
                        break
                if permitted:
                    break
            if not permitted:
                comparison = self._canonical(snapshot, self.selection.comparison_ref,
                                             "optimization_comparison", "scnsim.optimization_comparison")
                for name in ("baseline_locator", "winner_locator"):
                    saved = comparison.get(name)
                    if isinstance(saved, Mapping) and dict(saved) == dict(locator):
                        permitted = True
                        break
            if not permitted:
                raise _integrity("Physical Optimization candidate is outside the sealed result index.")
            return self._candidate(snapshot, locator,
                                   stage_id=locator.get("stage_id"), root=root)

    def read_generation(self, index: int) -> dict[str, Any]:
        from .validation.common import _integrity
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise IndexError(index)
        with self._store().reader() as snapshot:
            if snapshot is None:
                raise _integrity("Selected Workspace database disappeared.")
            _, root = self._selection_snapshot(snapshot)
            batches = snapshot.read_optimization_stage_indexes(self.selection.completion_ref["sha256"])
            if index >= len(batches):
                raise IndexError(index)
            stage_id = batches[index]["stage_id"]
            return {"stage_id": stage_id, "rows": self._history_rows(snapshot, root, stage_id)}

    def read_candidate_discretization(self, flat_population_index: int):
        if not isinstance(flat_population_index, int) or isinstance(flat_population_index, bool) or flat_population_index < 0:
            raise IndexError(flat_population_index)
        offset = flat_population_index
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, root = self._selection_snapshot(snapshot)
            sealed_stages = root.get("stages", ())
            indexed_stages = snapshot.read_optimization_stage_indexes(
                self.selection.completion_ref["sha256"]
            )
            if len(indexed_stages) != len(sealed_stages):
                from .validation.common import _integrity
                raise _integrity("Physical Optimization stage index count differs from its root.")
            for ordinal, (stage, indexed_stage) in enumerate(
                    zip(sealed_stages, indexed_stages, strict=True)):
                if not isinstance(stage, Mapping):
                    from .validation.common import _integrity
                    raise _integrity("Physical Optimization stage locator is malformed.")
                stage_id = stage.get("stage_id")
                model_identity = stage.get("model_identity")
                if (indexed_stage.get("stage_ordinal") != ordinal
                        or indexed_stage.get("stage_id") != stage_id
                        or indexed_stage.get("model_identity") != model_identity
                        or indexed_stage.get("history_ref") != stage.get("history_ref")):
                    from .validation.common import _integrity
                    raise _integrity("Physical Optimization stage index differs from its root.",
                                     stage_ordinal=ordinal)
                history = self._canonical(
                    snapshot, stage.get("history_ref"), "optimization_stage_history",
                    "scnsim.optimization_stage_history",
                )
                references = history.get("candidate_refs")
                sources = history.get("candidate_sources")
                if (not isinstance(stage_id, str) or not isinstance(model_identity, Mapping)
                        or history.get("stage_id") != stage_id
                        or history.get("model_identity") != dict(model_identity)
                        or not isinstance(references, list)
                        or not isinstance(sources, list) or len(sources) != len(references)
                        or history.get("row_count") != len(references)):
                    from .validation.common import _integrity
                    raise _integrity("Physical Optimization history differs from its stage locator.",
                                     stage_id=stage_id)
                if offset < len(references):
                    indexed_row = snapshot.read_optimization_history_row(
                        self.selection.completion_ref["sha256"], stage_id, offset,
                    )
                    if (indexed_row is None or indexed_row.get("ordinal") != offset
                            or indexed_row.get("model_identity") != dict(model_identity)
                            or indexed_row.get("candidate_ref") != references[offset]):
                        from .validation.common import _integrity
                        raise _integrity("Physical Optimization selected candidate differs from its index.",
                                         stage_id=stage_id, ordinal=offset)
                    source = sources[offset]
                    if (not isinstance(source, Mapping)
                            or not isinstance(source.get("task_id"), str)
                            or not isinstance(source.get("attempt_id"), str)
                            or not isinstance(source.get("batch_ref"), Mapping)):
                        from .validation.common import _integrity
                        raise _integrity("Physical Optimization selected candidate source is malformed.",
                                         stage_id=stage_id, ordinal=offset)
                    locator = {"kind": "evaluation", "stage_id": stage_id,
                               "evaluation_ordinal": indexed_row.get("summary_json", {}).get("evaluation_ordinal"),
                               "candidate_ref": dict(references[offset]),
                               "source_task_id": source["task_id"],
                               "source_attempt_id": source["attempt_id"],
                               "source_batch_ref": dict(source["batch_ref"])}
                    return self._candidate(
                        snapshot, locator, model_identity=model_identity,
                        stage_id=stage_id, root=root,
                    ).get("discretization")
                offset -= len(references)
        raise IndexError(flat_population_index)

    def _history_rows(self, snapshot, root, stage_id):
        from .validation.common import _integrity
        stage = next((row for row in root.get("stages", ()) if row.get("stage_id") == stage_id), None)
        if stage is None:
            raise KeyError(stage_id)
        history = self._canonical(snapshot, stage["history_ref"],
                                  "optimization_stage_history", "scnsim.optimization_stage_history")
        refs = history.get("candidate_refs")
        sources = history.get("candidate_sources")
        indexed = snapshot.read_optimization_history(self.selection.completion_ref["sha256"], stage_id)
        if (not isinstance(refs, list) or not isinstance(sources, list)
                or len(sources) != len(refs) or len(indexed) != len(refs)
                or history.get("row_count") != len(refs)):
            raise _integrity("Physical Optimization history count differs from its seal.", stage_id=stage_id)
        rows = []
        model = stage["model_identity"]
        for ordinal, (reference, source, row) in enumerate(zip(refs, sources, indexed, strict=True)):
            summary = row.get("summary_json")
            if (row.get("ordinal") != ordinal or row.get("model_identity") != model
                    or row.get("candidate_ref") != reference or not isinstance(summary, Mapping)
                    or not isinstance(source, Mapping)
                    or not isinstance(source.get("task_id"), str)
                    or not isinstance(source.get("attempt_id"), str)
                    or not isinstance(source.get("batch_ref"), Mapping)):
                raise _integrity("Physical Optimization history row differs from its index.",
                                 stage_id=stage_id, ordinal=ordinal)
            body = {"stage_id": stage_id, "model_identity": dict(model), **dict(summary)}
            body["locator"] = {"kind": "evaluation", "stage_id": stage_id,
                               "evaluation_ordinal": summary.get("evaluation_ordinal"),
                               "candidate_ref": dict(reference),
                               "source_task_id": source["task_id"],
                               "source_attempt_id": source["attempt_id"],
                               "source_batch_ref": dict(source["batch_ref"])}
            rows.append(body)
        return rows

    def project(self, kind: str, selector=None):
        from .validation.common import _integrity
        if kind not in {"stages", "stage_history", "comparison"}:
            raise ValueError(f"unsupported physical Optimization projection: {kind}")
        with self._store().reader() as snapshot:
            if snapshot is None:
                raise _integrity("Selected Workspace database disappeared.")
            _, root = self._selection_snapshot(snapshot)
            if kind == "stages":
                return self._stage_records(snapshot, root)
            if kind == "stage_history":
                if not isinstance(selector, str):
                    raise KeyError(selector)
                return {"stage_id": selector,
                        "model_identity": next((row["model_identity"] for row in root["stages"]
                                                 if row.get("stage_id") == selector), None),
                        "rows": self._history_rows(snapshot, root, selector)}
            comparison = self._canonical(snapshot, self.selection.comparison_ref,
                                         "optimization_comparison", "scnsim.optimization_comparison")
            declarations = comparison.get("declarations")
            if not isinstance(declarations, Mapping):
                raise _integrity("Physical Optimization comparison declarations are malformed.")
            baseline_ref = comparison.get("baseline_ref")
            winner_ref = comparison.get("winner_ref")
            baseline_locator = comparison.get("baseline_locator")
            winner_locator = comparison.get("winner_locator")
            if ((baseline_ref is None) != (baseline_locator is None)
                    or (winner_ref is None) != (winner_locator is None)
                    or baseline_ref is not None and baseline_locator.get("candidate_ref") != baseline_ref
                    or winner_ref is not None and winner_locator.get("candidate_ref") != winner_ref):
                raise _integrity("Physical Optimization comparison candidate locators are malformed.")
            baseline = None if baseline_ref is None else self._candidate(
                snapshot, baseline_locator, root=root)
            winner = None if winner_ref is None else self._candidate(
                snapshot, winner_locator, root=root)
            return {
                "comparison_model_identity": comparison.get("comparison_model_identity"),
                "comparison_stage_id": comparison.get("comparison_stage_id"),
                "declarations": dict(declarations), "baseline": baseline, "winner": winner,
            }


class FixedResultReader:
    """Read one already-selected immutable result without retaining a DB handle."""

    def __init__(self, binding_identity: Mapping[str, Any], selection: ResultSelection):
        self._root = Path(binding_identity["root"])
        self._operations = Path(binding_identity["leaf"]) / "operations"
        self._plan_sha256 = binding_identity["plan_sha256"]
        self._workspace_instance_id = binding_identity["workspace_instance_id"]
        self.selection = selection

    def _store(self) -> OperationStore:
        return OperationStore(
            self._operations,
            plan_sha256=self._plan_sha256,
            workspace_instance_id=self._workspace_instance_id,
        )

    def _canonical(self, snapshot, reference, role, schema=None):
        if reference.get("role") != role:
            from .validation.common import _integrity
            raise _integrity("Sealed result reference has the wrong role.", expected=role,
                             actual=reference.get("role"))
        raw = snapshot.get_object(reference)
        value = record_document(raw)
        if (not isinstance(value, dict) or record_bytes(value) != raw
                or (schema is not None and value.get("schema") != schema)):
            from .validation.common import _integrity
            raise _integrity("Sealed result object is not canonical or has the wrong schema.",
                             reference=dict(reference), expected_schema=schema)
        return value

    def _selection_snapshot(self, snapshot):
        selection = self.selection
        completion = self._canonical(snapshot, selection.completion_ref,
                                     "workspace_completion", "scnsim.workspace_completion")
        if (selection.completion_ref.get("sha256") != selection.completion_sha256
                or completion.get("schema_version") != 1
                or completion.get("workspace_instance_id") != selection.workspace_instance_id
                or completion.get("plan_sha256") != selection.plan_sha256
                or completion.get("request_sha256") != selection.request_sha256
                or completion.get("task_id") != selection.task_id
                or completion.get("attempt_id") != selection.attempt_id
                or completion.get("terminal_attempt_ref") != dict(selection.terminal_attempt_ref)
                or completion.get("result_ref") != dict(selection.result_ref)
                or completion.get("index_root_ref") != dict(selection.index_root_ref)
                or completion.get("generation_ancestry_ref") != (
                    None if selection.generation_ancestry_ref is None
                    else dict(selection.generation_ancestry_ref))
                or completion.get("generation_count") != selection.generation_count
                or completion.get("candidate_count") != selection.candidate_count):
            from .validation.common import _integrity
            raise _integrity("Result selection differs from its sealed completion.",
                             task_id=selection.task_id, attempt_id=selection.attempt_id)
        attempt_pointer = snapshot.read_pointer(
            f"attempt/{selection.task_id}/{selection.attempt_id}"
        )
        if (selection.terminal_attempt_ref.get("sha256") != selection.attempt_sha256
                or attempt_pointer is None
                or attempt_pointer.get("reference") != dict(selection.terminal_attempt_ref)):
            from .validation.common import _integrity
            raise _integrity("Selected terminal attempt reference changed.",
                             task_id=selection.task_id, attempt_id=selection.attempt_id)
        index_root = self._canonical(snapshot, selection.index_root_ref,
                                     "workspace_completion_index_root",
                                     "scnsim.workspace_completion_index_root")
        if (selection.index_root_ref.get("sha256") != selection.index_root_sha256
                or index_root.get("schema_version") != 1
                or index_root.get("workspace_instance_id") != selection.workspace_instance_id
                or index_root.get("plan_sha256") != selection.plan_sha256
                or index_root.get("request_sha256") != selection.request_sha256
                or index_root.get("task_id") != selection.task_id
                or index_root.get("attempt_id") != selection.attempt_id
                or index_root.get("result_ref") != dict(selection.result_ref)
                or index_root.get("baseline_ref") != completion.get("baseline_ref")
                or index_root.get("generation_ancestry_ref") != completion.get("generation_ancestry_ref")
                or index_root.get("generation_count") != selection.generation_count
                or index_root.get("candidate_count") != selection.candidate_count
                or index_root.get("best_ordinal") != completion.get("best_ordinal")):
            from .validation.common import _integrity
            raise _integrity("Result index root differs from its sealed completion.",
                             task_id=selection.task_id, attempt_id=selection.attempt_id)
        return completion, index_root

    def read_generation(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.selection.generation_count
        if index < 0 or index >= self.selection.generation_count:
            raise IndexError(index)
        generation = index + 1
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, index_root = self._selection_snapshot(snapshot)
            sealed = next((row for row in index_root.get("generations", ())
                           if row.get("generation") == generation), None)
            indexed = snapshot.read_generation_index(self.selection.completion_sha256, generation)
            if (sealed is None or indexed is None
                    or any(indexed.get(key) != sealed.get(key) for key in
                           ("generation", "block_ref", "first_ordinal", "row_count", "summary_json"))):
                from .validation.common import _integrity
                raise _integrity("Generation locator differs from its sealed index.", generation=generation)
            candidate_index = self._canonical(
                snapshot, sealed["candidate_index_ref"], "workspace_candidate_index",
                "scnsim.workspace_candidate_index",
            )
            block = self._canonical(snapshot, sealed["block_ref"], "benchmark_generation_evidence",
                                    "scnsim.benchmark_generation_evidence")
            if (candidate_index.get("workspace_instance_id") != self.selection.workspace_instance_id
                    or candidate_index.get("plan_sha256") != self.selection.plan_sha256
                    or candidate_index.get("request_sha256") != self.selection.request_sha256
                    or candidate_index.get("task_id") != self.selection.task_id
                    or candidate_index.get("generation") != generation
                    or candidate_index.get("block_ref") != sealed.get("block_ref")
                    or candidate_index.get("row_count") != sealed.get("row_count")
                    or block.get("task_id") != self.selection.task_id
                    or block.get("attempt_id") != candidate_index.get("attempt_id")
                    or block.get("attributes", {}).get("generation") != generation
                    or len(block.get("rows", ())) != sealed.get("row_count")):
                from .validation.common import _integrity
                raise _integrity("Generation evidence differs from its sealed locator.", generation=generation)
            return block

    def read_candidate(self, evaluation_ordinal: int) -> dict[str, Any]:
        if evaluation_ordinal <= 0 or evaluation_ordinal > self.selection.candidate_count:
            raise IndexError(evaluation_ordinal)
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, index_root = self._selection_snapshot(snapshot)
            indexed = snapshot.read_candidate_index(self.selection.completion_sha256, evaluation_ordinal)
            generation = indexed.get("generation") if indexed is not None else None
            generation_row = next((row for row in index_root.get("generations", ())
                                   if row.get("generation") == generation), None)
            candidate_block = None
            if generation_row is not None:
                candidate_block = self._canonical(snapshot, generation_row["candidate_index_ref"],
                                                  "workspace_candidate_index",
                                                  "scnsim.workspace_candidate_index")
            sealed = None if candidate_block is None else next(
                (row for row in candidate_block.get("candidates", ())
                 if row.get("ordinal") == evaluation_ordinal), None)
            if indexed is None or sealed is None or indexed != sealed:
                from .validation.common import _integrity
                raise _integrity("Candidate locator differs from its sealed index.",
                                 evaluation_ordinal=evaluation_ordinal)
            block = self._canonical(snapshot, sealed["block_ref"], "benchmark_generation_evidence",
                                    "scnsim.benchmark_generation_evidence")
            if (candidate_block.get("workspace_instance_id") != self.selection.workspace_instance_id
                    or candidate_block.get("plan_sha256") != self.selection.plan_sha256
                    or candidate_block.get("request_sha256") != self.selection.request_sha256
                    or candidate_block.get("task_id") != self.selection.task_id
                    or candidate_block.get("generation") != sealed.get("generation")
                    or candidate_block.get("block_ref") != sealed.get("block_ref")
                    or candidate_block.get("attempt_id") != block.get("attempt_id")
                    or block.get("task_id") != self.selection.task_id
                    or block.get("attributes", {}).get("generation") != sealed.get("generation")):
                from .validation.common import _integrity
                raise _integrity("Candidate generation reference differs from its selected task.",
                                 evaluation_ordinal=evaluation_ordinal)
            offset = sealed["row_offset"]
            rows = block.get("rows")
            if not isinstance(offset, int) or not isinstance(rows, list) or offset < 0 or offset >= len(rows):
                from .validation.common import _integrity
                raise _integrity("Candidate row locator is outside its generation block.",
                                 evaluation_ordinal=evaluation_ordinal)
            occurrence = rows[offset]
            if occurrence.get("value") != sealed.get("value_ref"):
                from .validation.common import _integrity
                raise _integrity("Candidate value reference differs from its generation block.",
                                 evaluation_ordinal=evaluation_ordinal)
            raw = snapshot.get_object(sealed["value_ref"])
            value = record_document(raw)
            if not isinstance(value, dict) or record_bytes(value) != raw:
                from .validation.common import _integrity
                raise _integrity("Candidate value body is not canonical.",
                                 evaluation_ordinal=evaluation_ordinal)
            value.update(occurrence.get("occurrence", {}))
            return value

    def read_candidate_discretization(self, flat_population_index: int) -> dict[str, Any]:
        return self.read_candidate(flat_population_index + 1)["discretization"]

    def project(self, kind: str, selector=None):
        if kind not in {"history", "objective", "residual", "parameter", "table", "comparison"}:
            raise ValueError(f"unsupported optimization projection: {kind}")
        with self._store().reader() as snapshot:
            if snapshot is None:
                from .validation.common import _integrity
                raise _integrity("Selected Workspace database disappeared.")
            _, index_root = self._selection_snapshot(snapshot)
            if kind == "comparison":
                baseline_block = self._canonical(snapshot, self.selection.baseline_locator["block_ref"],
                                                 "benchmark_baseline_evidence",
                                                 "scnsim.benchmark_baseline_evidence")
                baseline = self._canonical(snapshot, baseline_block["baseline"], "benchmark_value")
                baseline.update(baseline_block.get("baseline_occurrence", {}))
                best = baseline if self.selection.best_locator.get("generation", 0) == 0 else None
                if best is None:
                    best_ordinal = self.selection.best_locator["ordinal"]
                    indexed = snapshot.read_candidate_index(self.selection.completion_sha256, best_ordinal)
                    generation_row = next((row for row in index_root.get("generations", ())
                                           if row.get("generation") == indexed.get("generation")), None)
                    candidate_block = self._canonical(snapshot, generation_row["candidate_index_ref"],
                                                      "workspace_candidate_index",
                                                      "scnsim.workspace_candidate_index")
                    sealed = next((row for row in candidate_block.get("candidates", ())
                                   if row.get("ordinal") == best_ordinal), None)
                    if sealed is None or sealed != indexed:
                        from .validation.common import _integrity
                        raise _integrity("Best-candidate locator differs from its sealed index.")
                    gen_block = self._canonical(snapshot, sealed["block_ref"],
                                                "benchmark_generation_evidence",
                                                "scnsim.benchmark_generation_evidence")
                    row = gen_block["rows"][sealed["row_offset"]]
                    if row.get("value") != sealed.get("value_ref"):
                        from .validation.common import _integrity
                        raise _integrity("Best-candidate body reference differs from its evidence row.")
                    raw = snapshot.get_object(sealed["value_ref"])
                    best = record_document(raw)
                    if record_bytes(best) != raw:
                        from .validation.common import _integrity
                        raise _integrity("Best-candidate body is not canonical.")
                    best.update(row.get("occurrence", {}))
                request = self._canonical(snapshot, index_root["request_ref"], "operation_request")
                return {"baseline": baseline, "best": best,
                        "declarations": request.get("spec", {})}
            indexed_rows = snapshot.read_candidate_indexes(self.selection.completion_sha256)
            sealed_by_generation = {}
            for generation_row in index_root.get("generations", ()):
                candidate_block = self._canonical(snapshot, generation_row["candidate_index_ref"],
                                                  "workspace_candidate_index",
                                                  "scnsim.workspace_candidate_index")
                if (candidate_block.get("generation") != generation_row.get("generation")
                        or candidate_block.get("block_ref") != generation_row.get("block_ref")
                        or candidate_block.get("row_count") != generation_row.get("row_count")):
                    from .validation.common import _integrity
                    raise _integrity("Candidate index block differs from the sealed generation root.",
                                     generation=generation_row.get("generation"))
                sealed_by_generation[generation_row["generation"]] = candidate_block.get("candidates", ())
            if len(indexed_rows) != self.selection.candidate_count:
                from .validation.common import _integrity
                raise _integrity("SQL candidate index count differs from its completion.")
            rows = []
            for indexed in indexed_rows:
                generation_candidates = sealed_by_generation.get(indexed["generation"], ())
                sealed = next((row for row in generation_candidates
                               if row.get("ordinal") == indexed["ordinal"]), None)
                if sealed != indexed:
                    from .validation.common import _integrity
                    raise _integrity("SQL candidate summary differs from the sealed candidate block.",
                                     evaluation_ordinal=indexed["ordinal"])
                rows.append(indexed["summary_json"])
        # The index was authenticated against its immutable root before these
        # compact rows were detached. Result-layer code owns unit interpretation.
        if kind == "history":
            return rows
        if kind == "parameter":
            return rows
        if kind in {"objective", "residual", "table"}:
            selected = []
            for row in rows:
                components = row.get("components", ())
                if selector is not None:
                    components = tuple(item for item in components
                                       if item.get("objective_id") == selector)
                selected.append({**row, "components": list(components)})
            return selected
        raise AssertionError(kind)
