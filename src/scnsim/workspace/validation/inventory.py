"""Receipt and artifact inventory verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ...canonical import sha256_hex as _sha256
from ..artifacts import _verify_manifest_tree
from ..primitives import _inside
from ..storage import _decode_bytes
from .common import _IDENTIFIER, _integrity, _valid_sha
from .sweeps import _verify_parameter_sweep_artifacts

def _compare_artifacts(left: object, right: object, *, operation: object) -> None:
    if not isinstance(left, list) or not isinstance(right, list):
        raise _integrity("Outcome and receipt require artifact inventories.")
    normalized: list[list[tuple[object, ...]]] = []
    for inventory in (left, right):
        entries: list[tuple[object, ...]] = []
        identities: set[tuple[object, ...]] = set()
        paths: set[str] = set()
        for entry in inventory:
            if not isinstance(entry, dict):
                raise _integrity("Artifact inventory entry is malformed.")
            if set(entry) == {"id", "path", "sha256", "media_type", "byte_length"}:
                if (
                    not isinstance(entry.get("id"), str)
                    or not entry["id"]
                    or not isinstance(entry.get("path"), str)
                    or not entry["path"]
                    or not isinstance(entry.get("media_type"), str)
                    or not entry["media_type"]
                    or not isinstance(entry.get("byte_length"), int)
                    or isinstance(entry.get("byte_length"), bool)
                    or entry["byte_length"] < 1
                ):
                    raise _integrity("Artifact inventory entry is malformed.")
                identity = (entry["id"], entry["path"])
                if identity in identities or entry["path"] in paths:
                    raise _integrity("Artifact inventory contains a duplicate identity or path.")
                identities.add(identity)
                paths.add(entry["path"])
                entries.append((
                    *identity,
                    _valid_sha(entry.get("sha256")),
                    entry["media_type"],
                    entry["byte_length"],
                ))
            else:
                if set(entry) != {"id", "sha256"} or not isinstance(entry.get("id"), str) or not entry["id"]:
                    raise _integrity("Artifact inventory entry is malformed.")
                identity = (entry["id"],)
                if identity in identities:
                    raise _integrity("Artifact inventory contains a duplicate ID.", artifact_id=entry["id"])
                identities.add(identity)
                entries.append((*identity, _valid_sha(entry.get("sha256"))))
        normalized.append(entries)
    if normalized[0] != normalized[1]:
        raise _integrity("Outcome and receipt artifact inventories disagree.")


def _verify_artifact_inventory(
    directory: Path,
    result: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    request_path: Path,
    include_sweep_records: bool = False,
    workspace_native_index: bool = False,
    defer_native_optimization_ledgers: bool = False,
) -> list[dict[str, object]] | None:
    if result.get("result_kind") == "parameter_sweep":
        return _verify_parameter_sweep_artifacts(
            directory,
            result,
            receipt,
            request_path=request_path,
            include_records=include_sweep_records,
        )
    catalog = result.get("array_catalog")
    if catalog is None:
        catalog = {}
    if not isinstance(catalog, dict):
        raise _integrity("Result has no typed array catalog.")
    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, list):
        raise _integrity("Receipt has no artifact inventory.")
    declared: set[tuple[str, str]] = set()
    declared_ids: set[str] = set()
    for entry in receipt_artifacts:
        if not isinstance(entry, dict):
            raise _integrity("Receipt artifact inventory entry is malformed.")
        identifier = entry.get("id")
        digest = entry.get("sha256")
        if not isinstance(identifier, str) or not identifier or identifier in declared_ids:
            raise _integrity("Receipt artifact inventory has a duplicate or invalid ID.")
        declared_ids.add(identifier)
        declared.add((identifier, _valid_sha(digest)))
    resolved: set[tuple[object, object]] = set()
    resolved_ids: set[str] = set()
    resolved_paths: set[str] = set()
    for artifact in catalog.values():
        if not isinstance(artifact, dict):
            raise _integrity("Result array catalog entry is malformed.")
        identifier = artifact.get("id")
        digest = artifact.get("sha256")
        path = artifact.get("path")
        manifest = artifact.get("file_manifest")
        if not isinstance(identifier, str) or not isinstance(digest, str) or not isinstance(path, str) or not isinstance(manifest, str):
            raise _integrity("Result array catalog lacks required artifact identity.")
        if identifier in resolved_ids or path in resolved_paths or manifest in resolved_paths:
            raise _integrity("Result artifact catalog contains duplicate IDs or paths.")
        resolved_ids.add(identifier)
        resolved_paths.update({path, manifest})
        pair = (identifier, _valid_sha(digest))
        resolved.add(pair)
        artifact_path = _inside(directory, path)
        manifest_path = _inside(directory, manifest)
        if not artifact_path.is_dir() or artifact_path.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
            raise _integrity("Result artifact path is missing or unsafe.", artifact_id=identifier)
        manifest_bytes = manifest_path.read_bytes()
        if _sha256(manifest_bytes) != digest:
            raise _integrity("Artifact manifest hash disagrees with result catalog.", artifact_id=identifier)
        manifest_doc = _decode_bytes(manifest_bytes, "artifact manifest")
        if (
            manifest_doc.get("schema") != "scnsim.artifact_manifest"
            or manifest_doc.get("artifact_id") != identifier
            or manifest_doc.get("artifact_path") != path
        ):
            raise _integrity("Artifact manifest identity disagrees with result catalog.", artifact_id=identifier)
        _verify_manifest_tree(artifact_path, manifest_doc)
    ledgers = result.get("ledger_artifacts", [])
    if not isinstance(ledgers, list):
        raise _integrity("Optimization Result ledger catalog is malformed.")
    for artifact in ledgers:
        if not isinstance(artifact, dict):
            raise _integrity("Optimization ledger catalog entry is malformed.")
        identifier = artifact.get("id")
        digest = artifact.get("sha256")
        path = artifact.get("path")
        length = artifact.get("byte_length")
        if not isinstance(identifier, str) or not isinstance(digest, str) or not isinstance(path, str) or not isinstance(length, int):
            raise _integrity("Optimization ledger lacks its file artifact identity.")
        if identifier in resolved_ids or path in resolved_paths:
            raise _integrity("Result artifact catalog contains duplicate IDs or paths.")
        resolved_ids.add(identifier)
        resolved_paths.add(path)
        file_path = _inside(directory, path)
        if not file_path.is_file() or file_path.is_symlink() or file_path.stat().st_size != length:
            raise _integrity("Optimization ledger artifact path is missing or unsafe.", artifact_id=identifier)
        if not defer_native_optimization_ledgers and _sha256(file_path.read_bytes()) != _valid_sha(digest):
            raise _integrity("Optimization ledger hash disagrees with its result catalog.", artifact_id=identifier)
        resolved.add((identifier, digest))
    if declared != resolved:
        raise _integrity("Receipt artifact inventory does not exactly match Result catalog.")
    artifact_root = directory / "artifacts"
    if artifact_root.is_symlink():
        raise _integrity("Result artifact directory is unsafe.")
    expected_top = {
        "/".join(path.split("/")[:2])
        for path in resolved_paths
    }
    if workspace_native_index:
        expected_top.add("artifacts/native-index")
    if artifact_root.exists():
        if not artifact_root.is_dir():
            raise _integrity("Result artifact directory is unsafe.")
        actual_top = {
            child.relative_to(directory).as_posix()
            for child in artifact_root.iterdir()
        }
        if actual_top != expected_top:
            raise _integrity("Result artifact directory contains undeclared entries.")
    elif expected_top:
        raise _integrity("Result artifact directory is missing.")
