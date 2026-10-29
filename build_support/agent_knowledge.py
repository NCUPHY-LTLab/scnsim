"""Generate exact public knowledge copies for the existing setuptools backend.

The curated inventory and committed maintenance-generated catalogue index define
the bundle. Version comes from pyproject.toml; provenance binds whole canonical
UTF-8 files even when the README payload is an exact focused slice. Builds copy
committed catalogue Markdown, never import the producer or run maintenance.
This folder is sdist input,
not an installed SCNSim runtime dependency or a second documentation authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import tomllib


_CATALOGUE_INVENTORY = 'docs/agent-catalogue-resources.json'
_START_REF = 'README.md#agent-knowledge-start'
_START_BEGIN = b'<!-- agent-knowledge-start:begin -->\n'
_START_END = b'<!-- agent-knowledge-start:end -->'

_RESOURCE_FIELDS = {'id', 'kind', 'title', 'summary', 'status', 'path', 'source_ref'}
_ID = re.compile(r'[a-z0-9][a-z0-9._/-]*\Z')
_SEGMENT = re.compile(r'[A-Za-z0-9._-]+\Z')
_DEVICES = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)),
            *(f'lpt{i}' for i in range(1, 10))}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def _parts(value):
    if not isinstance(value, str) or not value:
        raise ValueError('knowledge paths must be nonempty relative POSIX strings')
    parts = value.split('/')
    if any(not _SEGMENT.fullmatch(part) or part in {'.', '..'} or part.endswith('.')
           or part.split('.')[0].casefold() in _DEVICES for part in parts):
        raise ValueError(f'unsafe knowledge path: {value!r}')
    return parts


def _regular(path):
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 0x400:
        raise ValueError(f'knowledge input/output must not be a link or reparse point: {path}')
    return metadata


def _source_bytes(root, source_ref):
    path = root
    for part in _parts(source_ref):
        path = path / part
        _regular(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'knowledge source must be a regular file: {source_ref}')
    data = path.read_bytes()
    data.decode('utf-8', errors='strict')
    return data


def _resource_bytes(root, source_ref):
    source_path, separator, _fragment = source_ref.partition('#')
    data = _source_bytes(root, source_path)
    if not separator:
        return data, data
    if source_ref != _START_REF:
        raise ValueError('only the declared README start supports section extraction')
    if data.count(_START_BEGIN) != 1 or data.count(_START_END) != 1:
        raise ValueError('README start requires exactly one pair of extraction markers')
    begin = data.index(_START_BEGIN) + len(_START_BEGIN)
    end = data.index(_START_END)
    if end <= begin:
        raise ValueError('README start extraction markers are out of order')
    return data, data[begin:end]


def _inventory_rows(root):
    inventory = json.loads(_source_bytes(root, 'docs/agent-usage-inventory.json'))
    if not isinstance(inventory, dict) or set(inventory) != {
            'distribution', 'resources', 'catalogue_inventory'}:
        raise ValueError('knowledge inventory requires distribution, resources and catalogue_inventory')
    if (inventory['distribution'] != 'scnsim' or
            not isinstance(inventory['resources'], list) or
            inventory['catalogue_inventory'] != _CATALOGUE_INVENTORY):
        raise ValueError('knowledge inventory must describe scnsim and its declared catalogue')
    catalogue = json.loads(_source_bytes(root, _CATALOGUE_INVENTORY))
    if (not isinstance(catalogue, dict) or
            set(catalogue) != {'distribution', 'generator', 'resources'} or
            catalogue['distribution'] != 'scnsim' or
            catalogue['generator'] != 'scripts/generate_agent_catalogue.py' or
            not isinstance(catalogue['resources'], list)):
        raise ValueError('generated catalogue inventory disagrees with its producer schema')
    return inventory['resources'] + catalogue['resources']


def generate(root):
    """Regenerate this producer's declared bundle; unchanged bytes stay untouched.

    Both build_py and sdist call this before setuptools discovers package data.
    All inputs are validated/read before generated output is changed. The output
    folder belongs exclusively to this generator; stale copies are removed so
    superseded documentation cannot enter a wheel alongside the current index.
    """
    root = Path(root).resolve(strict=True)
    rows = _inventory_rows(root)
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    if project['name'] != 'scnsim':
        raise ValueError('knowledge generator requires the scnsim project')
    version = project['version']
    if not isinstance(version, str) or not version:
        raise ValueError('knowledge distribution version must be explicit')
    resources, payloads = [], {}
    ids, folded_paths = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _RESOURCE_FIELDS:
            raise ValueError('knowledge inventory resource fields disagree with the producer schema')
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError('knowledge inventory resource values must be nonempty strings')
        identifier = row['id']
        if (not _ID.fullmatch(identifier) or
                any(part in {'', '.', '..'} for part in identifier.split('/')) or identifier in ids):
            raise ValueError(f'invalid or duplicate knowledge resource id: {identifier!r}')
        ids.add(identifier)
        parts = _parts(row['path'])
        if parts[-1].casefold() == 'manifest.json' or row['path'].casefold() in folded_paths:
            raise ValueError(f'reserved or colliding knowledge resource path: {row["path"]!r}')
        folded_paths.add(row['path'].casefold())
        source_data, data = _resource_bytes(root, row['source_ref'])
        payloads[row['path']] = data
        resources.append({key: row[key] for key in ('id', 'kind', 'title', 'summary', 'status', 'path')} |
                         {'content_sha256': hashlib.sha256(data).hexdigest(),
                          'provenance': {'source_ref': row['source_ref'],
                                         'source_sha256': hashlib.sha256(source_data).hexdigest()}})
    starts = [row for row in resources if row['kind'] == 'start']
    if len(starts) != 1 or starts[0]['id'] != 'getting-started' or starts[0]['provenance']['source_ref'] != _START_REF:
        raise ValueError('SCNSim requires exactly one README start resource: getting-started')
    for path in folded_paths:
        if any('/'.join(path.split('/')[:index]) in folded_paths
               for index in range(1, len(path.split('/')))):
            raise ValueError('knowledge resource file/directory paths collide')
    manifest = {'schema_version': '1.0', 'distribution': 'scnsim',
                'distribution_version': version, 'resources': sorted(resources, key=lambda row: row['id'])}
    manifest['bundle_id'] = 'sha256:' + hashlib.sha256(_canonical(manifest)).hexdigest()
    payloads['manifest.json'] = _canonical(manifest)
    destination = root / 'src/scnsim/_agent_knowledge'
    if destination.exists() or destination.is_symlink():
        _regular(destination)
        for path in destination.rglob('*'):
            metadata = _regular(path)
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
                raise ValueError(f'nonregular generated knowledge entry: {path}')
    destination.mkdir(parents=True, exist_ok=True)
    # First remove stale files and empty directories in this owned output only.
    for path in sorted(destination.rglob('*'), key=lambda path: len(path.parts), reverse=True):
        if path.is_file() and path.relative_to(destination).as_posix() not in payloads:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    for name, data in sorted(payloads.items()):
        path = destination.joinpath(*_parts(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
    return manifest
