#!/usr/bin/env python3
"""Maintain committed agent reference Markdown from an installed SCNSim.

This command alone imports the producer, resolves exports and constructs the
explicit public cases. Packaging only copies its outputs. No solver, Run,
rendering, native physics, arbitrary factory evaluation or documentation render
belongs here. Canonical code/docstrings and case JSON remain editable authority.
"""
from __future__ import annotations

import argparse
from enum import Enum
import hashlib
from importlib import metadata
import inspect
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import re
import traceback

GENERATOR = 'scripts/generate_agent_catalogue.py'
CASES = 'docs/agent-catalogue-cases.json'
INVENTORY = 'docs/agent-catalogue-resources.json'
OUTPUT = 'docs/reference/agent-catalogue'
CURATED = 'docs/agent-usage-inventory.json'
COMMON_SOURCES = (
    'src/scnsim/__init__.py', 'src/scnsim/authoring/catalogs.py',
    'src/scnsim/authoring/components.py', 'src/scnsim/authoring/assembly.py',
    'src/scnsim/authoring/physical_values.py', 'src/scnsim/authoring/parameters.py',
    'src/scnsim/authoring/handles.py', 'src/scnsim/authoring/provenance.py',
    'src/scnsim/authoring/snapshot.py', 'src/scnsim/authoring/factory_context.py',
)


class SourceBindingError(RuntimeError):
    """Installed contributor bytes do not match the canonical maintenance input."""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)


def qualified(value):
    return value.__module__ + '.' + value.__qualname__


def slug(name):
    """Catalogue names follow the actual public symbol, without word expansion."""
    return name.casefold()


def error_record(error):
    return dict(type=qualified(type(error)), message=str(error), traceback=traceback.format_exc())


def default_record(value, package):
    """Describe actual defaults without address/owner repr or hidden token values."""
    if value is inspect.Parameter.empty:
        return {'kind': 'required'}
    if isinstance(value, Enum):
        return {'kind': 'enum', 'type': qualified(type(value)), 'member': value.name}
    if value is None or isinstance(value, (str, bool, int)):
        return {'kind': 'literal', 'value': value}
    if isinstance(value, float):
        return {'kind': 'float', 'hex': value.hex()}
    if isinstance(value, package.units.registry.Quantity):
        magnitude = value.magnitude
        if hasattr(magnitude, 'tolist'):
            magnitude = magnitude.tolist()
        return {'kind': 'quantity', 'magnitude': magnitude, 'unit': str(value.units)}
    if isinstance(value, (tuple, list)):
        return {'kind': type(value).__name__, 'items': [default_record(item, package) for item in value]}
    if inspect.isclass(value):
        return {'kind': 'class', 'name': qualified(value)}
    return {'kind': 'opaque_default', 'type': qualified(type(value)),
            'description': 'Actual default type only; no mutable owner, sentinel value or repr serialized.'}


def annotation_record(value):
    if value is inspect.Signature.empty:
        return None
    if isinstance(value, str):
        return value
    if inspect.isclass(value):
        return qualified(value)
    return str(value)


def signature_record(value, package):
    if not callable(value):
        return {'state': 'not_callable'}
    try:
        signature = inspect.signature(value, eval_str=False)
    except (ValueError, TypeError) as error:
        return {'state': 'unavailable', 'error': error_record(error)}
    return {'state': 'available', 'parameters': [
        dict(name=parameter.name, kind=parameter.kind.name,
             annotation=annotation_record(parameter.annotation),
             default=default_record(parameter.default, package))
        for parameter in signature.parameters.values()],
        'return_annotation': annotation_record(signature.return_annotation)}


def public_record(value):
    """Keep errors explicit in public pages; raw messages/tracebacks stay private."""
    if isinstance(value, dict):
        return {key: ({'type': item['type'], 'detail': 'Actual error retained in the private maintenance observation.'}
                      if key == 'error' else public_record(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [public_record(item) for item in value]
    return value


def source_for(value, package_root, root):
    try:
        location = inspect.getsourcefile(inspect.unwrap(value))
    except TypeError:
        location = getattr(value, '__file__', None) if inspect.ismodule(value) else None
    if location is None:
        return None, None
    module = getattr(inspect.unwrap(value), '__module__', None)
    if (location.startswith('<frozen ') and location.endswith('>')
            and module is not None and module != 'scnsim' and not module.startswith('scnsim.')):
        # CPython exposes inherited stdlib code without an on-disk source file.
        # Preserve that real origin; producer sources still use strict resolution.
        return None, dict(kind='frozen_external_module', module=module, origin=location)
    path = Path(location).resolve(strict=True)
    try:
        relative = path.relative_to(package_root)
    except ValueError:
        return None, None  # An inherited stdlib descriptor has no producer source file.
    source = 'src/scnsim/' + relative.as_posix()
    if path.read_bytes() != (root / source).read_bytes():
        raise SourceBindingError('installed producer source differs from canonical contributor: ' + source)
    return source, None


def members_record(value, package, package_root, root):
    if not inspect.isclass(value):
        return [], set()
    members, sources = [], set()
    for name, descriptor in inspect.getmembers_static(value):
        if name.startswith('_'):
            continue
        if isinstance(descriptor, property):
            members.append(dict(name=name, kind='property', doc=inspect.getdoc(descriptor)))
            target = descriptor.fget
        elif isinstance(descriptor, (staticmethod, classmethod)) or inspect.isroutine(descriptor):
            target = getattr(value, name)
            members.append(dict(name=name, kind='method', signature=signature_record(target, package),
                                doc=inspect.getdoc(target)))
        else:
            continue
        source, external_origin = source_for(target, package_root, root)
        if external_origin is not None:
            members[-1]['source_origin'] = external_origin
        if source is not None:
            sources.add(source)
    return members, sources


def construct_value(value, package):
    """Closed public case vocabulary, not eval/exec or a producer adapter."""
    if not isinstance(value, dict):
        return value
    if set(value) == {'quantity'}:
        record = value['quantity']
        return package.units.registry.Quantity(record['magnitude'], record['unit'])
    if set(value) == {'rlgc'}:
        record = {key: construct_value(item, package) for key, item in value['rlgc'].items()}
        record['conductors'] = tuple(record['conductors'])
        return package.RLGC(**record)
    if set(value) == {'parameter'}:
        record = value['parameter']
        definitions = package.ParameterDefinitions(id=record['definitions_id'])
        return definitions.parameter(id=record['id'], baseline=construct_value(record['baseline'], package),
                                     spec=package.ParameterSpec(unit=package.units.registry.Unit(record['unit'])))
    raise TypeError('unsupported declared construction input')


def observe_case(case, package):
    try:
        kwargs = {key: construct_value(value, package) for key, value in case['kwargs'].items()}
        component = getattr(package.components, case['factory'])(**kwargs)
        if not isinstance(component, package.ComponentInstance):
            raise TypeError('public factory did not return ComponentInstance')
        pins = []
        for name in component.pins:
            if component.factory == 'transmission_line':
                edge, conductor = name.split('.', 1)
                pin = component.pin(edge, conductor=conductor)
            else:
                pin = component.pin(name)
            pins.append(dict(name=name, public_type=qualified(type(pin)), component_id=pin.component_id))
        observed = dict(id=component.id, factory=component.factory, kind=component.kind,
                        catalog_id=component.catalog_id, pins=pins,
                        parameters=[dict(name=name, definitions_id=ref.definitions_id, parameter_id=ref.id)
                                    for name, ref in component.parameters.items()],
                        coordinates=list(component.coordinates), branches=list(component.branches))
        # Whole-wheel RECORD provenance is observed privately, never embedded in
        # generated Markdown (which would make its own build hash recursive).
        return dict(case_id=case['id'], state='constructed', observed=observed,
                    private_catalog_source=dict(component.catalog_source))
    except Exception as error:
        return dict(case_id=case['id'], state='construction_error', error=error_record(error))


def link(source, target):
    return os.path.relpath(target, PurePosixPath(source).parent.as_posix()).replace(os.sep, '/')


def provenance(root, paths):
    return '\n'.join('- `' + path + '` — SHA-256 `' + digest(root / path) + '`'
                     for path in sorted(set(paths)))


def page(title, body, root, contributors):
    return ('<!-- Generated by scripts/generate_agent_catalogue.py; do not hand-edit. -->\n'
            '# ' + title + '\n\n' + body + '\n\n## Source provenance\n\n'
            + provenance(root, [GENERATOR, CASES, *contributors]) + '\n')


def resource(identifier, kind, title, summary, source):
    return dict(id=identifier, kind=kind, title=title, summary=summary,
                status='Generated current public reference; construction observations are limited to declared inputs.',
                path='resources/catalogue/' + source.removeprefix(OUTPUT + '/'), source_ref=source)


def document_inventory(root):
    """Read this project's explicit project.render list, without a YAML engine."""
    paths = {'README.md'}
    active = False
    for line in (root / '_quarto.yml').read_text(encoding='utf-8').splitlines():
        if line == '  render:':
            active = True
            continue
        if active and line and not line.startswith('    '):
            break
        if active and line.strip():
            match = re.fullmatch(r'    - ([A-Za-z0-9_./-]+\.(?:qmd|md))', line)
            if match is None:
                raise ValueError('project.render no longer uses the explicit public source list')
            value = match.group(1)
            if PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
                raise ValueError('unsafe public source inventory path')
            paths.add(value)
    if not active:
        raise ValueError('explicit project.render source inventory is missing')
    return sorted(paths)


def generate(root, observations):
    # Intentionally deferred: ordinary build hooks must not import this command.
    import scnsim as package

    package_root = Path(package.__file__).resolve(strict=True).parent
    if package_root.is_relative_to(root):
        raise RuntimeError('maintenance requires the isolated installed runtime, not a source-tree fallback')
    distribution = metadata.distribution('scnsim')
    direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
    if direct_url.get('dir_info', {}).get('editable') or distribution.read_text('RECORD') is None:
        raise RuntimeError('maintenance requires noneditable installed metadata and RECORD')
    for source in COMMON_SOURCES:
        installed = package_root / source.removeprefix('src/scnsim/')
        if installed.read_bytes() != (root / source).read_bytes():
            raise RuntimeError('installed runtime contributor mismatch: ' + source)
    names = tuple(package.__all__)
    if len(set(names)) != len(names):
        raise ValueError('actual public exports contain duplicate names')
    factories = {name: value for name, value in inspect.getmembers(package.components, predicate=callable)
                 if not name.startswith('_')}
    cases = json.loads((root / CASES).read_text(encoding='utf-8'))['factory_cases']
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('declared construction case ids must be unique')
    unknown = set(case['factory'] for case in cases) - set(factories)
    if unknown:
        raise ValueError('declared cases name absent public factories: ' + ', '.join(sorted(unknown)))
    observations.update(runtime_version=distribution.version, runtime_module=str(package_root),
                        public_names=sorted(names), factory_names=sorted(factories), exports={}, cases={})
    pages, resources, api_links, factory_links = {}, [], [], []
    for name in sorted(names):
        identifier = 'catalogue/api/' + slug(name)
        source = OUTPUT + '/api/' + slug(name) + '.md'
        contributors = {'src/scnsim/__init__.py'}
        try:
            value = getattr(package, name)
            definition, external_origin = source_for(value, package_root, root)
            if definition is not None:
                contributors.add(definition)
            members, member_sources = members_record(value, package, package_root, root)
            contributors.update(member_sources)
            record = dict(state='resolved', public_type=qualified(type(value)),
                          signature=signature_record(value, package), doc=inspect.getdoc(value), members=members)
            if external_origin is not None:
                record['source_origin'] = external_origin
        except SourceBindingError:
            raise
        except Exception as error:
            record = dict(state='resolution_error', error=error_record(error))
        observations['exports'][name] = record
        title = name + ' public API'
        summary = 'Public ' + name + ' declaration, signature, members and responsibility.'
        body = ('Public name: `scnsim.' + name + '`.\n\n'
                'This page records the declared API; it does not construct this class or execute its methods. '
                'Missing descriptions/signatures and actual resolution errors are explicit below.\n\n'
                '```json\n' + canonical(public_record(record)) + '\n```')
        authority = None
        if name in {'ParameterDefinitions', 'ParameterRef', 'ParameterSet', 'ParameterSpace', 'ParameterSpec', 'RLGCParameterSpec', 'AffineMap', 'ScalarRLGCMap'}:
            authority = 'docs/contracts/parameter-system.qmd'
        elif name in {'CircuitPlan', 'CompositePlan', 'SubsystemPlan'}:
            authority = 'docs/contracts/circuit-plan-assembly.qmd'
        elif name in {'Library', 'ComponentInstance', 'RLGC', 'ElectricalResolution'}:
            authority = 'docs/component-authoring.qmd'
        elif name in {'ReductionPipeline', 'NetworkViewRef'}:
            authority = 'docs/concepts/compilation-coordinates-and-network-views.qmd'
        elif name in {'StaticCapacitanceSpec', 'StaticCapacitanceElementSpec'}:
            authority = 'docs/guides/static-capacitance.qmd'
        elif name in {'SensitivitySpec', 'Jacobian', 'JVP', 'VJP', 'OutputCoordinate', 'SensitivityScales'}:
            authority = 'docs/guides/sensitivity.qmd'
        elif name == 'units':
            authority = 'docs/concepts/simulation-conventions.qmd'
        if authority is not None:
            body += '\n\n[Canonical usage and semantic authority](' + link(source, authority) + ').'
            contributors.add(authority)
        if name == 'RLGC':
            line_case = next(case for case in cases if case['factory'] == 'transmission_line')
            body += ('\n\n## Declared ordered RLGC construction input\n\n'
                     'This input is actually constructed only within the declared line case; '
                     'it is not evidence for other conductor bases or matrix values.\n\n```json\n'
                     + canonical(line_case['kwargs']['rlgc']) + '\n```')
        pages[source] = page(title, body, root, contributors)
        resources.append(resource(identifier, 'api', title, summary, source))
        api_links.append('- [' + name + '](' + link(OUTPUT + '/index.md', source) + ') — `' + identifier + '`')
    for name, factory in sorted(factories.items()):
        identifier, source = 'catalogue/components/' + name, OUTPUT + '/components/' + name + '.md'
        selected = [case for case in cases if case['factory'] == name]
        rows = []
        for case in selected:
            result = observe_case(case, package)
            observations['cases'][case['id']] = result
            public_result = {key: value for key, value in result.items() if key != 'private_catalog_source'}
            rows.append(dict(inputs=case, observation=public_record(public_result)))
        title = 'Josephson junction' if name == 'josephson_junction' else name.replace('_', ' ')
        title += ' component factory'
        signature = signature_record(factory, package)
        observations.setdefault('factory_signatures', {})[name] = signature
        body = ('Public callable: `scnsim.components.' + name + '`. Returns a factory-owned `ComponentInstance`.\n\n'
                '## Signature\n\n```json\n' + canonical(public_record(signature)) + '\n```\n\n'
                '## Description\n\n' + (inspect.getdoc(factory) or 'The canonical factory has no docstring; no description is inferred.')
                + '\n\n## Declared construction observations\n\n'
                + ('```json\n' + canonical(rows) + '\n```' if rows else 'No construction case is declared for this actual public name.')
                + '\n\nOnly these inputs were constructed. Pin names are not proof of independent electrical nodes. '
                'The source-bound [component authoring contract](' + link(source, 'docs/component-authoring.qmd')
                + ') owns topology, units and parameter rules. No solver, nonlinear response or rendering is observed.')
        pages[source] = page(title, body, root, [*COMMON_SOURCES, 'docs/component-authoring.qmd'])
        resources.append(resource(identifier, 'component', title, 'Construct ' + name + ' with its declared inputs and inspect actual public pins and branches.', source))
        factory_links.append('- [' + name + '](' + link(OUTPUT + '/index.md', source) + ') — `' + identifier + '`')
    index_source = OUTPUT + '/index.md'
    pages[index_source] = page('SCNSim public catalogue',
        'Names come from actual `scnsim.__all__` and the builtin `components` Library, including lazy exports. '
        'Each detail records its own source/signature; factory observations apply only to declared cases.\n\n'
        '## Component factories\n\n' + '\n'.join(factory_links) + '\n\n## Public API names\n\n' + '\n'.join(api_links),
        root, COMMON_SOURCES)
    resources.append(resource('catalogue/index', 'reference', 'SCNSim public catalogue',
                              'Browse every declared public export and builtin component factory.', index_source))
    known_source = OUTPUT + '/known-public-documents.md'
    curated = json.loads((root / CURATED).read_text(encoding='utf-8'))
    bundled = {row['source_ref'].split('#', 1)[0] for row in [*curated['resources'], *resources]}
    missing = [source for source in document_inventory(root) if source not in bundled]
    body = ('These public source paths are known from `_quarto.yml` project.render plus README, '
            'and are not bundled in this knowledge index. A local link may therefore be unresolved in installed knowledge; '
            'that is not evidence the canonical document is missing. No website alias is inferred.\n\n'
            + '\n'.join('- [`' + source + '`](' + link(known_source, source) + ')' for source in missing))
    pages[known_source] = page('Known public documents outside this bundle', body, root, ['_quarto.yml', CURATED])
    resources.append(resource('catalogue/known-public-documents', 'reference', 'Known public documents outside this bundle',
                              'Source-inventory list of public documents not packaged in installed knowledge.', known_source))
    if len({row['id'] for row in resources}) != len(resources):
        raise ValueError('generated stable IDs collide')
    observations.update(public_count=len(names), factory_count=len(factories), known_unbundled_sources=missing,
                        generator_sha256=digest(root / GENERATOR), cases_sha256=digest(root / CASES))
    pages[INVENTORY] = canonical(dict(distribution='scnsim', generator=GENERATOR,
                                    resources=sorted(resources, key=lambda row: row['id']))) + '\n'
    return {path: text.encode('utf-8') for path, text in pages.items()}


def reproject(root, observations, *, previous_root, previous_observations, expected_sha256):
    """Reproject bound committed pages; never import/construct the producer.

    Runtime observations keep their original version and origin. All contributing
    code/case/document bytes must still match before their observed facts can be
    reused. Only API naming, generated links and this generator's provenance vary.
    """
    if digest(previous_observations) != expected_sha256:
        raise SourceBindingError('previous runtime observation hash mismatch')
    prior = json.loads(previous_observations.read_text(encoding='utf-8'))
    if digest(previous_root / GENERATOR) != prior['generator_sha256']:
        raise SourceBindingError('previous generator/observation mismatch')
    if digest(previous_root / CASES) != prior['cases_sha256'] or digest(root / CASES) != prior['cases_sha256']:
        raise SourceBindingError('construction cases changed since the runtime observation')
    bound = prior['generated_files']
    for path, expected in bound.items():
        if digest(previous_root / path) != expected:
            raise SourceBindingError('previous committed output/observation mismatch: ' + path)
    previous_inventory = json.loads((previous_root / INVENTORY).read_text(encoding='utf-8'))
    rows = previous_inventory['resources']
    names = prior['public_names']
    by_title = {row['title']: row for row in rows if row['id'].startswith('catalogue/api/')}
    if set(by_title) != {name + ' public API' for name in names}:
        raise SourceBindingError('previous inventory does not cover the observed public names')
    path_map, id_map, replacements = {}, {}, {}
    for name in names:
        old = by_title[name + ' public API']
        source = OUTPUT + '/api/' + slug(name) + '.md'
        identifier = 'catalogue/api/' + slug(name)
        path_map[old['source_ref']] = source
        id_map[old['id']] = identifier
        replacements[old['id']] = dict(old, id=identifier, source_ref=source,
            path='resources/catalogue/api/' + slug(name) + '.md')
    resources = [replacements.get(row['id'], row) for row in rows]
    if len({row['id'] for row in resources}) != len(resources):
        raise ValueError('reprojected catalogue IDs collide')
    pages, contributor_hashes = {}, {}
    for old_source in sorted(set(bound) - {INVENTORY}):
        text = (previous_root / old_source).read_text(encoding='utf-8')
        body, provenance_section = text.rsplit('\n## Source provenance\n\n', 1)
        provenance_lines = provenance_section.splitlines()
        for line in provenance_lines:
            match = re.fullmatch(r'- `([^`]+)` — SHA-256 `([0-9a-f]{64})`', line)
            if match is None:
                raise ValueError('unexpected committed provenance line in ' + old_source)
            path, expected = match.groups()
            if digest(previous_root / path) != expected:
                raise SourceBindingError('previous contributor hash mismatch: ' + path)
            if path != GENERATOR and digest(root / path) != expected:
                raise SourceBindingError('canonical contributor changed since observation: ' + path)
            contributor_hashes[path] = digest(root / path)
        source = path_map.get(old_source, old_source)
        def rewrite_link(match):
            target, separator, anchor = match.group(2).partition('#')
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(old_source), target))
            if resolved not in path_map:
                return match.group(0)
            return match.group(1) + link(source, path_map[resolved]) + (separator + anchor if separator else '') + match.group(3)
        body = re.sub(r'(\]\()([^\n)]+)(\))', rewrite_link, body)
        for old_id, identifier in id_map.items():
            body = body.replace('`' + old_id + '`', '`' + identifier + '`')
        current_lines = []
        for line in provenance_lines:
            if line.startswith('- `' + GENERATOR + '` —'):
                line = '- `' + GENERATOR + '` — SHA-256 `' + digest(root / GENERATOR) + '`'
            current_lines.append(line)
        pages[source] = body + '\n## Source provenance\n\n' + '\n'.join(current_lines) + '\n'
    pages[INVENTORY] = canonical(dict(distribution='scnsim', generator=GENERATOR,
        resources=sorted(resources, key=lambda row: row['id']))) + '\n'
    observations.update(collection='REUSED_ORIGINAL_RUNTIME_OBSERVATION_NO_IMPORT_OR_CONSTRUCTION',
        reused_observation=dict(sha256=expected_sha256, runtime_version=prior['runtime_version'],
            runtime_module=prior['runtime_module'], previous_generator_sha256=prior['generator_sha256']),
        runtime_version=prior['runtime_version'], runtime_module=prior['runtime_module'],
        public_names=prior['public_names'], factory_names=prior['factory_names'],
        public_count=prior['public_count'], factory_count=prior['factory_count'],
        exports=prior['exports'], cases=prior['cases'], factory_signatures=prior['factory_signatures'],
        generator_sha256=digest(root / GENERATOR), cases_sha256=digest(root / CASES),
        verified_contributor_hashes=contributor_hashes, renamed_paths=path_map,
        known_unbundled_sources=prior['known_unbundled_sources'])
    return {path: text.encode('utf-8') for path, text in pages.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True,
                        help='New private JSON receipt outside the repository; never bundled')
    parser.add_argument('--reproject-from', type=Path, help='Immutable prior canonical source snapshot; no producer import')
    parser.add_argument('--reuse-observations', type=Path, help='Original private runtime observation to reuse')
    parser.add_argument('--reuse-observations-sha256', help='Exact authorized original observation SHA-256')
    args = parser.parse_args()
    reuse = (args.reproject_from, args.reuse_observations, args.reuse_observations_sha256)
    if any(value is not None for value in reuse) and not all(value is not None for value in reuse):
        parser.error('offline reprojection requires all three explicit reuse inputs')
    root = args.source_root.resolve(strict=True)
    observation_path = args.observations.resolve()
    if observation_path.is_relative_to(root):
        raise ValueError('maintenance observations must remain outside public source')
    if observation_path.exists():
        raise FileExistsError(observation_path)
    observations = {'state': 'started', 'mode': 'write' if args.write else 'check'}
    try:
        if args.reproject_from is None:
            files = generate(root, observations)
        else:
            files = reproject(root, observations,
                previous_root=args.reproject_from.resolve(strict=True),
                previous_observations=args.reuse_observations.resolve(strict=True),
                expected_sha256=args.reuse_observations_sha256)
        existing = {path.relative_to(root).as_posix() for path in (root / OUTPUT).rglob('*.md')}
        expected = set(files) - {INVENTORY}
        drift = [path for path, data in files.items() if not (root / path).is_file() or (root / path).read_bytes() != data]
        stale = sorted(existing - expected)
        if args.write:
            for path, data in sorted(files.items()):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            for path in stale:
                (root / path).unlink()  # Only obsolete generated Markdown in our owned folder.
        observations.update(state='written' if args.write else 'compared',
                            generated_files={path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
                            changed_paths=sorted(drift), stale_generated_paths=stale)
        failures = ([name for name, row in observations['exports'].items() if row['state'] == 'resolution_error']
                    + [name for name, row in observations['cases'].items() if row['state'] == 'construction_error'])
        observations['actual_error_names'] = failures
        if failures:
            return 2  # All names/errors remain emitted, but no successful runtime comparison is claimed.
        return 1 if args.check and (drift or stale) else 0
    except Exception as error:
        observations.update(state='raised', error=error_record(error))
        raise
    finally:
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_path.write_text(canonical(observations) + '\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
