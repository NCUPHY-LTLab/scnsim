"""Stage shared bilingual documentation inputs; preserve English code authority.

Chinese prose lives in zh-TW with mirrored paths. Generated code includes,
extension files and scientific assets are derived from the original project.
No notebook execution or translation happens here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)\r?\n?$")
GENERATED = Path('_generated/canonical-code')


def fenced_blocks(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    blocks = []
    index = 0
    while index < len(lines):
        opening = FENCE.fullmatch(lines[index])
        if not opening:
            index += 1
            continue
        marker = opening[1]
        end = index + 1
        closing = re.compile(r'^ {0,3}' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}\s*$')
        while end < len(lines) and not closing.fullmatch(lines[end]):
            end += 1
        if end == len(lines):
            raise ValueError('Unclosed Markdown code fence')
        blocks.append((index, end + 1, ''.join(lines[index:end + 1])))
        index = end + 1
    return blocks


def shared_assets(root: Path) -> None:
    child = root / 'zh-TW'
    # All directories are selected public source inputs, never Workspace outputs.
    for relative in ('_extensions/arfiligol/askr', 'tests/figures', 'docs/assets',
                     'examples/engineer/figures', 'examples/engineer/chapter-07/position-figures'):
        source = root / relative
        if not source.is_dir():
            raise FileNotFoundError(f'Required shared documentation directory: {relative}')
        shutil.copytree(source, child / relative, dirs_exist_ok=True)
    # Existing README badge/license destinations are shared public files.
    for name in ('LICENSE', 'NOTICE', 'pyproject.toml'):
        shutil.copyfile(root / name, child / name)
    schema = child / 'src/scnsim/_schemas'
    schema.mkdir(parents=True, exist_ok=True)
    for name in ('identity-common.schema.json', 'identity-v2.schema.json', 'identity-v3.schema.json'):
        shutil.copyfile(root / 'src/scnsim/_schemas' / name, schema / name)
    for source in (root / 'examples/engineer').rglob('*.ipynb'):
        destination = child / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def translated_sources(child: Path):
    for path in sorted(child.rglob('*')):
        relative = path.relative_to(child)
        if not path.is_file() or path.suffix not in ('.qmd', '.md'):
            continue
        if any(part in ('_extensions', '_generated', 'figures', 'position-figures') for part in relative.parts):
            continue
        yield path, relative


def prepare(root: Path) -> None:
    child = root / 'zh-TW'
    if not (child / '_quarto.yml').is_file():
        raise FileNotFoundError('Chinese Quarto project is missing')
    shared_assets(root)
    code_manifest = []
    for translation, relative in translated_sources(child):
        if relative.parts[0] == 'tests':
            continue  # Testing's static scientific tables are authored prose inputs.
        canonical = root / relative
        if not canonical.is_file():
            raise FileNotFoundError(f'Canonical English source missing: {relative}')
        blocks = fenced_blocks(canonical.read_text(encoding='utf-8'))
        if fenced_blocks(translation.read_text(encoding='utf-8')):
            raise ValueError(f'Use canonical code includes, not editable duplicate fences: {relative}')
        for ordinal, (_, _, code) in enumerate(blocks, 1):
            target = child / GENERATED / relative / f'block-{ordinal:03}.qmd'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding='utf-8')
            code_manifest.append({'source': relative.as_posix(), 'ordinal': ordinal,
                                  'sha256': hashlib.sha256(code.encode()).hexdigest(),
                                  'include': target.relative_to(child).as_posix()})
    generated = child / '_generated'
    generated.mkdir(exist_ok=True)
    (generated / 'canonical-code.json').write_text(json.dumps(code_manifest, indent=2) + '\n', encoding='utf-8')


def materialize(root: Path, translations: Path) -> None:
    """Compose reviewed translated prose once; verify and replace copied fences.

    This is an explicit source-composition command, not a render-time rewrite of
    authored prose. The delivered zh-TW postimages contain include references.
    """
    child = root / 'zh-TW'
    if translations.resolve() == child.resolve():
        raise ValueError('Translation input must remain separate from composed output')
    for source, relative in translated_sources(translations):
        text = source.read_text(encoding='utf-8')
        if relative.parts[0] != 'tests':
            canonical = root / relative
            expected = fenced_blocks(canonical.read_text(encoding='utf-8'))
            actual = fenced_blocks(text)
            if [x[2] for x in actual] != [x[2] for x in expected]:
                raise ValueError(f'Translated code differs from canonical English source: {relative}')
            lines = text.splitlines(keepends=True)
            for ordinal in range(len(actual), 0, -1):
                start, end, _ = actual[ordinal - 1]
                include = GENERATED / relative / f'block-{ordinal:03}.qmd'
                href = posixpath.relpath(include.as_posix(), relative.parent.as_posix())
                lines[start:end] = ['{{< include ' + href + ' >}}\n']
            text = ''.join(lines)
        destination = child / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding='utf-8')
    prepare(root)


def render_pages(project: Path) -> list[str]:
    """Read the explicit render list emitted by the maintained Quarto config."""
    lines = (project / '_quarto.yml').read_text(encoding='utf-8').splitlines()
    result = []
    active = False
    for line in lines:
        if line == '  render:':
            active = True
        elif active:
            item = re.fullmatch(r' {2,4}- (.+)', line)
            if not item:
                break
            value = item[1]
            if not value or value.startswith(('/', '!')) or '..' in Path(value).parts:
                raise ValueError('Expected explicit relative render paths')
            result.append(value)
    if not result:
        raise ValueError('Quarto project has no explicit render list')
    return result


def post_render(root: Path) -> None:
    # Follow Askr 0.12's whole-project hook; per-page renders never recurse.
    if os.environ.get('QUARTO_PROJECT_RENDER_ALL') != '1' or os.environ.get('SCNSIM_BILINGUAL_SEQUENTIAL') == '1':
        return
    command = ['quarto', 'render', str(root / 'zh-TW'), '--no-clean', '--no-execute']
    # Quarto exposes CLI-activated profiles through this documented variable.
    # Pass it explicitly so a Pages child cannot silently use local routes.
    profile = os.environ.get('QUARTO_PROFILE')
    if profile:
        if 'pages' in profile.split(',') and not (root / 'zh-TW/_quarto-pages.yml').is_file():
            raise FileNotFoundError('Chinese Pages profile is missing')
        command += ['--profile', profile]
    subprocess.run(command, cwd=root, check=True)


def sequential(root: Path, profile: str | None) -> None:
    """Render each configured page once with one foreground Quarto process."""
    prepare(root)
    profile = profile or os.environ.get('QUARTO_PROFILE')
    environment = os.environ.copy()
    environment['SCNSIM_BILINGUAL_SEQUENTIAL'] = '1'
    environment['SCNSIM_BILINGUAL_PREPARED'] = '1'
    for project in (root, root / 'zh-TW'):
        if profile and 'pages' in profile.split(',') and not (project / '_quarto-pages.yml').is_file():
            raise FileNotFoundError(f'Pages profile is missing: {project}')
        for relative in render_pages(project):
            if not (project / relative).is_file():
                raise FileNotFoundError(project / relative)
            command = ['quarto', 'render', relative, '--no-clean', '--no-execute']
            if profile:
                command += ['--profile', profile]
            subprocess.run(command, cwd=project, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'materialize', 'post-render', 'sequential'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--translations', type=Path)
    parser.add_argument('--profile')
    args = parser.parse_args()
    if args.command == 'prepare':
        if os.environ.get('SCNSIM_BILINGUAL_PREPARED') != '1':
            prepare(args.root)
    elif args.command == 'materialize':
        if args.translations is None:
            parser.error('materialize requires --translations')
        materialize(args.root, args.translations)
    elif args.command == 'post-render':
        post_render(args.root)
    else:
        sequential(args.root, args.profile)


if __name__ == '__main__':
    main()
