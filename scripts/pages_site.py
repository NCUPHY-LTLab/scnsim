"""Prepare and verify source-bound, non-executing versioned Pages renders."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
import posixpath
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
from urllib.parse import unquote, urlsplit


SITE = "https://ncuphy-ltlab.github.io/scnsim/"
PREFIX = "/scnsim/"
REPO = "https://github.com/NCUPHY-LTLab/scnsim"
SHA = re.compile(r"[0-9a-f]{40}\Z")
RELEASE = re.compile(r"(?P<line>\d+\.\d+\.\d+)(?:(?P<alpha>a\d+)|(?P<beta>b\d+))?\Z")
DEVELOPMENT = re.compile(r"(?P<line>\d+\.\d+\.\d+) dev\Z")
DEVELOPMENT_SOURCE = re.compile(r"(?P<line>\d+\.\d+\.\d+)(?:a\d+|b\d+|\.dev\d+)?\Z")
# The 40 unmodified files in the selected Askr v0.12.1 vendor tree.
ASKR_SHA256 = "c2ead1a3539c004abdc6441ba6a65be65e535a7956c7e121ea1f9c33548a3c9d"


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name in ("href", "src"):
                self.paths.append(value)


def _regular_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Pages source/output is not a regular directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Pages tree contains an unsafe entry: {path}")


def _docs_source(path: Path) -> str:
    """Return the persistent documentation commit, never the product branch head."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"commit"}:
        raise ValueError("Documentation source pointer must contain only commit")
    commit = value["commit"]
    if not isinstance(commit, str) or not SHA.fullmatch(commit):
        raise ValueError("Documentation source commit must be a full lowercase SHA")
    return commit


def _catalog(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "development", "releases"} or value["schema"] != "scnsim.pages_versions.v1":
        raise ValueError("Pages version catalog is malformed")
    dev = value["development"]
    if not isinstance(dev, dict) or set(dev) != {"label", "root"} or not isinstance(dev["label"], str):
        raise ValueError("Pages development entry is malformed")
    match = DEVELOPMENT.fullmatch(dev["label"])
    if not match or dev["root"] != PREFIX + match["line"] + "-dev/":
        raise ValueError("Pages development label/root is invalid")
    releases = value["releases"]
    if not isinstance(releases, list) or not releases:
        raise ValueError("Pages catalog has no published version")
    seen: set[tuple[str, str]] = {(match["line"], "development")}
    for item in releases:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ValueError("Pages pinned version entry is malformed")
        pinned_dev = DEVELOPMENT.fullmatch(item["label"])
        published = RELEASE.fullmatch(item["label"])
        if pinned_dev:
            line, stage = pinned_dev["line"], "development"
            valid = set(item) == {"label", "root", "commit"} and item["root"] == PREFIX + line + "-dev/"
        elif published:
            line = published["line"]
            stage = "alpha" if published["alpha"] else "beta" if published["beta"] else "final"
            valid = set(item) == {"label", "root", "tag", "commit"} and item["root"] == PREFIX + item["label"] + "/" and item["tag"] == "v" + item["label"]
        else:
            raise ValueError("Pages pinned version label is invalid")
        if not valid or (line, stage) in seen or not isinstance(item["commit"], str) or not SHA.fullmatch(item["commit"]):
            raise ValueError("Pages pinned version line, tag, root, or commit is invalid")
        seen.add((line, stage))
    return value


def _entries(catalog: dict[str, object]) -> list[dict[str, str]]:
    dev = catalog["development"]
    releases = catalog["releases"]
    assert isinstance(dev, dict) and isinstance(releases, list)
    entries = [{"label": dev["label"], "root": dev["root"], "kind": "development"}]
    for row in releases:
        entry = {"label": row["label"], "root": row["root"], "kind": "release" if "tag" in row else "pinned-development", "commit": row["commit"]}
        if "tag" in row:
            entry["tag"] = row["tag"]
        entries.append(entry)
    return entries


def _slug(entry: dict[str, str]) -> str:
    return entry["root"].removeprefix(PREFIX).rstrip("/")


def _manifest(catalog: dict[str, object], current_slug: str, *, english: bool = False, bilingual: set[str] | None = None) -> bytes:
    versions = []
    for entry in _entries(catalog):
        # Askr resolves roots from this version directory, not the origin.
        slug = _slug(entry)
        if english:
            target = slug + ('/en' if bilingual and slug in bilingual else '')
            relative = posixpath.relpath(target, current_slug + '/en')
            root = './' if relative == '.' else relative + '/'
        else:
            root = "./" if slug == current_slug else f"../{slug}/"
        row: dict[str, object] = {"label": entry["label"], "root": root}
        versions.append(row)
    return (json.dumps({"mode": "inline", "versions": versions}, indent=2) + "\n").encode()


def _version(source: Path) -> str:
    value = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    version = value["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("Pages source has no package version")
    return version


def _vendor_digest(root: Path) -> str:
    _regular_tree(root)
    rows = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }
    if len(rows) != 40:
        raise ValueError("Pages Askr vendor does not have the reviewed file set")
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bind_language_hrefs(config: Path, version_root: str, current: str) -> None:
    """Bind the two native selector entries in an exported project config.

    Quarto profiles concatenate distinct list entries. Keep exactly one items
    authority per project instead of adding a second list through the profile.
    """
    lines = config.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip() == "askr:"]
    if len(starts) != 1:
        raise ValueError("Bilingual project must have one native Askr configuration")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and not lines[i][0].isspace()), len(lines))
    block = lines[start:end]
    if sum(line.rstrip() == "  languages:" for line in block) != 1:
        raise ValueError("Bilingual project must have one native language configuration")
    if sum(line.strip() == "current: " + current for line in block) != 1:
        raise ValueError("Bilingual project's current language differs from its role")
    entries = []
    for i, line in enumerate(block):
        match = re.fullmatch(r" {4,6}- code: (.+?)\s*", line)
        if match:
            entries.append((i, match[1]))
    if len(entries) != 2 or {code for _, code in entries} != {"en", "zh-TW"}:
        raise ValueError("Bilingual selector must contain exactly en and zh-TW")
    for ordinal, (index, code) in enumerate(entries):
        limit = entries[ordinal + 1][0] if ordinal + 1 < len(entries) else len(block)
        hrefs = [i for i in range(index + 1, limit) if re.match(r"^\s+href:", block[i])]
        if len(hrefs) != 1:
            raise ValueError("Native language selector entry must contain one href")
        index = hrefs[0]
        indentation = block[index][:len(block[index]) - len(block[index].lstrip())]
        href = version_root + ("en/" if code == "en" else "")
        block[index] = indentation + "href: " + json.dumps(href) + "\n"
    lines[start:end] = block
    config.write_text("".join(lines), encoding="utf-8")


def _prepare(source: Path, vendor: Path, catalog: dict[str, object], label: str, source_sha: str, presentation_sha: str) -> None:
    if not SHA.fullmatch(source_sha) or not SHA.fullmatch(presentation_sha):
        raise ValueError("Pages source/presentation commit is invalid")
    entry = next((item for item in _entries(catalog) if item["label"] == label), None)
    if entry is None or (entry["kind"] != "development" and source_sha != entry["commit"]):
        raise ValueError("Pages source does not match the pinned catalog entry")
    _regular_tree(source)
    if (source / ".git").exists() or not (source / "_quarto.yml").is_file() or not (source / "index.qmd").is_file():
        raise ValueError("Pages source must be a clean exported Quarto project")
    version = _version(source)
    if entry["kind"] == "release" and version != label:
        raise ValueError("Pages tag content version differs from the catalog")
    if entry["kind"] != "release":
        parsed_version = DEVELOPMENT_SOURCE.fullmatch(version)
        if not parsed_version or parsed_version["line"] != label.removesuffix(" dev"):
            raise ValueError("Pages development source has a different version line")
    profile = source / "_quarto-pages.yml"
    if profile.exists():
        raise ValueError("Pages render profile already exists in exported source")
    if _vendor_digest(vendor) != ASKR_SHA256:
        raise ValueError("Pages presentation vendor differs from Askr 0.12.1")
    destination = source / "_extensions/arfiligol/askr"
    if destination.exists():
        _regular_tree(destination)
    shutil.copytree(vendor, destination, dirs_exist_ok=True)
    if _vendor_digest(destination) != ASKR_SHA256:
        raise ValueError("Pages presentation overlay is not exact Askr 0.12.1")
    version_url = SITE + _slug(entry) + "/"
    bilingual = (source / 'zh-TW/_quarto.yml').is_file()
    site_url = version_url + ('en/' if bilingual else '')
    profile.write_text(
        "website:\n"
        f"  site-url: {json.dumps(site_url)}\n"
        f"  repo-url: {json.dumps(REPO)}\n"
        f"  repo-branch: {json.dumps(source_sha)}\n"
        "  page-footer:\n"
        f"    center: {json.dumps(label + ' · package ' + version + ' · content [' + source_sha + '](' + REPO + '/tree/' + source_sha + ') · publisher [' + presentation_sha + '](' + REPO + '/tree/' + presentation_sha + ')')}\n",
        encoding="utf-8",
    )
    if entry["kind"] == "development":
        with profile.open("a", encoding="utf-8") as handle:
            handle.write(
                "  announcement:\n"
                "    icon: info-circle\n"
                "    dismissable: false\n"
                "    position: above-navbar\n"
                "    type: warning\n"
                f"    content: {json.dumps('R2 is under development and has not been officially released.')}\n"
            )
    if bilingual:
        # Current bilingual source only; pinned historical projects stay as-is.
        _bind_language_hrefs(source / '_quarto.yml', entry['root'], 'en')
        _bind_language_hrefs(source / 'zh-TW/_quarto.yml', entry['root'], 'zh-TW')
        child_profile = source / 'zh-TW/_quarto-pages.yml'
        child_profile.write_text(
            profile.read_text(encoding='utf-8').replace(
                f"  site-url: {json.dumps(site_url)}\n",
                f"  site-url: {json.dumps(version_url)}\n").replace(
                    json.dumps('R2 is under development and has not been officially released.'),
                    json.dumps('R2 開發中，尚未正式發佈。', ensure_ascii=False)), encoding='utf-8')
        # English-only links are explicit /en/ routes in authored Chinese prose.
        # Bind just those Markdown/YAML/HTML hrefs in the exported source, not
        # code includes, scientific assets or historical project sources.
        link = re.compile(r'(\]\(|href:\s*|href=["\'])/en/')
        child = source / 'zh-TW'
        for path in child.rglob('*'):
            relative = path.relative_to(child)
            if not path.is_file() or path.suffix not in ('.qmd', '.md', '.yml'):
                continue
            if any(part in ('_extensions', '_generated', 'figures', 'position-figures') for part in relative.parts):
                continue
            if path.name == '_quarto-pages.yml':
                continue
            text = path.read_text(encoding='utf-8')
            path.write_text(link.sub(lambda match: match[1] + version_url + 'en/', text), encoding='utf-8')
    print(f"prepared {label} package={version} content={source_sha} presentation={presentation_sha}")


def _local_reference(site: Path, page: Path, value: str, root: str) -> Path | None:
    parsed = urlsplit(value)
    if value.startswith("#"):
        return None
    if parsed.scheme or parsed.netloc:
        # Quarto preserves full URLs between separately rendered language
        # projects. Still validate same-site destinations against the artifact.
        if not value.startswith(SITE):
            return None
    path = unquote(parsed.path)
    if not path:
        return None
    if path.startswith(root):
        return site / path.removeprefix(root)
    if path.startswith("/"):
        raise ValueError(f"cross-site absolute reference in {root}: {value}")
    target = (page.parent / path).resolve()
    try:
        target.relative_to(site.resolve())
    except ValueError:
        raise ValueError(f"local reference escapes the version directory: {value}") from None
    return target


def _inspect(source: Path, entry: dict[str, str], source_sha: str, presentation_sha: str) -> Path:
    site = source / "_site"
    site_url = SITE + _slug(entry) + "/"
    bilingual = (source / "zh-TW/_quarto.yml").is_file()
    profile_url = site_url + ("en/" if bilingual else "")
    profile = source / "_quarto-pages.yml"
    if not profile.is_file() or f"  site-url: {json.dumps(profile_url)}\n" not in profile.read_text(encoding="utf-8"):
        raise ValueError(f"{entry['label']} render profile has the wrong version root")
    _regular_tree(site)
    index = site / "index.html"
    search = site / "search.json"
    if not index.is_file() or not search.is_file():
        raise ValueError(f"{entry['label']} render lacks home or local search")
    json.loads(search.read_text(encoding="utf-8"))
    html = index.read_text(encoding="utf-8")
    for required in (entry["label"], _version(source), source_sha, presentation_sha, REPO):
        if required not in html:
            raise ValueError(f"{entry['label']} home lacks source/presentation provenance: {required}")
    if entry["kind"] == "development":
        notice = "R2 開發中，尚未正式發佈。" if bilingual else "R2 is under development and has not been officially released."
        if notice not in html:
            raise ValueError("Current documentation home lacks its unreleased R2 announcement")
    if f"/{source_sha}/" not in html:
        raise ValueError(f"{entry['label']} View source does not target the content commit")
    image = site / "docs/assets/readme-hero-orca-penguin.png"
    if (source / "docs/assets/readme-hero-orca-penguin.png").is_file() and not image.is_file():
        raise ValueError(f"{entry['label']} render lacks its README hero image")
    notebook = site / "examples/engineer/chapter-01/chapter.ipynb"
    if (source / "examples/engineer/chapter-01/chapter.ipynb").is_file() and not notebook.is_file():
        raise ValueError(f"{entry['label']} render lacks its linked Chapter notebook")
    for page in (index, site / "docs/index.html"):
        if not page.is_file():
            raise ValueError(f"{entry['label']} render lacks navigation page: {page}")
        refs = References()
        refs.feed(page.read_text(encoding="utf-8"))
        for value in refs.paths:
            target = _local_reference(site, page, value, entry["root"])
            if target is not None and not target.is_file() and not target.is_dir():
                raise ValueError(f"{entry['label']} render has a broken home/course asset: {value}")
    if bilingual:
        if not (site / 'en/index.html').is_file() or not (site / 'en/search.json').is_file():
            raise ValueError('Bilingual render lacks English home or per-language search')
        json.loads((site / 'en/search.json').read_text(encoding='utf-8'))
        child_profile = source / 'zh-TW/_quarto-pages.yml'
        if not child_profile.is_file() or f"  site-url: {json.dumps(site_url)}\n" not in child_profile.read_text(encoding='utf-8'):
            raise ValueError('Chinese Pages profile has the wrong version root')
        # Include both project homes and course maps in the existing local-link guard.
        for page in (site / 'en/index.html', site / 'en/docs/index.html'):
            if not page.is_file():
                raise ValueError(f'English navigation page missing: {page}')
            refs = References()
            refs.feed(page.read_text(encoding='utf-8'))
            for value in refs.paths:
                target = _local_reference(site, page, value, entry['root'])
                if target is not None and not target.is_file() and not target.is_dir():
                    raise ValueError(f'English render has a broken home/course asset: {value}')
    math_page = site / ("en" if bilingual else "") / "docs/implementation/diagonal-root-numerical-procedure.html"
    if not math_page.is_file() or "mathjax" not in math_page.read_text(encoding="utf-8").lower():
        raise ValueError(f"{entry['label']} render lacks declared MathJax support")
    return site


def _missing_page(catalog: dict[str, object]) -> str:
    entries = _entries(catalog)
    routes = {"prefix": PREFIX, "development": entries[0], "releases": entries[1:]}
    data = json.dumps(routes, separators=(",", ":")).replace("<", "\\u003c")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>SCNSim documentation path</title></head><body>\n'
        '<main><h1 id="heading">Page not found</h1>'
        '<p id="message">This documentation page is unavailable.</p>'
        f'<p><a id="destination" href="{escape(entries[0]["root"])}">Open development documentation</a></p></main>\n'
        '<script type="application/json" id="routes">' + data + '</script>\n'
        r"""<script>
(function () {
  var routes = JSON.parse(document.getElementById("routes").textContent);
  var path = location.pathname, dev = routes.development.root;
  var heading = document.getElementById("heading");
  var message = document.getElementById("message");
  var link = document.getElementById("destination");
  function explain(title, body, href, label) {
    heading.textContent = title; message.textContent = body;
    link.href = href; link.textContent = label;
  }
  function inRoot(root) { return path === root.slice(0, -1) || path.indexOf(root) === 0; }
  var oldDev = routes.prefix + "develop/";
  if (inRoot(oldDev)) {
    var rest = path === oldDev.slice(0, -1) ? "" : path.slice(oldDev.length);
    location.replace(dev + rest + location.search + location.hash);
    return;
  }
  var oldMain = routes.prefix + "main/";
  if (inRoot(oldMain)) {
    explain("The former main docs URL has moved", "Main was a branch view, not an Alpha release. Choose a current version rather than treating it as Alpha content.", dev, "Open development documentation");
    return;
  }
  var current = [routes.development].concat(routes.releases).filter(function (item) { return inRoot(item.root); })[0];
  if (current) {
    location.replace(current.root + location.search + location.hash);
    return;
  }
  var alpha = path.slice(routes.prefix.length).match(/^(\d+\.\d+\.\d+)a\d+(?:\/|$)/);
  if (path.indexOf(routes.prefix) === 0 && alpha) {
    var replacement = routes.releases.filter(function (item) {
      return item.label.match(/^\d+\.\d+\.\d+/)[0] === alpha[1] && /a\d+$/.test(item.label);
    })[0];
    explain("This Alpha documentation is retired", "The requested Alpha is not published at this URL. Its content has not been replaced under the old version name.", replacement ? replacement.root : dev, replacement ? "Open the current " + alpha[1] + " documentation" : "Open development documentation");
  }
})();
</script></body></html>
"""
    )


def _assemble(catalog: dict[str, object], template: Path, sources: Path, docs_source_sha: str, presentation_sha: str, output: Path) -> None:
    if output.exists() or not SHA.fullmatch(docs_source_sha) or not SHA.fullmatch(presentation_sha):
        raise ValueError("Pages output or commit identity is invalid")
    entries = _entries(catalog)
    sites: dict[str, Path] = {}
    for entry in entries:
        source_sha = docs_source_sha if entry["kind"] == "development" else entry["commit"]
        sites[_slug(entry)] = _inspect(sources / _slug(entry), entry, source_sha, presentation_sha)
    root_html = template.read_text(encoding="utf-8")
    root_html = root_html.replace("{{DEV_SLUG}}", escape(_slug(entries[0]))).replace("{{DEV_LABEL}}", escape(entries[0]["label"]))
    if "{{" in root_html:
        raise ValueError("Pages root redirect has an unresolved token")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="scnsim-pages-", dir=output.parent))
    try:
        bilingual = {slug for slug, site in sites.items() if (site / 'en/index.html').is_file()}
        for slug, site in sites.items():
            shutil.copytree(site, temporary / slug)
            (temporary / slug / "askr-versions.json").write_bytes(_manifest(catalog, slug))
            if slug in bilingual:
                (temporary / slug / 'en/askr-versions.json').write_bytes(
                    _manifest(catalog, slug, english=True, bilingual=bilingual))
        (temporary / "index.html").write_text(root_html, encoding="utf-8")
        not_found = _missing_page(catalog)
        (temporary / "404.html").write_text(not_found, encoding="utf-8")
        for legacy in ("develop", "main"):
            (temporary / legacy).mkdir()
            (temporary / legacy / "index.html").write_text(not_found, encoding="utf-8")
        _regular_tree(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print("assembled " + ", ".join(f"{item['label']}={_slug(item)}" for item in entries))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source")
    source.add_argument("--pointer", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--catalog", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--vendor", type=Path, required=True)
    prepare.add_argument("--label", required=True)
    prepare.add_argument("--source-sha", required=True)
    prepare.add_argument("--publication-sha", "--presentation-sha", dest="publication_sha", required=True)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--catalog", type=Path, required=True)
    assemble.add_argument("--template", type=Path, required=True)
    assemble.add_argument("--sources", type=Path, required=True)
    assemble.add_argument("--docs-source-sha", "--develop-sha", dest="docs_source_sha", required=True,
                          help="Explicit documentation commit; --develop-sha is a legacy argument name only")
    assemble.add_argument("--publication-sha", "--presentation-sha", dest="publication_sha", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source":
        print(_docs_source(args.pointer))
        return
    catalog = _catalog(args.catalog)
    if args.command == "plan":
        for item in _entries(catalog):
            print("|".join((item["kind"], item["label"], _slug(item), item.get("tag", "-"), item.get("commit", "-"))))
    elif args.command == "prepare":
        _prepare(args.source, args.vendor, catalog, args.label, args.source_sha, args.publication_sha)
    else:
        _assemble(catalog, args.template, args.sources, args.docs_source_sha, args.publication_sha, args.output)


if __name__ == "__main__":
    main()
