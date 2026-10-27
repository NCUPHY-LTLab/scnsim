"""Check source includes/relative links and actual rendered website root URLs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


def prose_only(text):
    """Retain the CI check's existing exclusion of fenced code examples."""
    lines = []
    fence_char = None
    fence_length = 0
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence_char is None:
            if marker:
                fence_char = marker.group(1)[0]
                fence_length = len(marker.group(1))
            else:
                lines.append(line)
        elif (
            marker
            and marker.group(1)[0] == fence_char
            and len(marker.group(1)) >= fence_length
            and not marker.group(2).strip()
        ):
            fence_char = None
    return "\n".join(lines)


def check_links(root: Path, sources, *, site_root: Path = Path("_site")) -> int:
    """Check tracked prose using source contexts and the rendered web root.

    Quarto's English output is _site/en and its Chinese output is _site.
    Root URLs name files beneath this shared website root; an existing source
    document never substitutes for a missing rendered page or download.
    Relative-source and include checks retain their previous CI behavior.
    """
    root = root.resolve()
    site_root = (root / site_root).resolve()
    sources = [root / source for source in sources]
    include_contexts = {}
    for source in sources:
        for included in re.findall(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}", prose_only(source.read_text())):
            path = (source.parent / included.strip("\"'")).resolve()
            assert path.exists(), f"broken local include in {source}: {included}"
            include_contexts.setdefault(path, set()).add(source.parent)

    for source in sources:
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", prose_only(source.read_text())):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            relative = target.split("#", 1)[0].strip("<>")
            if not relative:
                continue
            for context in include_contexts.get(source.resolve(), {source.parent}):
                if relative.startswith("/"):
                    path = site_root / relative.lstrip("/")
                    if relative.endswith("/"):
                        path = path / "index.html"
                    path = path.resolve()
                    assert path.is_relative_to(site_root) and path.is_file(), (
                        f"broken rendered local link in {source}: {target} (expected {path})"
                    )
                else:
                    path = (context / relative).resolve()
                    assert path.exists(), f"broken local link in {source}: {target}"
    return len(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--site-root", type=Path, default=Path("_site"))
    args = parser.parse_args()
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.md", "*.qmd"], cwd=args.root, text=True,
    ).splitlines()
    count = check_links(args.root, tracked, site_root=args.site_root)
    print("verified local links in", count, "tracked Markdown/QMD files")


if __name__ == "__main__":
    main()
