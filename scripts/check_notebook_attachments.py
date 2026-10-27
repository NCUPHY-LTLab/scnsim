"""Validate source-bound SVG attachments in generated, portable notebooks.

Only Markdown image references may carry attachments. Each attachment names
an existing SVG relative to the notebook, inside the repository, and embeds
exactly that file's bytes. No notebook execution or third-party imports occur.
"""

from __future__ import annotations

import base64
import binascii
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import xml.etree.ElementTree as ET


class _Images(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.line_offsets = [0] + [match.end() for match in re.finditer("\n", source)]
        self.markdown_raw = None
        self.raw_regions = []
        self.references = set()
        self.raw_text = None

    def handle_starttag(self, tag, attrs):
        if self.raw_text is not None:
            return
        if tag in {"script", "style", "pre", "textarea", "title"} and self.markdown_raw is None:
            self.markdown_raw = (tag, self._offset())
        if tag in {"script", "style", "textarea", "title"}:
            self.raw_text = tag
        if tag == "img":
            for name, value in attrs:
                if name == "src" and value and value.lower().startswith("attachment:"):
                    self.references.add(value.partition(":")[2])

    def _offset(self):
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_endtag(self, tag):
        if self.raw_text is not None and tag != self.raw_text:
            return
        if tag == self.raw_text:
            self.raw_text = None
        if self.markdown_raw is not None and tag == self.markdown_raw[0]:
            end = self.source.find(">", self._offset())
            self.raw_regions.append((self.markdown_raw[1], end + 1))
            self.markdown_raw = None

    def markdown_source(self):
        regions = list(self.raw_regions)
        if self.markdown_raw is not None:
            regions.append((self.markdown_raw[1], len(self.source)))
        text = self.source
        for start, end in reversed(regions):
            text = text[:start] + re.sub(r"[^\s]", "x", text[start:end]) + text[end:]
        return text

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)


# Destinations may be angle-bracketed, including paths containing spaces.
_DESTINATION = r'(?:<([^<>\n]+)>|([^\s)]+))'
_INLINE_IMAGE = re.compile(r'(?<!\\)!\[(?:\\.|[^\]\\])*\]\(\s*' + _DESTINATION + r'(?:\s+["\'][^\n]*["\'])?\s*\)')
_REFERENCE_IMAGE = re.compile(r'(?<!\\)!\[((?:\\.|[^\]\\])*)\](?:\[([^\]\n]*)\])?')
_DEFINITION = re.compile(r'^ {0,3}\[([^\]\n]+)\]:\s*' + _DESTINATION, re.M)


def _prose(source):
    """Remove code and comments so an example URI cannot authorize an image."""
    lines = []
    fence = None
    for line in source.splitlines():
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence is None:
            if marker:
                fence = marker.group(1)
            elif not line.startswith(("    ", "\t")):
                lines.append(line)
        elif marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence) and not marker.group(2).strip():
            fence = None
    text = re.sub(r'<!--.*?-->', '', '\n'.join(lines), flags=re.S)
    return re.sub(r'(`+).*?\1', '', text, flags=re.S)


# CommonMark raw HTML blocks suppress Markdown until their closing marker
# (script/pre/style/textarea, comments, declarations) or a blank line (other
# block tags). Real HTML img elements are still collected separately above.
_BLOCK_TAGS = (
    "address|article|aside|base|basefont|blockquote|body|caption|center|col|"
    "colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|"
    "footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|"
    "link|main|menu|menuitem|nav|noframes|ol|optgroup|option|p|param|search|"
    "section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul"
)
_BLOCK_START = re.compile(r'^ {0,3}</?(?:' + _BLOCK_TAGS + r')(?=[\s/>]|$)', re.I)
_COMPLETE_TAG = re.compile(r'^ {0,3}</?[A-Za-z][A-Za-z0-9-]*(?:\s+[^<>]*)?/?>\s*$')


def _markdown_prose(prose, masked):
    lines = []
    closing = None
    until_blank = False
    # Original lines decide HTML block boundaries; masked lines exclude raw
    # text even when an inline script starts after ordinary prose.
    for line, visible in zip(prose.splitlines(), masked.splitlines(), strict=True):
        if until_blank:
            if not line.strip():
                until_blank = False
                lines.append("")
            continue
        if closing is None:
            raw = re.match(r'^ {0,3}<(script|pre|style|textarea)(?=[\s>]|$)', line, re.I)
            if raw:
                closing = re.compile(r'</' + raw.group(1) + r'\s*>', re.I)
            elif re.match(r'^ {0,3}<!--', line):
                closing = re.compile(r'-->')
            elif re.match(r'^ {0,3}<\?', line):
                closing = re.compile(r'\?>')
            elif re.match(r'^ {0,3}<!\[CDATA\[', line):
                closing = re.compile(r'\]\]>')
            elif re.match(r'^ {0,3}<![A-Z]', line):
                closing = re.compile(r'>')
            elif _BLOCK_START.match(line) or _COMPLETE_TAG.match(line):
                until_blank = True
                continue
            else:
                lines.append(visible)
                continue
        if closing.search(line):
            closing = None
    return "\n".join(lines)


def _references(source):
    prose = _prose(source)
    images = _Images(prose)
    images.feed(prose)
    images.close()
    prose = _markdown_prose(prose, images.markdown_source())
    targets = [match.group(1) or match.group(2) for match in _INLINE_IMAGE.finditer(prose)]
    definitions = {
        ' '.join(match.group(1).lower().split()): match.group(2) or match.group(3)
        for match in _DEFINITION.finditer(prose)
    }
    for match in _REFERENCE_IMAGE.finditer(prose):
        if prose[match.end():].startswith("("):
            continue  # An inline image is not also a shortcut reference.
        label = match.group(2) or match.group(1)
        target = definitions.get(' '.join(label.lower().split()))
        if target is not None:
            targets.append(target)
    images.references.update(target.partition(":")[2] for target in targets if target.lower().startswith("attachment:"))
    return images.references


def validate_cell_attachments(cell: dict, notebook_path: Path, repository_root: Path) -> dict:
    """Return validated attachments for inclusion in the freshness comparison.

    Both distributed and freshly rendered cells use the same artifact Path;
    a display label is not a filesystem base. Invalid attachments raise
    AssertionError, including missing files and invalid encoding.
    """
    assert isinstance(notebook_path, Path), "notebook artifact must be a Path"
    assert isinstance(repository_root, Path), "repository root must be a Path"
    attachments = cell.get("attachments", {})
    assert isinstance(attachments, dict), "attachments must be a mapping"
    if cell.get("cell_type") != "markdown":
        assert not attachments, "only Markdown cells may contain attachments"
        return {}
    source = cell.get("source", "")
    if isinstance(source, list):
        assert all(isinstance(line, str) for line in source), "invalid Markdown source"
        source = "".join(source)
    assert isinstance(source, str), "invalid Markdown source"
    references = _references(source)
    assert references == set(attachments), "missing or unreferenced Markdown attachment"
    try:
        root = repository_root.resolve(strict=True)
        notebook = notebook_path.resolve()
        assert notebook.is_relative_to(root), "notebook is outside repository"
        normalized = {}
        for name, bundle in attachments.items():
            assert isinstance(name, str) and name, "invalid attachment path"
            assert not any(ord(char) < 32 or ord(char) == 127 for char in name), "control character in attachment path"
            assert not any(char in name for char in "\\:%?#"), "attachment path must be a relative filesystem path"
            assert not PurePosixPath(name).is_absolute() and not PureWindowsPath(name).drive, "absolute attachment path"
            assert PurePosixPath(name).suffix == ".svg", "only SVG attachments are allowed"
            path = (notebook.parent / name).resolve(strict=True)
            assert path.is_relative_to(root) and path.is_file(), "attachment source is outside repository or not a file"
            assert isinstance(bundle, dict) and set(bundle) == {"image/svg+xml"}, "only image/svg+xml is allowed"
            encoded = bundle["image/svg+xml"]
            assert isinstance(encoded, str), "attachment encoding must be a string"
            # Quarto emits MIME base64 with CR/LF line wrapping. Strip only
            # those two characters; validate=True still rejects other whitespace
            # and illegal characters, and re-encoding checks canonical pad bits.
            compact = encoded.replace("\r", "").replace("\n", "")
            decoded = base64.b64decode(compact, validate=True)
            assert base64.b64encode(decoded).decode("ascii") == compact, "noncanonical base64 attachment"
            assert decoded == path.read_bytes(), "attachment bytes differ from repository source"
            assert ET.fromstring(decoded).tag in {"svg", "{http://www.w3.org/2000/svg}svg"}, "attachment source is not an SVG image"
            normalized[name] = {"image/svg+xml": compact}
    except (OSError, ValueError, binascii.Error, ET.ParseError) as error:
        raise AssertionError(f"invalid source-bound attachment: {error}") from error
    return normalized
