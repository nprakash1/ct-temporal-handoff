#!/usr/bin/env python3
"""Portable verification of published snapshots; no original files or GPU needed."""
import ast
import hashlib
import json
from pathlib import Path
import re

HANDOFF = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((HANDOFF / 'publication_manifest.json').read_text())
    assert len(manifest['notebooks']) == 8
    for entry in manifest['notebooks']:
        path = HANDOFF / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['published_sha256'], path
        nb = json.loads(path.read_text())
        assert nb['nbformat'] == 4
        assert [i for i, c in enumerate(nb['cells']) if c.get('outputs')] == entry['retained_output_cells']
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if not any(l.lstrip().startswith(('!', '%', '?')) for l in source.splitlines()):
                    ast.parse(source)
    for path in HANDOFF.glob('*.md'):
        doc = path.read_text()
        assert doc.count('```') % 2 == 0
        for link in re.findall(r'\]\(([^)]+)\)', doc):
            if not link.startswith(('https://', 'http://', '#')):
                assert not link.startswith('/'), (path, link)
                assert (path.parent / link.split('#')[0]).is_file(), (path, link)
    for path in (HANDOFF / 'tools').glob('*.py'):
        ast.parse(path.read_text())
    print('PASS: eight published notebook hashes/output selections, plain-cell syntax, '
          'relative documentation links and tool syntax.')


if __name__ == '__main__':
    main()