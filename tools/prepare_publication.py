#!/usr/bin/env python3
"""One-time publication-copy preparation. Originals are never modified.

Run once before the initial commit; refuses to repeat once the manifest exists.
Paths below document the original export location; recipients do not run this.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HANDOFF = '/Users/nealprakash/3dCT/handoff/'
ORIGINAL_REPO = '/Users/nealprakash/3dCT/'
REPO_URL = 'https://github.com/nprakash1/3dCT/blob/main/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    destination = ROOT / 'publication_manifest.json'
    if destination.exists():
        raise SystemExit('Already prepared; refusing to transform snapshots again.')
    inventory = json.loads((ROOT / 'notebook_inventory.json').read_text())
    # Preserve only plain-text aggregate metrics/result tables, never HTML/widgets.
    keep = {'01': {31}, '02': {12}, '03': {32}, '06': {7, 8, 9, 10}}
    snapshots = []
    for path in sorted((ROOT / 'notebooks').rglob('*.ipynb')):
        original = path.read_bytes()
        nb = json.loads(original)
        kept = []
        for index, cell in enumerate(nb['cells']):
            outputs = []
            if index in keep.get(path.name[:2], set()):
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream':
                        outputs.append({'output_type': 'stream', 'name': output.get('name', 'stdout'),
                                        'text': output['text']})
                    elif 'text/plain' in output.get('data', {}):
                        outputs.append({'output_type': 'display_data', 'metadata': {},
                                        'data': {'text/plain': output['data']['text/plain']}})
            cell['metadata'] = {}
            cell.pop('attachments', None)
            if cell['cell_type'] == 'code':
                cell['outputs'] = outputs
                cell['execution_count'] = None
            if outputs:
                kept.append(index)
        nb['metadata'] = {k: v for k, v in nb.get('metadata', {}).items()
                          if k in ('kernelspec', 'language_info')}
        data = (json.dumps(nb, indent=1, ensure_ascii=False) + '\n').encode()
        path.write_bytes(data)
        source = next(e for e in inventory['selected_notebooks']
                      if Path(e['handoff_file']).name == path.name)
        snapshots.append(dict(file=path.relative_to(ROOT).as_posix(),
                              original_sha256=sha(original), published_sha256=sha(data),
                              source=source['source'], retained_output_cells=kept,
                              source_code_unchanged=True))
    for path in ROOT.glob('*.md'):
        doc = path.read_text()
        doc = re.sub(r'\]\(' + re.escape(ORIGINAL_HANDOFF) + r'([^)]*)\)', r'](\1)', doc)
        doc = re.sub(r'\]\(' + re.escape(ORIGINAL_REPO) + r'([^)]*)\)',
                     lambda m: '](' + REPO_URL + m.group(1) + ')', doc)
        doc = doc.replace(ORIGINAL_HANDOFF, './')
        doc = doc.replace(ORIGINAL_REPO + '.venv/bin/python', 'python3')
        path.write_text(doc)
    for entry in snapshots:
        entry['source'] = entry['source'].replace(ORIGINAL_REPO, 'original-repo:').replace(
            '/Users/nealprakash/Downloads/ct_notebook_copies/', 'original-executed-copy:')
    destination.write_text(json.dumps(dict(
        policy='Notebook source cells preserved; only selected plain-text aggregate outputs kept. '
               'All other outputs, attachments and session metadata removed. '
               'Original hashes and full local evidence remain with the owner. '
               'This is not a comprehensive privacy certification.',
        notebooks=snapshots), indent=2) + '\n')
    print('Prepared', len(snapshots), 'publication snapshots; source cells unchanged.')


if __name__ == '__main__':
    main()