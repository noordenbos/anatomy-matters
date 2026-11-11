#!/usr/bin/env python3
"""Install exactly the data paired with this code version, with SHA-256 checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install(root, *, source_dir=None, record_id=None, check_only=False, study_source=None):
    root = Path(root)
    manifest = json.loads((root/'release.json').read_text())
    record = record_id or manifest['zenodo_record_id']
    if record is not None and not str(record).isdigit():
        raise ValueError('Zenodo record ID must contain digits only')
    for item in manifest['files']:
        dest = root/item['path']
        if dest.is_file():
            if digest(dest) != item['sha256']:
                raise ValueError(f"{item['path']} does not match release {manifest['version']}. Preserve your local file and install the matching release data; it will not be overwritten.")
            print(f"Verified {item['path']}")
            continue
        if check_only:
            raise FileNotFoundError(f"Missing release data: {item['path']}")
        source = None
        if study_source and item['path'] == 'data/DLBCL_location_2026.h5ad':
            source = Path(study_source)
        elif source_dir:
            source = Path(source_dir)/item['name']
            if not source.exists(): source = Path(source_dir)/Path(item['path']).name
        if source is None and not record:
            raise FileNotFoundError('This release candidate has no published Zenodo record yet. Use --source-dir with its two data assets; no older or latest-version bundle will be substituted.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=dest.name+'.', suffix='.partial', dir=dest.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            if source is not None:
                shutil.copyfile(source, tmp)
            else:
                url = f"https://zenodo.org/records/{record}/files/{item['name']}?download=1"
                with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if tmp.stat().st_size != item['bytes'] or digest(tmp) != item['sha256']:
                raise ValueError(f"Checksum/size mismatch for {item['name']}; download was not installed")
            with tmp.open('rb') as stream:
                if stream.read(8) != b'\x89HDF\r\n\x1a\n':
                    raise ValueError('Expected an HDF5 file')
            os.replace(tmp, dest)
            print(f"Installed {item['path']}")
        finally:
            tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--study-source', type=Path)
    parser.add_argument('--record-id', default=os.environ.get('ZENODO_RECORD_ID'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    install(ROOT, source_dir=args.source_dir, study_source=args.study_source, record_id=args.record_id, check_only=args.check)

if __name__ == '__main__':
    main()
