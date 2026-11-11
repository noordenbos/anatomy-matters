"""Backed-up, atomic cleanup of duplicate case labels in a local H5AD bundle.

Preview: python -m dlbcl.migrate_case_assignments data/DLBCL_location_2026.h5ad
Apply:   python -m dlbcl.migrate_case_assignments data/DLBCL_location_2026.h5ad --apply
The canonical discovery/validation tables and all measured data are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import pandas as pd
from anndata.io import read_elem, write_elem

from .case_assignments import clean_assignment_storage

CANONICAL = ('uns/case_classifications', 'uns/validation_cohort/case_classification_validation')
TOP_KEYS = ('case_classifications', 'case_clinical', 'tumorimmune_archetype_id',
            'tumorimmune_archetype_id_raw', 'abundance_cluster_30', 'abundance_cluster')
VC_KEYS = ('case_classification_validation', 'meta', 'predictions', 'prediction_scores')


def read_assignment_payload(path: Path) -> dict:
    with h5py.File(path, 'r') as f:
        uns = {k: read_elem(f['uns'][k]) for k in TOP_KEYS if k in f['uns']}
        vc = f['uns/validation_cohort']
        uns['validation_cohort'] = {k: read_elem(vc[k]) for k in VC_KEYS if k in vc}
    return uns


def migration_plan(before: dict, after: dict) -> dict:
    changes = {'removed_uns_keys': sorted(set(before) - set(after)), 'removed_columns': {},
               'removed_validation_keys': sorted(set(before['validation_cohort']) - set(after['validation_cohort']))}
    if 'case_clinical' in before:
        changes['removed_columns']['case_clinical'] = sorted(set(before['case_clinical']) - set(after['case_clinical']))
    changes['removed_columns']['validation_cohort/meta'] = sorted(
        set(before['validation_cohort']['meta']) - set(after['validation_cohort']['meta']))
    changes['scores_preserved_as'] = 'validation_cohort/prediction_scores'
    changes['canonical_cases'] = {k:len(before[k]) for k in ['case_classifications']}
    changes['canonical_cases']['validation_cohort/case_classification_validation'] = len(before['validation_cohort']['case_classification_validation'])
    return changes


def migrate_bundle(path: Path | str, *, apply: bool = False) -> dict:
    path = Path(path).resolve(strict=True)  # preserve any release symlink
    before = read_assignment_payload(path)
    after = clean_assignment_storage(before)
    plan = migration_plan(before, after)
    plan['path'] = str(path)
    if not apply:
        return plan
    # An already-clean bundle is a no-op, including no extra backup.
    if (not plan['removed_uns_keys'] and not plan['removed_validation_keys']
            and not any(plan['removed_columns'].values())):
        plan['status'] = 'already clean'
        return plan
    backup = path.with_name(path.stem + '.before_case_assignment_cleanup.h5ad')
    if backup.exists():
        raise FileExistsError(f'Refusing to replace migration backup: {backup}')
    source_stat = path.stat()
    shutil.copy2(path, backup)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem+'.migration-', suffix='.h5ad', dir=path.parent)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        shutil.copy2(path, temporary)
        with h5py.File(temporary, 'r+') as f:
            uns = f['uns']
            for key in plan['removed_uns_keys']:
                del uns[key]
            if plan['removed_columns'].get('case_clinical'):
                del uns['case_clinical']
                write_elem(uns, 'case_clinical', after['case_clinical'])
            vc = uns['validation_cohort']
            for key in plan['removed_validation_keys']:
                del vc[key]
            if plan['removed_columns']['validation_cohort/meta']:
                del vc['meta']
                write_elem(vc, 'meta', after['validation_cohort']['meta'])
            if 'prediction_scores' not in vc and 'prediction_scores' in after['validation_cohort']:
                write_elem(vc, 'prediction_scores', after['validation_cohort']['prediction_scores'])
        with h5py.File(backup, 'r') as old, h5py.File(temporary, 'r') as new:
            for key in CANONICAL:
                pd.testing.assert_frame_equal(read_elem(old[key]), read_elem(new[key]))
        verified = read_assignment_payload(temporary)
        assert not migration_plan(verified, clean_assignment_storage(verified))['removed_validation_keys']
        if path.stat().st_mtime_ns != source_stat.st_mtime_ns or path.stat().st_size != source_stat.st_size:
            raise RuntimeError('Source H5AD changed during migration; original left untouched')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    plan.update(status='applied', backup=str(backup), canonical_tables_unchanged=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate_bundle(args.path, apply=args.apply), indent=2))

if __name__ == '__main__':
    main()
