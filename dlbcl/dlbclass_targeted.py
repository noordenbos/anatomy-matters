"""DLBclass targeted-panel inputs with reference-mean imputation.

The source GSM contains already-selected study variants. This adapter changes
encoding/availability, not variant selection. BLYMFv2 coverage is provisional.
The canonical case tables are never written by this module.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'dlbclass'


def prepare_inputs(legacy_gsm, translocations, reference=None):
    """Return corrected GSM, measurement mask and reduced matrix (cases × 21).

    legacy_gsm uses the study builder's nonsilent=1 encoding (L265P=2).
    Only pass matrices from that builder; silent=1 in an official GSM must NOT
    be converted by this adapter. Missing FISH and regional CN are unmeasured.
    """
    ref = pd.read_csv(CONFIG/'feature_reference.csv', index_col=0) if reference is None else reference.copy()
    if not legacy_gsm.index.is_unique or not legacy_gsm.columns.is_unique:
        raise ValueError('Duplicate GSM feature or patient IDs')
    if set(legacy_gsm.index) != set(ref.index):
        raise ValueError('GSM feature set differs from the 163-feature reference')
    gsm = legacy_gsm.reindex(ref.index).astype(float)
    if not np.isfinite(gsm.to_numpy()).all():
        raise ValueError('Legacy GSM has nonfinite values')
    if not translocations.index.is_unique or not gsm.columns.isin(translocations.index).all():
        raise ValueError('Translocation table must uniquely cover every GSM patient')
    mask = pd.DataFrame(False, index=gsm.index, columns=gsm.columns)
    for feature, row in ref.iterrows():
        if row['type'] == 'Mutation':
            if not gsm.loc[feature].isin([0,1,2]).all():
                raise ValueError(f'Unexpected mutation encoding for {feature}')
            mask.loc[feature] = bool(row.blymfv2_mutation_measured) | gsm.loc[feature].gt(0)
            gsm.loc[feature] = gsm.loc[feature].replace(1,2)
        elif row['type'] == 'Regional CNV':
            if gsm.loc[feature].ne(0).any():
                raise ValueError('Regional CN calls present: supply a verified CN availability mask')
        elif row['type'] == 'Rearrangement':
            gene = feature.removeprefix('SV.').lower()
            status = translocations.reindex(gsm.columns)[gene+'_final']
            values = pd.to_numeric(status, errors='raise')
            if not values.dropna().isin([0,1]).all():
                raise ValueError(f'Unexpected FISH encoding for {feature}')
            mask.loc[feature] = values.notna()
            gsm.loc[feature] = values.fillna(0).to_numpy()*3
        else:
            raise ValueError(f'Unknown feature type: {row["type"]}')
    corrected = gsm.where(mask, ref.reference_mean, axis=0)
    weights = pd.read_csv(CONFIG/'reduction_weights.csv',index_col=0)
    reduced = corrected.T @ weights.loc[corrected.index]
    if not np.isfinite(reduced.to_numpy()).all():
        raise ValueError('Nonfinite reduced inputs')
    return corrected, mask, reduced


def run_official_ensemble(reduced, vendor):
    """Use the author's ensemble and confidence calibration without modification."""
    import os
    import torch
    vendor = Path(vendor).resolve()
    sys.path.insert(0,str(vendor))
    import src.classify_generic as classifier
    import src.nn as nn
    files=sorted((vendor/'saved_models/FINALMODEL_NN_evaluation_seeds1_100_folds5_reducedV3.4_removeN5').glob('*'))
    if len(files)!=500:raise ValueError(f'Expected 500 original models, found {len(files)}')
    nets=[]
    for file in files:
        state=torch.load(file,map_location='cpu',weights_only=True)
        net=nn.Net(10,reduced.shape[1],5);net.load_state_dict(state);net.eval();nets.append(net)
    with torch.no_grad():
        results=classifier.classify_samples_winning_model(reduced,nets=nets)
    results['DLBclass']=results.PredictedCluster.astype(int).map(lambda x:f'C{x}')
    models_digest=hashlib.sha256(''.join(f.name+hashlib.sha256(f.read_bytes()).hexdigest() for f in files).encode()).hexdigest()
    return results,models_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-gsm',required=True,type=Path)
    parser.add_argument('--translocations',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--vendor',type=Path,help='Official DLBclass-tool checkout (requires torch); omit to prepare inputs only')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(args.legacy_gsm,sep='\t',index_col=0)
    fish=pd.read_csv(args.translocations,sep='\t',index_col=0)
    gsm,mask,reduced=prepare_inputs(source,fish)
    gsm.to_csv(args.output/'imputed_gsm.tsv',sep='\t')
    mask.to_csv(args.output/'measured_mask.tsv',sep='\t')
    reduced.to_csv(args.output/'reduced_inputs.tsv',sep='\t')
    report={'n_cases':len(reduced),'n_features':len(gsm),'input_encoding':'study legacy nonsilent=1 (not official silent=1)',
        'provisional_panel_assumption':json.loads((CONFIG/'provenance.json').read_text())['coverage_assumption'],
        'imputed_features_per_case':(~mask).sum().to_dict(),
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.legacy_gsm,args.translocations,*sorted(CONFIG.glob('*'))]}}
    if args.vendor:
        result,digest=run_official_ensemble(reduced,args.vendor)
        result.to_csv(args.output/'dlbclass_results.tsv',sep='\t')
        report['model_ensemble_sha256']=digest;report['class_counts']=result.DLBclass.value_counts().to_dict()
    (args.output/'run_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','imputed_features_per_case']},indent=2))

if __name__=='__main__':main()
