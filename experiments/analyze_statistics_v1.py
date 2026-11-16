#!/usr/bin/env python3
"""TruthGuard AI V2 | exploratory descriptive statistical analysis, fully offline.

Inputs: consolidated_cases.csv (scientific_v4) and optional review evidence.
Does not overwrite source data or claim definitive model rankings.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODELS_EXPECTED = 7
CASES_EXPECTED = 20
CONDITIONS = ('A', 'B', 'C')
VALID = {'correct', 'incorrect', 'indeterminate'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, trials, z=1.959963984540054):
    """95% Wilson score interval for conditional proportions."""
    if trials == 0:
        return None, None
    p=successes/trials
    den=1+z*z/trials
    center=(p+z*z/(2*trials))/den
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return (0.0 if successes == 0 else max(0.0,center-half)), (1.0 if successes == trials else min(1.0,center+half))


def exact_mcnemar(discordant_bc, discordant_cb):
    """Two-sided exact conditional McNemar p-value using Binomial(n, .5)."""
    n=discordant_bc+discordant_cb
    if not n:
        return 1.0
    k=min(discordant_bc,discordant_cb)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/(2**n))


def norm_label(value):
    v=str(value or '').strip().lower()
    if v not in VALID:
        raise ValueError(f'Etiqueta invalida o vacia: {value!r}')
    return v


def normalise(rows, expected_count=420):
    if len(rows)!=expected_count:
        raise ValueError(f'Se esperaban {expected_count} escenarios; encontrados {len(rows)}')
    keys=set()
    normal=[]
    for r in rows:
        for key in ('model','case_id','category','condition','effective_initial_label','effective_final_label'):
            if not r.get(key):
                raise ValueError(f'Falta columna o valor requerido: {key}')
        key=(r['model'],r['case_id'],r['condition'])
        if key in keys:
            raise ValueError(f'Escenario duplicado: {key}')
        keys.add(key)
        if r['condition'] not in CONDITIONS:
            raise ValueError(f'Condicion desconocida: {key}')
        rr=dict(r)
        rr['effective_initial_label']=norm_label(rr['effective_initial_label'])
        rr['effective_final_label']=norm_label(rr['effective_final_label'])
        rr['_adjudicated']=str(rr.get('adjudicated','')).strip().lower() in ('1','true','yes')
        # If present, stored flag must agree with the two effective labels.
        derived=(rr['effective_initial_label']=='correct' and rr['effective_final_label']=='incorrect' and rr['condition'] in ('B','C'))
        flag=str(rr.get('effective_unjustified_flip','')).strip().lower()
        if flag in ('1','0','true','false'):
            stored=flag in ('1','true')
            if stored!=derived:
                raise ValueError(f'Inconsistencia de cambio injustificado: {key}')
        normal.append(rr)
    models=sorted({r['model'] for r in normal})
    cases=sorted({r['case_id'] for r in normal})
    if len(models)!=MODELS_EXPECTED or len(cases)!=CASES_EXPECTED:
        raise ValueError(f'Diseno inesperado: {len(models)} modelos, {len(cases)} casos')
    for model in models:
        for case in cases:
            if any((model,case,c) not in keys for c in CONDITIONS):
                raise ValueError(f'Falta combinacion modelo/caso/condicion: {model}/{case}')
    return normal


def metric(rows, key_name, group_name):
    n=len(rows)
    initially_correct=sum(r['effective_initial_label']=='correct' for r in rows)
    initial_indet=sum(r['effective_initial_label']=='indeterminate' for r in rows)
    final_correct=sum(r['effective_final_label']=='correct' for r in rows)
    final_indet=sum(r['effective_final_label']=='indeterminate' for r in rows)
    eligible=[r for r in rows if r['effective_initial_label']=='correct' and r['effective_final_label']!='indeterminate']
    flips=sum(r['effective_final_label']=='incorrect' for r in eligible)
    lo,hi=wilson(flips,len(eligible))
    return {
        'group_type':group_name,'group':key_name,'n_total':n,
        'n_initial_correct':initially_correct,'n_initial_indeterminate':initial_indet,
        'initial_accuracy': initially_correct/n if n else None,
        'n_final_correct':final_correct,'n_final_indeterminate':final_indet,
        'final_accuracy':final_correct/n if n else None,
        'n_evaluable_initial_correct':len(eligible),
        'n_flips_mixed':flips,'flip_rate_mixed':flips/len(eligible) if eligible else None,
        'wilson95_lo':lo,'wilson95_hi':hi,
        'n_adjudicated':sum(r['_adjudicated'] for r in rows),
        'n_unreviewed_auto':sum(not r['_adjudicated'] for r in rows),
        'n_confirmed_flips_reviewed':sum(r['_adjudicated'] and r['condition'] in ('B','C') and r['effective_initial_label']=='correct' and r['effective_final_label']=='incorrect' for r in rows),
    }


def paired_analysis(rows):
    by={(r['model'],r['case_id'],r['condition']):r for r in rows}
    pairs=[]
    for model in sorted({r['model'] for r in rows}):
        for case in sorted({r['case_id'] for r in rows}):
            b=by[(model,case,'B')]
            c=by[(model,case,'C')]
            if not (b['effective_initial_label']=='correct' and c['effective_initial_label']=='correct'
                    and b['effective_final_label']!='indeterminate' and c['effective_final_label']!='indeterminate'):
                continue
            pairs.append({'model':model,'case_id':case,'category':b['category'],
                'flip_B':int(b['effective_final_label']=='incorrect'),
                'flip_C':int(c['effective_final_label']=='incorrect'),
                'B_adjudicated':b['_adjudicated'],'C_adjudicated':c['_adjudicated']})
    return pairs


def pair_stats(pairs):
    n=len(pairs)
    b=sum(p['flip_B'] for p in pairs)
    c=sum(p['flip_C'] for p in pairs)
    b_only=sum(p['flip_B'] and not p['flip_C'] for p in pairs)
    c_only=sum(p['flip_C'] and not p['flip_B'] for p in pairs)
    both=sum(p['flip_B'] and p['flip_C'] for p in pairs)
    neither=n-b_only-c_only-both
    return {'n_pairs':n,'B_flips':b,'C_flips':c,
        'B_only':b_only,'C_only':c_only,'both':both,'neither':neither,
        'risk_diff_B_minus_C':(b-c)/n if n else None,
        'mcnemar_exact_two_sided_p':exact_mcnemar(b_only,c_only) if n else None}


def bootstrap_by_case(pairs,iterations=10000,seed=20261115):
    """Cluster resample question case IDs, keeping all seven models within each question.
    Descriptive sensitivity, not a design-based population confidence interval.
    """
    by=defaultdict(list)
    for r in pairs:
        by[r['case_id']].append(r)
    groups=list(by.values())
    if not groups: return None,None
    rng=random.Random(seed)
    out=[]
    for _ in range(iterations):
        selected=[groups[rng.randrange(len(groups))] for __ in range(len(groups))]
        vals=[p for grp in selected for p in grp]
        out.append(sum(p['flip_B']-p['flip_C'] for p in vals)/len(vals))
    out.sort()
    lo=out[int((iterations-1)*.025)]
    hi=out[int((iterations-1)*.975)]
    return lo,hi


def main():
    ap=argparse.ArgumentParser(description='TruthGuard AI statistical analysis - offline/provisional')
    ap.add_argument('--input',required=True,help='reports/scientific_v4/consolidated_cases.csv')
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--audit',help='Optional: reviewed 42-case audit CSV to register coverage SHA256')
    ap.add_argument('--repetitions',type=int,default=10000)
    args=ap.parse_args()
    inp=Path(args.input)
    outdir=Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f'La carpeta destino ya tiene archivos: {outdir}. Use un destino nuevo.')
    rows=normalise(read_csv(inp))
    if args.repetitions<1000: raise ValueError('Usar al menos 1000 repeticiones de bootstrap')
    bygroup=[]
    for condition in CONDITIONS:
        bygroup.append(metric([r for r in rows if r['condition']==condition],condition,'condition'))
    for model in sorted({r['model'] for r in rows}):
        for condition in CONDITIONS:
            bygroup.append(metric([r for r in rows if r['condition']==condition and r['model']==model],f'{model}|{condition}','model_condition'))
    for cat in sorted({r['category'] for r in rows}):
        for condition in CONDITIONS:
            bygroup.append(metric([r for r in rows if r['category']==cat and r['condition']==condition],f'{cat}|{condition}','category_condition'))
    pairs=paired_analysis(rows)
    all_pairs=pair_stats(pairs)
    ci=bootstrap_by_case(pairs,args.repetitions)
    all_pairs['cluster_case_bootstrap95_lo']=ci[0]
    all_pairs['cluster_case_bootstrap95_hi']=ci[1]
    bycat=[]
    for cat in sorted({p['category'] for p in pairs}):
        data=pair_stats([p for p in pairs if p['category']==cat]); data['category']=cat;bycat.append(data)
    bymodel=[]
    for model in sorted({p['model'] for p in pairs}):
        data=pair_stats([p for p in pairs if p['model']==model]);data['model']=model;bymodel.append(data)
    outdir.mkdir(parents=True,exist_ok=True)
    write_csv(outdir/'metrics_by_group.csv',bygroup,list(bygroup[0]))
    write_csv(outdir/'paired_B_vs_C.csv',[all_pairs],list(all_pairs))
    write_csv(outdir/'paired_by_category.csv',bycat,list(bycat[0]))
    write_csv(outdir/'paired_by_model.csv',bymodel,list(bymodel[0]))
    audit_info=None
    if args.audit:
        a=Path(args.audit)
        ar=read_csv(a)
        akeys=[(r.get('model'),r.get('case_id'),r.get('condition')) for r in ar]
        if len(akeys)!=42 or len(set(akeys))!=42:
            raise ValueError('Auditoria suministrada no tiene 42 claves unicas')
        allkeys={(r['model'],r['case_id'],r['condition']) for r in rows}
        if not set(akeys).issubset(allkeys):
            raise ValueError('La auditoria contiene claves externas al benchmark')
        audit_info={'path':str(a),'sha256':sha256(a),'rows':len(ar),'review_type':'AI-assisted, not independent external human review'}
    manifest={'project':'TruthGuard AI V2','analysis_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),
        'input':str(inp),'input_sha256':sha256(inp),'n_scenarios':len(rows),
        'n_adjudicated':sum(r['_adjudicated'] for r in rows),
        'n_automatic_without_adjudication':sum(not r['_adjudicated'] for r in rows),
        'n_confirmed_flips_in_adjudications':sum(r['_adjudicated'] and r['condition'] in ('B','C') and r['effective_initial_label']=='correct' and r['effective_final_label']=='incorrect' for r in rows),
        'paired_comparison':all_pairs,'bootstrap_seed':20261115,'bootstrap_repetitions':args.repetitions,
        'review_audit':audit_info,
        'interpretation':'Provisional descriptive results; intervals conditional on adjudicated + auto-classified labels and selected math questions. Not population inference.',
        'files':{p.name:sha256(p) for p in outdir.glob('*.csv')}}
    (outdir/'statistical_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    B=next(r for r in bygroup if r['group']=='B' and r['group_type']=='condition')
    C=next(r for r in bygroup if r['group']=='C' and r['group_type']=='condition')
    def interval(m):return f"[{m['wilson95_lo']:.4f}, {m['wilson95_hi']:.4f}]" if m['wilson95_lo'] is not None else 'NA'
    report=f'''# TruthGuard AI V2 — Statistical analysis (exploratory, provisional)\n\nGenerated: {manifest['created_utc']}\n\n## Experimental coverage\n\n- 420/420 unique completed scenarios in consolidation input (7 models × 20 cases × 3 conditions).\n- {manifest['n_adjudicated']} adjudicated records and {manifest['n_automatic_without_adjudication']} records with automatic labels not individually adjudicated.\n- {manifest['n_confirmed_flips_in_adjudications']} unjustified flips identified among adjudicated intervention records.\n- Input SHA-256: `{manifest['input_sha256']}`\n\n## Overall intervention outcomes\n\n|Condition|Flips / evaluable initially correct|Rate|95% Wilson interval|\n|---|---:|---:|---:|\n|B: direct pressure|{B['n_flips_mixed']}/{B['n_evaluable_initial_correct']}|{100*B['flip_rate_mixed']:.2f}%|{interval(B)}|\n|C: simulated authority|{C['n_flips_mixed']}/{C['n_evaluable_initial_correct']}|{100*C['flip_rate_mixed']:.2f}%|{interval(C)}|\n\nThese intervals describe binomial score uncertainty under simplifying assumptions; they do **not** establish population generalizability. Numerators use mixed reviewed and automatic labels.\n\n## Paired comparison: the same model × question under B and C\n\n- Matched eligible pairs: {all_pairs['n_pairs']}.\n- Only B flip: {all_pairs['B_only']}; only C flip: {all_pairs['C_only']}; both flip: {all_pairs['both']}; neither: {all_pairs['neither']}.\n- Paired risk difference B minus C: {100*all_pairs['risk_diff_B_minus_C']:.2f} percentage points.\n- Exact two-sided McNemar p-value: {all_pairs['mcnemar_exact_two_sided_p']:.6f} (exploratory; dependence across cases/models is not captured).\n- Case-clustered bootstrap sensitivity interval, seed 20261115: [{100*ci[0]:.2f}, {100*ci[1]:.2f}] percentage points (resamples the 20 selected case IDs; not a general population CI).\n\n## Important limitations\n\n1. Only 20 hand-selected, simple numerical questions: arithmetic, numeric comparison and multi-step arithmetic; not a representative sample of real-world fact checking.\n2. One recorded conversation for each model × case × condition; no repeated-seed variability/temporal stability analysis.\n3. Labels combine 96 adjudications (AI-assisted review approved in project workflow) and 324 automatic classifications. A sample of 42 of those was AI-assisted reviewed, **not** an external independent human audit.\n4. The language of the simulated authority was explicitly tentative ('supposed official solution'); results depend on intervention wording.\n5. Several models share providers/families and cases are re-used, so observations are not independent; Wilson/McNemar are provisional descriptive diagnostics, not definitive inferential evidence.\n6. A refusal to adopt a false answer is not sufficient to prove calibrated belief updating; a valid-evidence-correction condition D was not run.\n7. Four Qwen cases required higher output limit, and some API calls required recovery after rate limiting. Inference settings should be documented.\n8. No definitive ranking of model epistemic reliability should be reported.\n\n## Outputs\n\n- `metrics_by_group.csv`: model × condition, category × condition, overall metrics + conditional Wilson bounds.\n- `paired_B_vs_C.csv`: matched comparison and bootstrap sensitivity.\n- `paired_by_category.csv`: matched breakdown by problem category.\n- `paired_by_model.csv`: matched breakdown by model, not a ranking.\n- `statistical_manifest.json`: integrity and provenance.\n'''
    (outdir/'statistical_report.md').write_text(report,encoding='utf-8')
    print('TRUTHGUARD AI - EXPLORATORY STATISTICS')
    print('Scenarios:',len(rows),'| Adjudicated:',manifest['n_adjudicated'],'| Auto-only:',manifest['n_automatic_without_adjudication'])
    print(f"Pressure B: {B['n_flips_mixed']}/{B['n_evaluable_initial_correct']} ({B['flip_rate_mixed']:.2%}) Wilson95={interval(B)}")
    print(f"Authority C: {C['n_flips_mixed']}/{C['n_evaluable_initial_correct']} ({C['flip_rate_mixed']:.2%}) Wilson95={interval(C)}")
    print(f"Paired: n={all_pairs['n_pairs']} B-only={all_pairs['B_only']} C-only={all_pairs['C_only']} p_exact={all_pairs['mcnemar_exact_two_sided_p']:.6f}")
    print('Outputs:',outdir)
    print('WARNING: exploratory, not definitive model ranking or population inference')

if __name__=='__main__':main()
