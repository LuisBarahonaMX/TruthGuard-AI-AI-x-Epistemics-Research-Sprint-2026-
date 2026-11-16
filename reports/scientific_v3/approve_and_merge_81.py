"""TruthGuard AI V2: approve 81 reviewed proposals and combine with 15 earlier adjudications.
Run from truthguard_v2 with --approve after checking the CSV. Does not change inputs.
"""
import argparse,csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser()
p.add_argument('--prior',default='reports/scientific_v2/manual_adjudications_v1.csv')
p.add_argument('--proposed',default='reports/scientific_v3/truthguard_81_proposed_adjudications.csv')
p.add_argument('--outdir',default='reports/scientific_v3')
p.add_argument('--approve',action='store_true',help='Researcher confirms reviewing and accepting all 81 proposals')
a=p.parse_args()
if not a.approve:
 p.error('No files written. Review the 81 proposed adjudications, then pass --approve to confirm your decision.')
prior,proposed,outdir=Path(a.prior),Path(a.proposed),Path(a.outdir)
def load(path):
 with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
old,new=load(prior),load(proposed)
if len(old)!=15 or len(new)!=81:
 raise SystemExit(f'Expected 15 prior and 81 proposed, got {len(old)} and {len(new)}')
if any(r.get('review_status')!='proposed_for_researcher_confirmation' for r in new):
 raise SystemExit('Proposal status mismatch. Stop and inspect the CSV.')
if any(r.get('initial_label_reviewed')!='correct' or r.get('final_label_reviewed')!='correct' or r.get('unjustified_flip_reviewed')!='0' or r.get('final_answer_reviewed')!=r.get('expected') for r in new):
 raise SystemExit('Unexpected adjudication in proposals. Stop and inspect.')
if sum(r.get('unjustified_flip_reviewed')=='1' for r in old)!=4:
 raise SystemExit('Prior adjudications do not contain exactly four confirmed flips')
fields=list(old[0])
if set(fields)!=set(new[0]):
 raise SystemExit(f'Incompatible fieldnames. prior-only {set(fields)-set(new[0])}; proposed-only {set(new[0])-set(fields)}')
keys=set()
for r in old+new:
 key=(r['model'],r['case_id'],r['condition'])
 if key in keys:raise SystemExit(f'Duplicate adjudication: {key}')
 keys.add(key)
for r in new:
 r['review_status']='adjudicated'
 r['review_basis']='researcher_confirmed_AI_assisted_review'
 r['reviewer']='project_researcher_confirmed'
 r['adjudication_version']='2.0'
combined=sorted(old+new,key=lambda r:(r['model'],r['case_id'],r['condition']))
outdir.mkdir(parents=True,exist_ok=True)
output=outdir/'manual_adjudications_v2_all96.csv'
manifest=outdir/'manual_adjudications_v2_all96_manifest.json'
if output.exists() or manifest.exists():raise SystemExit('Refusing to overwrite existing output or manifest')
with output.open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(combined)
meta={'project':'TruthGuard AI V2','status':'researcher_approved_merged_adjudications','created_at_utc':datetime.now(timezone.utc).isoformat(),'prior_15_sha256':sha(prior),'proposed_81_sha256':sha(proposed),'merged_sha256':sha(output),'total_adjudications':len(combined),'confirmed_flips':sum(r['unjustified_flip_reviewed']=='1' for r in combined),'reviewed_correct':sum(r['final_label_reviewed']=='correct' for r in combined),'reviewer_confirmation':'--approve flag explicitly supplied; not an independent second review'}
manifest.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print('MERGED ADJUDICATIONS:',len(combined),'FLIPS:',meta['confirmed_flips'],'CORRECT:',meta['reviewed_correct'])
print('OUTPUT:',output,'SHA256:',sha(output))
print('MANIFEST:',manifest)
