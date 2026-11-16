"""All-attempt frozen-LLM baseline utility comparison, retaining invalid generations."""
import argparse
from pathlib import Path
from tourism.data import read_json,write_json
from tourism.report import paired_test,holm
p=argparse.ArgumentParser();p.add_argument('--direct',required=True);p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=[]
for path in sorted(Path(a.direct).glob('*.json')):
    llm=read_json(path)
    source=llm.get('source_episode')
    if source is None: source=path.name.replace('llm_direct','random_matched')
    ref=read_json(Path(a.runs)/source);last=ref['rows'][-1]
    full_path=Path(a.runs)/source.replace('random_matched','full')
    if not full_path.exists():raise ValueError('Missing paired full episode')
    full=read_json(full_path)['rows'][-1]
    net=llm['net_utility'] if llm['valid'] else -ref['config']['evaluation_cost']*last['judgments_requested']
    rows.append({'user':last['user'],'seed':last['seed'],'regime':last['regime'],'evidence':last['evidence'],
                 'valid':llm['valid'],'llm_net_utility':net,'full_net_utility':full['net_utility']})
tests=[]
for evidence,regime in sorted({(r['evidence'],r['regime']) for r in rows}):
    subset=[r for r in rows if r['evidence']==evidence and r['regime']==regime];units={}
    for r in subset:
        key=r['user'] if evidence.startswith('human') else f"{r['seed']}:{r['user']}"
        units.setdefault(key,[]).append(r['full_net_utility']-r['llm_net_utility'])
    differences=[sum(v)/len(v) for v in units.values()]
    tests.append({'evidence':evidence,'regime':regime,**paired_test(differences),
                  'generation_validity':sum(r['valid'] for r in subset)/len(subset)})
write_json(a.out,{'rows':rows,'tests':holm(tests),'invalid_policy':'zero reference utility; retain requested-judgment cost'})
print(a.out)
