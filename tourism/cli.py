import argparse
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import json
import time
import numpy as np
from .data import Catalog, read_json, write_json, fetch_synthtrips, make_synthetic_catalog, digest
from .experiment import Config, METHODS, run_suite, run_episode, split_catalog
from .llm import extract
from .report import summarize


def load_config(path):
    values=read_json(path) if path else {}
    for key in ('weights','context'):
        if key in values: values[key]=tuple(values[key])
    cfg=Config(**values)
    if min(cfg.particles,cfg.actions,cfg.candidates,cfg.shortlist,cfg.pair_pool,cfg.k)<1 or cfg.budget<0:
        raise ValueError('Invalid size or budget')
    if not 0<=cfg.skip<=1 or min(cfg.cost,cfg.evaluation_cost)<0: raise ValueError('Invalid cost/skip')
    return cfg


def tune(args):
    cfg=load_config(args.config); cat=split_catalog(Catalog(read_json(args.catalog)),'validation')
    trials=[]; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for particles,cost,tau in product((256,512),(.001,.004),(.7,1.)):
        candidate=replace(cfg,particles=particles,cost=cost,tau=tau)
        path=out/f'trial_{particles}_{cost}_{tau}.json'
        signature={'catalog':cat.hash,'config':asdict(candidate),'users':args.users,
                   'code_hash':digest({p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')})}
        if path.exists():
            trial=read_json(path)
            if trial['signature']!=json.loads(json.dumps(signature)): raise ValueError('Stale tuning trial')
        else:
            scores=[]; start=time.perf_counter()
            for regime in ('aligned','conflict','mixed','nonlinear'):
                for user in range(args.users):
                    episode=run_episode(cat,candidate,'full',f'validation-{user}',7,regime)
                    scores.append(episode['rows'][-1]['net_utility'])
            trial={'signature':signature,'score':float(np.mean(scores)),
                   'seconds':time.perf_counter()-start,'config':asdict(candidate)}
            write_json(path,trial)
        trials.append(trial)
    # All candidates are scored using the SAME evaluation_cost, regardless of controller lambda.
    best=max(trials,key=lambda t:(t['score'],-t['config']['particles'],-t['config']['cost'],-t['config']['tau']))
    write_json(out/'selected.json',best['config'])
    write_json(out/'selection.json',{'criterion':'validation user mean net utility; fixed evaluation burden',
        'split':'destination-disjoint validation','seed':7,'trials':trials,'selected':best})
    return {'selected':str(out/'selected.json'),'score':best['score']}


def main():
    p=argparse.ArgumentParser(description='Tourism preference disambiguation experiments')
    sub=p.add_subparsers(dest='command',required=True)
    fetch=sub.add_parser('fetch'); fetch.add_argument('--out',default='data/catalog_raw.json'); fetch.add_argument('--revision',default='main'); fetch.add_argument('--max-cities',type=int,default=200)
    fixture=sub.add_parser('fixture'); fixture.add_argument('--out',default='data/fixture.json'); fixture.add_argument('--items',type=int,default=100)
    ex=sub.add_parser('extract'); ex.add_argument('--catalog',required=True); ex.add_argument('--out',required=True); ex.add_argument('--cache',default='data/cache'); ex.add_argument('--backend',choices=('lexical','llm'),default='lexical'); ex.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct'); ex.add_argument('--revision')
    for name in ('run','tune','replay','benchmark'):
        q=sub.add_parser(name); q.add_argument('--catalog',required=True); q.add_argument('--out',required=True); q.add_argument('--config')
        if name=='run':
            q.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS)); q.add_argument('--seeds',nargs='+',type=int,default=[11,29,47]); q.add_argument('--users',type=int,default=30); q.add_argument('--regimes',nargs='+',default=['aligned','mixed','conflict','nonlinear']); q.add_argument('--split',choices=('train','validation','test'),default='test'); q.add_argument('--max-seconds',type=float)
        elif name=='tune': q.add_argument('--users',type=int,default=4)
        elif name=='replay':
            q.add_argument('--responses',required=True); q.add_argument('--protocol',required=True); q.add_argument('--seeds',nargs='+',type=int,default=[11,29,47]); q.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
        else: q.add_argument('--repeats',type=int,default=10)
    su=sub.add_parser('survey'); su.add_argument('--catalog',required=True); su.add_argument('--out',default='data/surveys'); su.add_argument('--participant',required=True); su.add_argument('--seed',type=int,default=11)
    dr=sub.add_parser('direct'); dr.add_argument('--catalog',required=True); dr.add_argument('--runs',required=True); dr.add_argument('--out',required=True); dr.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct'); dr.add_argument('--revision')
    re=sub.add_parser('report'); re.add_argument('--runs',required=True); re.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='fetch':
        payload=fetch_synthtrips(a.out,a.revision,a.max_cities); result={'items':len(payload['items']),'revision':payload['revision']}
    elif a.command=='fixture': write_json(a.out,make_synthetic_catalog(a.items)); result={'path':a.out,'evidence':'synthetic fixture only'}
    elif a.command=='extract': result={'items':len(extract(a.catalog,a.out,a.cache,a.backend,a.model,a.revision)['items'])}
    elif a.command=='run': result=run_suite(a.catalog,a.out,load_config(a.config),a.methods,a.seeds,a.users,a.regimes,a.split,a.max_seconds)
    elif a.command=='tune': result=tune(a)
    elif a.command=='report': result=summarize(a.runs,a.out)
    elif a.command=='direct':
        from .direct import direct_rank
        result=direct_rank(a.catalog,a.runs,a.out,a.model,a.revision)
    elif a.command=='survey':
        from .human import make_survey
        result=make_survey(a.catalog,a.out,a.participant,a.seed)
    elif a.command=='replay':
        from .human import replay
        result=replay(a.catalog,a.responses,a.protocol,a.out,load_config(a.config),a.methods,a.seeds)
    elif a.command=='benchmark':
        cfg=load_config(a.config); cat=split_catalog(Catalog(read_json(a.catalog)),'validation'); rows=[]
        for particles,actions,short in product((128,256,512),(16,64),(4,12)):
            for rep in range(a.repeats):
                conf=replace(cfg,particles=particles,actions=actions,shortlist=short,budget=1,cost=0.)
                start=time.perf_counter(); r=run_episode(cat,conf,'full',f'benchmark-{rep}',11,'mixed')
                rows.append({'particles':particles,'actions_cap':actions,'shortlist':short,'rep':rep,
                             'seconds':time.perf_counter()-start,'query_p50':r['latency_p50'],
                             'environment':r['environment']})
        write_json(a.out,rows); result={'measurements':len(rows),'path':a.out}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
