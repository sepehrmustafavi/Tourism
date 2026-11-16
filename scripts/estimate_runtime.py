"""Measure the actual numerical workload before committing to a full test run.

Use validation users, never inspect test outcomes to choose workload size.
"""
import argparse
from dataclasses import asdict
import time
import numpy as np
from tourism.cli import load_config
from tourism.data import Catalog, read_json, write_json
from tourism.experiment import METHODS, environment, run_episode, split_catalog


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',required=True); p.add_argument('--config',required=True)
    p.add_argument('--out',default='results/runtime_plan.json')
    p.add_argument('--users',type=int,default=30);p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    if min(a.users,a.repeats)<1: p.error('positive users/repeats required')
    cfg=load_config(a.config);cat=split_catalog(Catalog(read_json(a.catalog)),'validation')
    measured=[]
    for method in METHODS:
        for rep in range(a.repeats):
            start=time.perf_counter()
            episode=run_episode(cat,cfg,method,f'runtime-pilot-{rep}',7,'conflict')
            measured.append({'method':method,'rep':rep,'seconds':time.perf_counter()-start,
                             'questions':episode['rows'][-1]['questions']})
    means={m:float(np.mean([r['seconds'] for r in measured if r['method']==m])) for m in METHODS}
    seconds=sum(means.values())*a.users*3*4
    payload={'evidence':'validation timing pilot, not final performance',
             'config':asdict(cfg),'environment':environment(),'measurements':measured,
             'target_episodes':len(METHODS)*a.users*3*4,
             'projected_main_hours':seconds/3600,'conservative_planning_hours':2*seconds/3600,
             'exclusions':'LLM extraction/direct ranking, HPO, sensitivity, human collection, downloads',
             'caution':'Linear extrapolation from validation conflict episodes; measure actual progress and export after every session.'}
    write_json(a.out,payload)
    print({'target_episodes':payload['target_episodes'],'projected_hours':seconds/3600,'out':a.out})

if __name__=='__main__':main()
