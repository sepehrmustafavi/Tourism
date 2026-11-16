import argparse
from dataclasses import replace
from tourism.cli import load_config
from tourism.experiment import run_suite
p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--config',default='configs/main.json');p.add_argument('--out',default='results/judgment_budget');p.add_argument('--users',type=int,default=10);p.add_argument('--max-seconds',type=float,default=32400);a=p.parse_args()
cfg=replace(load_config(a.config),budget=10,judgment_budget=10)
print(run_suite(a.catalog,a.out,cfg,['full','travel_only','random_matched'],[11,29,47],a.users,['mixed','conflict','nonlinear'],max_seconds=a.max_seconds))
