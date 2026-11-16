"""Generate a preregisterable one-factor sensitivity manifest, optionally execute."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from tourism.cli import load_config
from tourism.data import write_json
from tourism.experiment import run_suite
p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--config',default='configs/main.json');p.add_argument('--out',default='results/sensitivity');p.add_argument('--users',type=int,default=10);p.add_argument('--execute',action='store_true');p.add_argument('--max-seconds',type=float,default=32400);a=p.parse_args()
base=load_config(a.config);root=Path(a.out);settings={}
for field,values in {'particles':[128,256,512,1024],'shortlist':[4,12,48],'actions':[16,64,128],
                     'response_skip':[0.,.15,.4],'response_tau':[.5,1.,1.5],
                     'weights':[(.7,.2,.1),(.25,.6,.15),(.2,.3,.5)],'context':[(1.,0.),(2.,0.),(3.,0.)]}.items():
    for index,value in enumerate(values):settings[f'{field}_{index}']=replace(base,**{field:value})
write_json(root/'plan.json',{'users_per_seed_regime':a.users,'seeds':[11,29,47],
    'regimes':['aligned','mixed','conflict','nonlinear'],'settings':{k:asdict(v) for k,v in settings.items()}})
if a.execute:
    import time
    start=time.perf_counter()
    for name,cfg in settings.items():
        remaining=a.max_seconds-(time.perf_counter()-start)
        if remaining<=0:break
        r=run_suite(a.catalog,root/name,cfg,['full','random_matched','independent'],[11,29,47],a.users,
                    ['aligned','mixed','conflict','nonlinear'],max_seconds=remaining)
        if r['paused']:break
print(root/'plan.json')
