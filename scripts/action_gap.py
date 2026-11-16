import argparse
from itertools import permutations
import numpy as np
from tourism.data import Catalog,read_json,write_json
from tourism.core import Posterior,build_actions,list_scores
p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--out',default='results/action_gap.json');p.add_argument('--items',type=int,default=8);a=p.parse_args()
if not 3<=a.items<=9:raise ValueError('Exhaustive check limited to 3–9 items')
cat=Catalog(read_json(a.catalog));ix=[];seen=set()
for i in cat.feasible((3.,0.)):
    if cat.destinations[i] not in seen:ix.append(i);seen.add(cat.destinations[i])
    if len(ix)==a.items:break
cat=cat.subset(ix);chi=cat.context((3.,0.));all_actions=list(permutations(range(len(ix)),3));rows=[]
for seed in [11,29,47]:
    post=Posterior(6,7,4,512,seed);v,t=post.utilities(cat.phi,cat.psi,chi)
    exact=list_scores(v,t,all_actions,cat.distance,(.25,.6,.15))[0]
    for cap in [16,64,128]:
        approx=build_actions(cat.phi,cat.psi,chi,cat.destinations,cat.distance,range(len(ix)),3,cap)
        j=list_scores(v,t,approx,cat.distance,(.25,.6,.15))[0]
        rows.append({'seed':seed,'cap':cap,'represented':len(approx),'exhaustive':len(all_actions),
                     'prior_bayes_utility_gap':float((post.weights@exact).max()-(post.weights@j).max())})
write_json(a.out,{'note':'Prior-only numerical search diagnostic, not catalog-wide optimality guarantee','rows':rows})
print(a.out)
