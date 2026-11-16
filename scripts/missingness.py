import argparse
import copy
import random
from tourism.data import read_json,write_json
p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);p.add_argument('--rate',type=float,default=.3);a=p.parse_args()
if not 0<=a.rate<=1:raise ValueError('Rate outside unit interval')
c=copy.deepcopy(read_json(a.catalog));rng=random.Random(11);by_destination={}
for item in c['items']:
    d=item['destination']
    if d not in by_destination:by_destination[d]=[rng.random()<a.rate for _ in range(2)]
    for key,missing in zip(('budget','walkability'),by_destination[d]):
        if missing:item['attributes'][key]=None
c['missingness_intervention']={'rate':a.rate,'seed':11,'unit':'destination_attribute'};write_json(a.out,c)
print(a.out)
