"""Prespecified held-out human travel NLL family from user-mean exports."""
import argparse
import csv
from tourism.data import write_json
from tourism.report import paired_test,holm
p=argparse.ArgumentParser();p.add_argument('--users',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=list(csv.DictReader(open(a.users)));groups={}
for row in rows:
    if row['evidence']=='human_stated_preference' and row.get('nll_T'):
        groups.setdefault(row['method'],{})[row['user']]=float(row['nll_T'])
tests=[]
for method,users in groups.items():
    if method=='full':continue
    paired=sorted(set(users)&set(groups.get('full',{})))
    if paired:tests.append({'comparison':method+' minus full','metric':'human held-out travel NLL; positive favors full',**paired_test([users[u]-groups['full'][u] for u in paired])})
write_json(a.out,{'primary_family':holm(tests)});print(a.out)
