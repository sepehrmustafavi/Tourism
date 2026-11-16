"""Export randomized source-linked feature records for independent semantic audit."""
import argparse
import csv
import random
from tourism.data import read_json
p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);p.add_argument('--records',type=int,default=100);a=p.parse_args()
items=read_json(a.catalog)['items'];records=[]
for item in items:
    for aspect,feature in item['features'].items():
        records.append({'item_id':item['id'],'aspect':aspect,'value':feature['value'],'source_span':feature['span'],
                        'source_text':item['text'],'supported_0_or_1':'','annotator':'','comment':''})
random.Random(11).shuffle(records);records=records[:a.records]
if not records:raise ValueError('No accepted features to audit; report extraction failure')
with open(a.out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print({'records':len(records),'output':a.out,'note':'Copy for each independent annotator; source containment is not semantic entailment.'})
