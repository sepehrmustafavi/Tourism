"""Publication exports from measured aggregate curves only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tourism.data import read_json
p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--out',required=True);a=p.parse_args()
curves=read_json(a.summary)['curves'];root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
for evidence,regime in sorted({(r['evidence'],r['regime']) for r in curves}):
    for metric in ('net_utility','regret','ndcg_T'):
        fig,ax=plt.subplots(figsize=(6,4))
        for method in ('full','random_matched','entropy_matched','bt_random','single_utility','no_questions'):
            points=sorted((r for r in curves if r['evidence']==evidence and r['regime']==regime and r['metric']==metric and r['method']==method),key=lambda r:r['budget'])
            if points:ax.plot([r['budget'] for r in points],[r['mean'] for r in points],marker='o',ms=3,label=method)
        ax.set(xlabel='Maximum clarification queries',ylabel=metric.replace('_',' '),title=f'{evidence}: {regime}')
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False);fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(root/f'{evidence}_{regime}_{metric}.{ext}',dpi=200)
        plt.close(fig)
