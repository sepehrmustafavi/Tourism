"""Paired user-level inference and chapter-aligned exports; no invented results."""
import csv
from pathlib import Path
import numpy as np
from .data import read_json, write_json

METRICS=("utility","net_utility","regret","ndcg_V","ndcg_T","nll_V","nll_T",
         "brier_V","brier_T","accuracy_V","accuracy_T","diversity","questions",
         "judgments_requested","judgments_answered","ess","decision_seconds")
METRICS += ("heldout_answered_V","heldout_answered_T","heldout_skipped_V","heldout_skipped_T")

def bootstrap(values,rng,repeats=2000):
    x=np.asarray(values,float)
    if not len(x): return None
    draws=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(repeats)])
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def paired_test(differences,seed=91,repeats=10000):
    x=np.asarray(differences,float); rng=np.random.default_rng(seed)
    if not len(x): raise ValueError("No paired users")
    observed=abs(x.mean()); extreme=0
    # Monte Carlo sign-flip test; user means are the resampling unit.
    for _ in range(repeats):
        extreme+=abs(np.mean(x*rng.choice([-1,1],len(x))))>=observed-1e-15
    return {"n":len(x),"mean_difference":float(x.mean()),
            "ci95":bootstrap(x,rng),"p":float((extreme+1)/(repeats+1))}


def holm(tests):
    ordered=sorted(range(len(tests)),key=lambda i:tests[i]["p"])
    last=0.
    for rank,i in enumerate(ordered):
        last=max(last,min(1.,(len(tests)-rank)*tests[i]["p"]))
        tests[i]["p_holm"]=last
    return tests


def summarize(directory,out):
    root=Path(directory); dest=Path(out); dest.mkdir(parents=True,exist_ok=True)
    by={}; curves={}; latencies={}; comparable={}
    for path in sorted(root.glob('*.json')):
        result=read_json(path)
        if "rows" not in result: continue
        row=result["rows"][-1]
        # Repeated seeds on one human are averaged BEFORE statistical inference.
        uid=row["user"] if row["evidence"].startswith("human") else f'{row["seed"]}:{row["user"]}'
        group=(row["evidence"],row["regime"],row["method"])
        key=(row['evidence'],row['regime'],row['seed'],row['user'])
        # Mixing different action sets or utility references invalidates paired comparisons.
        signature=tuple(str(row.get(k)) for k in ('catalog_hash','action_hash','context','weights'))
        if key in comparable and comparable[key]!=signature:
            raise ValueError('Incomparable catalog/actions/context/priorities in paired report')
        comparable[key]=signature
        by.setdefault(group,{}).setdefault(uid,[]).append(row)
        for budget in range(result["config"]["budget"]+1):
            point=result["rows"][min(budget,len(result["rows"])-1)]
            curves.setdefault((*group,budget),{}).setdefault(uid,[]).append(point)
        if result.get("latency_p50") is not None:
            latencies.setdefault(group,[]).append(result["latency_p50"])
    summary=[]; user_rows=[]; means={}
    for group,units in by.items():
        evidence,regime,method=group; unit_means={}
        for uid,rows in units.items():
            unit_means[uid]={m:float(np.mean([r[m] for r in rows if r.get(m) is not None]))
                              for m in METRICS if any(r.get(m) is not None for r in rows)}
            user_rows.append({"evidence":evidence,"regime":regime,"method":method,"user":uid,**unit_means[uid]})
        means[group]=unit_means
        for metric in METRICS:
            values=[r[metric] for r in unit_means.values() if metric in r]
            if not values: continue
            summary.append({"evidence":evidence,"regime":regime,"method":method,"metric":metric,
                "n":len(values),"mean":float(np.mean(values)),"sd_users":float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                "ci95":bootstrap(values,np.random.default_rng(91))})
    tests=[]
    # Distinct predeclared endpoints and multiplicity families for each evidence layer.
    for group,units in means.items():
        evidence,regime,method=group
        if method=="full" or (evidence,regime,"full") not in means: continue
        reference=means[(evidence,regime,"full")]
        metric='nll_T' if evidence.startswith('human') else 'net_utility'
        ids=sorted(i for i in set(units)&set(reference) if metric in units[i] and metric in reference[i])
        if not ids: continue
        direction=-1 if metric=='nll_T' else 1
        test=paired_test([direction*(reference[i][metric]-units[i][metric]) for i in ids])
        tests.append({"evidence":evidence,"regime":regime,"comparison":f"full - {method}",
                      "metric":metric,"difference":"positive favors full",
                      "full_users":len(reference),"baseline_users":len(units),**test})
    for evidence in {t['evidence'] for t in tests}:
        holm([t for t in tests if t['evidence']==evidence])
    curve_rows=[]
    for group,units in curves.items():
        evidence,regime,method,budget=group
        for metric in ("utility","net_utility","regret","ndcg_T","questions"):
            values=[np.mean([r[metric] for r in rs]) for rs in units.values()]
            curve_rows.append({"evidence":evidence,"regime":regime,"method":method,"budget":budget,
                               "metric":metric,"mean":float(np.mean(values)),"n":len(values)})
    completeness={'complete':None,'expected_episodes':None,'completed_episodes':sum(len(rs) for us in by.values() for rs in us.values())}
    manifest=root/'manifest.json'
    if manifest.exists():
        spec=read_json(manifest)['signature']
        expected=len(spec['seeds'])*len(spec['regimes'])*spec['users']*len(spec['methods'])
        completeness.update(expected_episodes=expected,complete=completeness['completed_episodes']==expected)
    write_json(dest/'summary.json',{"summary":summary,"paired_tests":tests,"completeness":completeness,
        "latency_episode_medians":{str(k):v for k,v in latencies.items()},"curves":curve_rows})
    for name,rows in (("summary",summary),("paired_tests",tests),("users",user_rows),("curves",curve_rows)):
        if not rows: continue
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (dest/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    text=['# Evaluation results','',
          'Only completed runs are included. Simulation and human stated-preference evidence are separate.',
          'Intervals resample users; repeated human seeds are averaged within participant. Holm families are separate by evidence layer.',
          'Simulation primary: net utility. Human primary: held-out travel NLL. Positive paired differences favor full.',
          f'Completion audit: {completeness}. Partial reports are provisional; comparisons use matched available users.',
          '', '| Evidence | Regime | Method | Users | Net utility | Regret | Travel NDCG | Questions |',
          '|---|---|---|---:|---:|---:|---:|---:|']
    for group,units in means.items():
        def avg(m): return np.mean([x[m] for x in units.values()])
        text.append('| '+' | '.join(map(str,[*group,len(units),*[f'{avg(m):.4f}' for m in ('net_utility','regret','ndcg_T','questions')]]))+' |')
    (dest/'results.md').write_text('\n'.join(text)+'\n')
    return {"groups":len(means),"comparisons":len(tests)}
