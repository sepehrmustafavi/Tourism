"""Paired, resumable experiments with no access to test labels during decisions."""
from dataclasses import dataclass, asdict, replace
from itertools import combinations
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
from scipy.special import expit
from .core import Posterior, Observation, build_actions, list_scores, query_value, shortlist
from .data import Catalog, read_json, write_json, digest

METHODS = ("full","random_matched","entropy_matched","disagreement_matched", "ucb_matched",
           "no_questions","single_utility","independent","shared_transfer","travel_only",
           "no_cost","no_rejuvenation","no_diversity","bt_random","bt_no_questions","popularity")

@dataclass(frozen=True)
class Config:
    particles: int=512
    actions: int=64
    candidates: int=24
    shortlist: int=12
    pair_pool: int=48
    judgment_budget: int | None=None
    budget: int=5
    k: int=5
    tau: float=1.
    delta: float=.2
    sigma: float=1.
    evaluation_cost: float=.002
    cost: float=.002
    response_tau: float=1.
    response_delta: float=.2
    response_skip: float=.15
    skip: float=.15
    rejuvenation: int=2
    weights: tuple=(.25,.6,.15)
    context: tuple=(3.,0.)
    purpose: str="planning"
    require_verified: bool=False

    def __post_init__(self):
        sizes=(self.particles,self.actions,self.candidates,self.shortlist,self.pair_pool,self.k)
        if any(type(x) is not int or x<1 for x in sizes) or self.particles<2:
            raise ValueError('Positive integer sizes and at least two particles required')
        if type(self.budget) is not int or self.budget<0 or self.rejuvenation<0:
            raise ValueError('Invalid query budget or rejuvenation count')
        if self.judgment_budget is not None and (type(self.judgment_budget) is not int or self.judgment_budget<0):
            raise ValueError('Invalid judgment budget')
        values=(self.tau,self.delta,self.sigma,self.response_tau,self.response_delta)
        if any(not np.isfinite(x) or x<=0 for x in values):
            raise ValueError('Positive finite response/prior scales required')
        if any(not np.isfinite(x) or not 0<=x<=1 for x in (self.skip,self.response_skip)):
            raise ValueError('Invalid skip probability')
        if any(not np.isfinite(x) or x<0 for x in (self.cost,self.evaluation_cost)):
            raise ValueError('Invalid judgment cost')
        w=np.asarray(self.weights,float)
        if w.shape!=(3,) or not np.all(np.isfinite(w)) or np.any(w<0) or not np.isclose(w.sum(),1):
            raise ValueError('Confirmed simplex weights required')
        if len(self.context)!=2 or not np.all(np.isfinite(self.context)) or self.purpose not in {'planning','browsing'}:
            raise ValueError('Invalid confirmed context/purpose')


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16],16))


def environment():
    info={"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__}
    try:
        import scipy
        info["scipy"]=scipy.__version__
        info["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except (ImportError,subprocess.CalledProcessError): pass
    try:
        info["nvidia_smi"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],text=True,stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError,subprocess.CalledProcessError): info["nvidia_smi"]=None
    try:
        import torch
        info.update(torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())
    except ImportError: pass
    return info


def split_catalog(catalog, split):
    """Destination-disjoint split prevents two cards of one place crossing splits."""
    bounds={"train":(0,60),"validation":(60,80),"test":(80,100)}
    if split not in bounds: raise ValueError("Unknown split")
    lo,hi=bounds[split]
    indices=[i for i,d in enumerate(catalog.destinations)
             if lo <= int(hashlib.sha256(d.encode()).hexdigest()[:8],16)%100 < hi]
    if len(set(catalog.destinations[i] for i in indices)) < 2:
        raise ValueError("Split too small; use a larger catalog")
    return catalog.subset(indices)


def episode_catalog(catalog, user, seed, cfg):
    rng=rng_for("catalog",user,seed)
    feasible=catalog.feasible(cfg.context,cfg.purpose,cfg.require_verified)
    if not feasible: raise ValueError("No feasible destinations; explicit relaxation required")
    # One card per city for evaluation episodes; alternate card selected reproducibly.
    groups={}
    for i in feasible: groups.setdefault(catalog.destinations[i],[]).append(i)
    dests=sorted(groups)
    selected=rng.choice(len(dests),size=min(cfg.candidates,len(dests)),replace=False)
    indices=[int(rng.choice(groups[dests[j]])) for j in selected]
    return catalog.subset(indices)


def truth(catalog, user, seed, regime, cfg):
    """Independent truth draw, including deliberately misspecified nonlinear users."""
    rng=rng_for("truth",user,seed,regime)
    p=Posterior(catalog.phi.shape[1],catalog.psi.shape[1],4,particles=2,
                seed=int(rng.integers(2**31)),tau=cfg.tau,delta=cfg.delta)
    s=p.state[:1].copy(); d=p.d
    if regime=="aligned": s[:,p.real_dim:]=5.
    elif regime=="conflict":
        s[:,d:2*d]=-s[:,:d]+rng.normal(0,.3,(1,d)); s[:,p.real_dim:]=-3.
    elif regime not in {"mixed","nonlinear"}: raise ValueError("Unknown truth regime")
    v,t=p.utilities(catalog.phi,catalog.psi,catalog.context(cfg.context),s)
    if regime=="nonlinear":
        # Outside assumed linear family and response scale; no latent gate recovery claim.
        x=catalog.phi
        v += 1.5*np.sin(2*x[:,0])[None,:]
        t += 2*(x[:,1]*x[:,2])[None,:] - 1.2*x[:,0]**2
    return v,t


def simulated_answer(v,t,pair,target,user,seed,regime,cfg):
    i,j=pair; u=(v,t)[target=="T"]
    delta=float(u[0,i]-u[0,j]); tau=cfg.response_tau*(1.4 if regime=="nonlinear" else 1)
    a,b=expit((delta-cfg.response_delta)/tau),expit((-delta-cfg.response_delta)/tau)
    probs=np.array([(1-cfg.response_skip)*a,(1-cfg.response_skip)*b,(1-cfg.response_skip)*max(0,1-a-b),cfg.response_skip])
    rng=rng_for("answer",user,seed,regime,pair,target)
    return str(rng.choice(("A","B","tie","skip"),p=probs/probs.sum()))


def ndcg(scores, relevance, k):
    order=np.argsort(-np.asarray(scores),kind="stable")[:k]
    ideal=np.sort(relevance)[::-1][:len(order)]
    discount=1/np.log2(np.arange(len(order))+2)
    den=ideal@discount
    return float(np.asarray(relevance)[order]@discount/den) if den>0 else 0.


def run_episode(catalog, cfg, method, user=0, seed=11, regime="mixed", human=None):
    if method not in METHODS: raise ValueError("Unknown method")
    if human is None:
        cat=episode_catalog(catalog,user,seed,cfg)
    else:
        lookup={item["id"]:i for i,item in enumerate(catalog.items)}
        cat=catalog.subset([lookup[x] for x in human["item_ids"]])
        cfg=replace(cfg,context=tuple(human["context"]),weights=tuple(human["weights"]))
        if set(cat.feasible(cfg.context,cfg.purpose,cfg.require_verified)) != set(range(len(cat.items))):
            raise ValueError("Human episode contains infeasible candidates")
    chi=cat.context(cfg.context)
    # The same frozen action pool is used by every method and hypothetical answer.
    actions=build_actions(cat.phi,cat.psi,chi,cat.destinations,cat.distance,
                          range(len(cat.items)),cfg.k,cfg.actions,seed=19)
    if not actions: raise ValueError("No feasible actions")
    pair_rng=rng_for("pairs",user,seed)
    pairs=list(combinations(range(len(cat.items)),2)); pair_rng.shuffle(pairs)
    warm=pairs[:min(6,len(pairs)//4)]
    holdout=pairs[len(warm):len(warm)+min(12,len(pairs)//4)]
    pool=pairs[len(warm)+len(holdout):][:cfg.pair_pool]
    if human is None:
        tv,tt=truth(cat,user,seed,regime,cfg)
        answer=lambda pair,target:simulated_answer(tv,tt,pair,target,user,seed,regime,cfg)
        relevance=(expit(tv[0]),expit(tt[0]))
        oracle_j=list_scores(tv,tt,actions,cat.distance,cfg.weights)[0][0]
    else:
        warm=[tuple(p) for p in human["warmup_pairs"]]
        holdout=[tuple(p) for p in human["test_pairs"]]
        pool=[tuple(p) for p in human["query_pairs"]]
        if set(warm)&set(holdout) or set(pool)&set(holdout) or set(warm)&set(pool):
            raise ValueError("Human pair leakage")
        answer=lambda pair,target:human["answers"][f"{pair[0]}:{pair[1]}:{target}"]
        relevance=tuple(np.asarray(human["ratings"][t],float)/4 for t in ("V","T"))
        # This is empirical stated-rating utility, not true latent psychological utility.
        lv=[np.log(np.clip(r,1e-5,1-1e-5)/(1-np.clip(r,1e-5,1-1e-5)))[None,:] for r in relevance]
        oracle_j=list_scores(*lv,actions,cat.distance,cfg.weights)[0][0]
    mode={"independent":"independent","single_utility":"single","shared_transfer":"shared"}.get(method,"transfer")
    p=Posterior(cat.phi.shape[1],cat.psi.shape[1],chi.shape[1],cfg.particles,seed,
        sigma=cfg.sigma,tau=cfg.tau,delta=cfg.delta,mode=mode,
        rejuvenation=0 if method=="no_rejuvenation" else cfg.rejuvenation)
    def assimilate(pair,target,ans):
        p.update(Observation(f"{pair[0]}:{pair[1]}:{target}",*pair,target,ans,tuple(cfg.context)),
                 cat.phi,cat.psi,cat.context)
    for pair in warm: assimilate(pair,"V",answer(pair,"V"))
    costs=[]; rows=[]; attempts=[]; judgments=0; answered=0
    if method=='no_diversity' and sum(cfg.weights[:2])<=0:
        raise ValueError('No-diversity ablation requires a positive relevance priority')
    weights=cfg.weights if method!="no_diversity" else tuple(np.array(cfg.weights)*[1,1,0]/sum(np.array(cfg.weights)*[1,1,0]))
    decision_start=time.perf_counter()
    for turn in range(cfg.budget+1):
        v,t=p.utilities(cat.phi,cat.psi,chi)
        if method in {"bt_random","bt_no_questions"}:
            from .baselines import map_bradley_terry
            v,t=map_bradley_terry(p,cat,cfg.context)
        elif method=="popularity":
            pop={"low":.2,"medium":.5,"high":.8}
            scores=np.array([pop.get(str(item["attributes"].get("popularity","")).lower(),.5) for item in cat.items])
            logits=np.log(scores/(1-scores))
            v=t=np.broadcast_to(logits,(len(p.weights),len(logits))).copy()
        j,fv,ft,div=list_scores(v,t,actions,cat.distance,weights)
        best=int(np.argmax(p.weights@j)); chosen=actions[best]
        row={"turn":turn,"method":method,"user":str(user),"seed":seed,"regime":regime,
            "evidence":"human_stated_preference" if human else "simulation",
            "catalog_hash":cat.hash,"action_hash":digest(actions),"context":list(cfg.context),
            "weights":list(cfg.weights),"list_ids":[cat.items[i]["id"] for i in chosen],
            "utility":float(oracle_j[best]),"regret":float(oracle_j.max()-oracle_j[best]),
            "diversity":float(div[best]),"questions":len(attempts),"judgments_requested":judgments,
            "judgments_answered":answered,"ess":p.ess,"resamples":p.resamples,
            "decision_seconds":time.perf_counter()-decision_start,
            "posterior_regret":float(p.weights@j.max(axis=1)-(p.weights@j).max()),
            "net_utility":float(oracle_j[best]-cfg.evaluation_cost*judgments),
            "mh_acceptance":p.mh_acceptances/max(1,p.mh_proposals),
            "constraint_violations":0, # All admitted actions explicitly validated below.
            "destination_duplicates":len(chosen)-len({cat.destinations[i] for i in chosen})}
        valid=set(cat.feasible(cfg.context,cfg.purpose,cfg.require_verified))
        row["constraint_violations"]=sum(i not in valid for i in chosen)
        for ti,target in enumerate(("V","T")):
            prediction=p.weights@expit((v,t)[ti])
            # List NDCG uses the actual selected order, not a separate relevance ranking.
            rel=relevance[ti]; rho=1/np.log2(np.arange(len(chosen))+2)
            den=np.sort(rel)[::-1][:len(chosen)]@rho
            row[f"ndcg_{target}"]=float(rel[list(chosen)]@rho/den) if den else 0.
            row[f"score_rmse_{target}"]=float(np.sqrt(np.mean((prediction-rel)**2)))
            nll=[]; brier=[]; correct=[]
            for pair in holdout:
                ans=answer(pair,target)
                if ans=="skip": continue
                probs=p.weights@p.comparison((v,t)[ti][:,pair[0]]-(v,t)[ti][:,pair[1]])
                label=("A","B","tie").index(ans)
                nll.append(-np.log(max(probs[label],1e-15)))
                brier.append(np.square(probs-np.eye(3)[label]).sum())
                correct.append(int(np.argmax(probs)==label))
            row[f"nll_{target}"]=float(np.mean(nll)) if nll else None
            row[f"brier_{target}"]=float(np.mean(brier)) if brier else None
            row[f"accuracy_{target}"]=float(np.mean(correct)) if correct else None
            row[f"heldout_answered_{target}"]=len(nll)
            row[f"heldout_skipped_{target}"]=len(holdout)-len(nll)
        row["trace"]={"history_event_ids":[o.event_id for o in p.history],
            "sources":[{"id":cat.items[i]["id"],"source":cat.items[i]["source_id"],
                        "features":cat.items[i]["features"]} for i in chosen],
            "viewing_contribution":float(weights[0]*(p.weights@fv)[best]),
            "travel_contribution":float(weights[1]*(p.weights@ft)[best]),
            "diversity_contribution":float(weights[2]*div[best])}
        rows.append(row)
        if turn==cfg.budget or method in {"no_questions","bt_no_questions","popularity"} or not pool: break
        targets=("T",) if method=="travel_only" else ("V","T")
        if cfg.judgment_budget is not None and judgments+len(targets)>cfg.judgment_budget: break
        tic=time.perf_counter()
        if method in {"random_matched","bt_random"}: pair=pool[int(rng_for("policy",user,seed,turn).integers(len(pool)))]; value=None
        elif method in {"disagreement_matched","ucb_matched"}:
            if method=="disagreement_matched": pair=shortlist(p,cat.phi,cat.psi,chi,pool,1)[0]
            else:
                # An explicit UCB heuristic on pair differences; not a reproduction of PEBOL.
                means=[p.weights@(t[:,i]-t[:,jj]) for i,jj in pool]
                std=[np.sqrt(p.weights@((t[:,i]-t[:,jj])-mu)**2) for (i,jj),mu in zip(pool,means)]
                pair=pool[int(np.argmax(np.abs(means)+np.asarray(std))) ]
            value=None
        else:
            short=shortlist(p,cat.phi,cat.psi,chi,pool,cfg.shortlist)
            values=[]
            for pair0 in short:
                likelihood=p.outcome_probs(cat.phi,cat.psi,chi,*pair0,targets,cfg.skip)
                if method=="entropy_matched":
                    marginal=p.weights@likelihood
                    values.append(float(-np.sum(marginal*np.log(np.maximum(marginal,1e-15)))))
                else:
                    burden=0 if method=="no_cost" else cfg.cost*len(targets)
                    values.append(query_value(j,p.weights,likelihood,burden))
            idx=int(np.argmax(values)); pair,value=short[idx],values[idx]
            if method!="entropy_matched" and value<=0:
                costs.append(time.perf_counter()-tic)
                break
        query_seconds=time.perf_counter()-tic; costs.append(query_seconds)
        responses={target:answer(pair,target) for target in targets}
        for target,ans in responses.items(): assimilate(pair,target,ans)
        attempts.append({"pair":list(pair),"responses":responses,"net_evsi":value,"seconds":query_seconds})
        judgments+=len(targets); answered+=sum(ans!="skip" for ans in responses.values())
        pool.remove(pair) # Attempted skips cannot create loops.
    return {"rows":rows,"queries":attempts,"history":[asdict(o) for o in p.history],
            "episode_item_ids":[item["id"] for item in cat.items],
            "actions_ids":[[cat.items[i]["id"] for i in action] for action in actions],
            "action_reference_utility":oracle_j.tolist(),"latency_p50":float(np.median(costs)) if costs else None,
            "latency_p95":float(np.quantile(costs,.95)) if costs else None,
            "config":asdict(cfg),"environment":environment()}


def run_suite(catalog_path,out,config,methods,seeds,users,regimes,split="test",max_seconds=None):
    if users<1 or not methods or not seeds or not regimes:
        raise ValueError('A nonempty experiment and positive user count are required')
    if len(set(methods))!=len(methods) or len(set(seeds))!=len(seeds) or len(set(regimes))!=len(regimes):
        raise ValueError('Duplicate experiment cells')
    if any(m not in METHODS for m in methods) or any(r not in {'aligned','mixed','conflict','nonlinear'} for r in regimes):
        raise ValueError('Unknown method or simulation regime')
    cat=split_catalog(Catalog(read_json(catalog_path)),split)
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    signature={"catalog":cat.hash,"config":asdict(config),"methods":methods,"seeds":seeds,
               "users":users,"regimes":regimes,"split":split,"implementation":"v1"}
    # Include source hash so stale output cannot be reused after code changes.
    signature["code_hash"]=digest({p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')})
    manifest=root/"manifest.json"
    if manifest.exists() and read_json(manifest)["signature"]!=json.loads(json.dumps(signature)):
        raise ValueError("Output belongs to different code/config/data; choose a new output directory")
    write_json(manifest,{"signature":signature,"environment":environment()})
    started=time.perf_counter(); completed=0
    for seed in seeds:
        for regime in regimes:
            for user in range(users):
                for method in methods:
                    target=root/f"{regime}_{seed}_{user}_{method}.json"
                    if target.exists(): continue
                    if max_seconds and time.perf_counter()-started>=max_seconds:
                        return {"completed_now":completed,"paused":True}
                    result=run_episode(cat,config,method,user,seed,regime)
                    write_json(target,result); completed+=1
    return {"completed_now":completed,"paused":False}
