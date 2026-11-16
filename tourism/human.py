"""Offline blinded research instrument and validated replay, with no hosting."""
from itertools import combinations
from pathlib import Path
import html
import json
import numpy as np
from .data import Catalog, read_json, write_json, digest
from .experiment import Config, episode_catalog, rng_for, run_episode, METHODS


def make_survey(catalog_path,out,participant,seed=11):
    cat=Catalog(read_json(catalog_path))
    cfg=Config(candidates=12,k=5)
    cat=episode_catalog(cat,participant,seed,cfg)
    pairs=list(combinations(range(len(cat.items)),2)); rng=rng_for('survey',participant,seed); rng.shuffle(pairs)
    warm=pairs[:6]; test=pairs[6:18]; query=pairs[18:42]
    protocol={"protocol":"human-v1","participant":participant,"catalog_hash":Catalog(read_json(catalog_path)).hash,
        "item_ids":[i['id'] for i in cat.items],"context":list(cfg.context),"weights":list(cfg.weights),
        "warmup_pairs":warm,"test_pairs":test,"query_pairs":query,"answers":{},
        "ratings":{"V":[],"T":[]},"consent":False,"completed":False}
    cards=[{"text":i['text'],"destination":i['destination'],"attributes":i['attributes']} for i in cat.items]
    tasks=[]
    for kind,ps in [('warmup',warm),('query',query),('test',test)]:
        for pair in ps:
            targets=['V'] if kind=='warmup' else ['V','T']
            if len(targets)>1 and rng.random()<.5: targets.reverse()
            swapped=bool(rng.integers(2))
            tasks.append({"kind":kind,"pair":pair,"targets":targets,"swap":swapped})
    # Block order of query/test judgments is randomized; labels stay isolated by role.
    tail=tasks[len(warm):]; rng.shuffle(tail); tasks=tasks[:len(warm)]+tail
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    write_json(root/f'{participant}_protocol.json',protocol)
    data=json.dumps({"protocol":protocol,"cards":cards,"tasks":tasks},ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>Tourism preference study</title>
<style>body{font:17px system-ui;max-width:900px;margin:40px auto;padding:20px;color:#24302b;background:#faf9f5}article{white-space:pre-wrap;padding:20px;background:white;margin:10px 0;border:1px solid #ddd}button,select,input{font:inherit;padding:10px;margin:8px}label{display:block}small{color:#555}</style>
<h1>Tourism content and travel preferences</h1><div id="screen"></div>
<script>const DATA=__DATA__;let P=DATA.protocol, stage=-1, start=Date.now(); P.response_log=[];
const S=document.getElementById('screen');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function card(i){const c=DATA.cards[i];return '<article><b>'+esc(c.destination)+'</b><p>'+esc(c.text)+'</p><small>'+esc(JSON.stringify(c.attributes))+'</small></article>'}
function initial(){S.innerHTML='<p>This study compares interest in viewing tourism cards with stated interest in visiting their destinations. These are source-derived text cards, not social-media videos. Budget categories are relative source attributes, not verified current trip prices.</p><p>No name or contact information is requested. Participation is voluntary; skip or stop at any time. Responses stay in this browser until you choose to download them and share them with the researcher. Do not enter personal information.</p><label><input id="consent" type="checkbox">I am 18 or older and consent to take part.</label><p>Confirm a hypothetical trip context: destinations up to <b>High</b> relative budget, no minimum walkability requirement. Rate priorities; they will be normalized once and kept fixed.</p><label>Viewing priority <input id="wv" type="number" min="0" value="25"></label><label>Travel priority <input id="wt" type="number" min="0" value="60"></label><label>Diversity priority <input id="wd" type="number" min="0" value="15"></label><button onclick="begin()">Begin</button>'}
function begin(){if(!document.getElementById('consent').checked)return;let w=['wv','wt','wd'].map(x=>Number(document.getElementById(x).value));if(w.some(x=>!Number.isFinite(x)||x<0)||w.reduce((a,b)=>a+b,0)<=0)return;let sum=w.reduce((a,b)=>a+b,0);P.weights=w.map(x=>x/sum);P.consent=true;P.started_at=new Date().toISOString();next()}
function next(){stage++;start=Date.now();if(stage<DATA.tasks.length){let t=DATA.tasks[stage], shown=t.swap?[t.pair[1],t.pair[0]]:t.pair;S.innerHTML='<p>Comparison '+(stage+1)+' / '+DATA.tasks.length+'</p><h2>Card A</h2>'+card(shown[0])+'<h2>Card B</h2>'+card(shown[1])+t.targets.map(r=>'<label>'+(r==='V'?'Which card would you rather view?':'Which destination would you rather consider visiting under the confirmed trip context?')+'<select id="'+r+'"><option value="">Choose</option><option>A</option><option>B</option><option>tie</option><option>skip</option></select></label>').join('')+'<button onclick="savePair()">Continue</button>';return}
let i=stage-DATA.tasks.length;if(i<DATA.cards.length){S.innerHTML='<h2>Independent rating '+(i+1)+' / '+DATA.cards.length+'</h2>'+card(i)+['V','T'].map(r=>'<label>'+(r==='V'?'Interest in viewing this card':'Interest in considering this destination for the confirmed trip')+'<select id="'+r+'"><option value="">Choose</option>'+[0,1,2,3,4].map(x=>'<option>'+x+'</option>').join('')+'</select></label>').join('')+'<p>0 = none, 4 = very high. You may stop instead of answering.</p><button onclick="saveRating('+i+')">Continue</button>';return}
P.completed=true;P.completed_at=new Date().toISOString();P.elapsed_seconds=(Date.now()-Date.parse(P.started_at))/1000;S.innerHTML='<p>Complete. Download your responses, then share the file with the researcher if you wish.</p><button onclick="download()">Download responses</button>';}
function savePair(){let t=DATA.tasks[stage], ans={};for(let r of t.targets){let a=document.getElementById(r).value;if(!a)return;if(t.swap&&a==='A')a='B';else if(t.swap&&a==='B')a='A';ans[r]=a;}for(let r in ans)P.answers[t.pair[0]+':'+t.pair[1]+':'+r]=ans[r];P.response_log.push({stage,kind:t.kind,seconds:(Date.now()-start)/1000,swapped:t.swap,target_order:t.targets});next()}
function saveRating(i){for(let r of ['V','T']){let x=document.getElementById(r).value;if(x==='')return;}for(let r of ['V','T'])P.ratings[r][i]=Number(document.getElementById(r).value);P.response_log.push({stage,kind:'rating',seconds:(Date.now()-start)/1000});next()}
function download(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(P,null,2)],{type:'application/json'}));a.download=P.participant+'_responses.json';a.click();URL.revokeObjectURL(a.href)}initial();</script>'''.replace('__DATA__',data)
    (root/f'{participant}.html').write_text(page)
    return {"html":str(root/f'{participant}.html'),"protocol":str(root/f'{participant}_protocol.json')}


def validate_human(record,catalog,protocol=None):
    if not record.get('consent') or not record.get('completed'): raise ValueError('Incomplete or unconsented record')
    if record['catalog_hash']!=catalog.hash: raise ValueError('Catalog version changed')
    if protocol:
        for key in ('protocol','participant','catalog_hash','item_ids','context','warmup_pairs','query_pairs','test_pairs'):
            if record[key]!=protocol[key]: raise ValueError('Protocol altered: '+key)
    ids=record['item_ids']; n=len(ids)
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate items')
    groups=[]
    for key in ('warmup_pairs','query_pairs','test_pairs'):
        ps=[tuple(x) for x in record[key]]
        if any(not (0<=i<j<n) for i,j in ps) or len(ps)!=len(set(ps)): raise ValueError('Invalid pairs')
        groups.append(set(ps))
    if any(groups[i]&groups[j] for i,j in combinations(range(3),2)): raise ValueError('Pair leakage')
    for key,targets in (('warmup_pairs',('V',)),('query_pairs',('V','T')),('test_pairs',('V','T'))):
        for i,j in record[key]:
            for target in targets:
                if record['answers'].get(f'{i}:{j}:{target}') not in ('A','B','tie','skip'): raise ValueError('Missing or invalid answer')
    for target in ('V','T'):
        ratings=record['ratings'][target]
        if len(ratings)!=n or any(type(r) not in (int,float) or not np.isfinite(r) or not 0<=r<=4 for r in ratings): raise ValueError('Invalid ratings')
    w=np.array(record['weights'],float)
    if w.shape!=(3,) or not np.all(np.isfinite(w)) or min(w)<0 or not np.isclose(w.sum(),1): raise ValueError('Unconfirmed weights')
    if len(record['context'])!=2 or not np.all(np.isfinite(record['context'])): raise ValueError('Invalid context')
    return True


def replay(catalog_path,response_path,protocol_path,out,config,methods,seeds):
    cat=Catalog(read_json(catalog_path)); human=read_json(response_path)
    validate_human(human,cat,read_json(protocol_path))
    for seed in seeds:
        for method in methods:
            path=Path(out)/f"human_{human['participant']}_{seed}_{method}.json"
            code_hash=digest({p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')})
            signature=digest([cat.hash,human,config.__dict__,method,seed,code_hash])
            if path.exists():
                if read_json(path).get('signature')!=signature: raise ValueError('Stale human output')
                continue
            result=run_episode(cat,config,method,human['participant'],seed,'human',human)
            result['signature']=signature; write_json(path,result)
    return {"participant":human['participant'],"runs":len(methods)*len(seeds)}
