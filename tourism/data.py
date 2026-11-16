"""Versioned real-catalog adapter, evidence spans and explicit missingness."""
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from collections import defaultdict
import numpy as np

ASPECTS = ("nature", "heritage", "food", "nightlife", "art", "shopping")
KEYWORDS = {
 "nature": ("park", "garden", "beach", "mountain", "lake", "forest"),
 "heritage": ("castle", "historic", "medieval", "cathedral", "palace", "roman"),
 "food": ("restaurant", "food", "cuisine", "market", "cafe", "café"),
 "nightlife": ("nightclub", "nightlife", "bar", "pub", "music"),
 "art": ("museum", "gallery", "art", "theatre", "theater"),
 "shopping": ("shopping", "shop", "boutique", "mall"),
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)

def read_json(path): return json.loads(Path(path).read_text())

def fetch_synthtrips(out, revision="main", max_cities=200):
    # Resolve the mutable revision once, then download only by immutable commit.
    api = f"https://huggingface.co/api/datasets/ashmib/SynthTRIPs/revision/{revision}"
    with urllib.request.urlopen(api, timeout=90) as r: info = json.load(r)
    sha = info["sha"]
    url = f"https://huggingface.co/datasets/ashmib/SynthTRIPs/resolve/{sha}/kb/eu-cities-database.csv"
    with urllib.request.urlopen(url, timeout=120) as r: raw = r.read()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    grouped = defaultdict(list)
    for index,row in enumerate(rows):
        grouped[(row["country"], row["city"])].append((index,row))
    items = []
    for (country,city), records in sorted(grouped.items())[:max_cities]:
        selected = records[:8]
        text = "\n".join(r["interest_text"] for _,r in selected)
        base = records[0][1]
        # Source-derived cards, not observed social-media posts or video features.
        for variant in (0,1):
            item_id = digest([country,city,variant])[:16]
            post = (f"{city}, {country}\n{text}" if variant == 0 else
                    f"Explore {city}\n"+"\n".join(r["interest_text"] for _,r in reversed(selected)))
            items.append({"id":item_id, "destination":f"{country}/{city}", "country":country,
                "text":post, "source_id":url, "source_version":sha,
                "source_rows":[i+2 for i,_ in selected],
                "card_variant":variant, "verified":False,
                "attributes":{"budget":base.get("budget") or None,
                    "walkability":base.get("walkability") or None,
                    "popularity":base.get("popularity") or None},
                "features":{}, "feature_source":"none"})
    payload = {"kind":"source_derived_tourism_cards", "source":url,
        "revision":sha, "sha256":hashlib.sha256(raw).hexdigest(),
        "license":"MIT per source dataset card; verify underlying source terms before redistribution",
        "items":items}
    write_json(out, payload)
    return payload

def validate_features(item, features):
    """Span containment validates provenance, NOT semantic entailment. Audit separately."""
    accepted = {}
    for name,record in features.items():
        if name not in ASPECTS or not isinstance(record,dict): continue
        span, value = record.get("span"), record.get("value")
        if (isinstance(span,str) and span.strip() and span in item["text"]
            and isinstance(value,(int,float)) and not isinstance(value,bool)
            and np.isfinite(value) and 0 <= value <= 1):
            accepted[name] = {"value":float(value),"span":span,
                "source_id":item["source_id"], "version":item["source_version"]}
    return accepted

def lexical_features(item):
    result = {}
    text = item["text"]; lower = text.lower()
    for aspect, words in KEYWORDS.items():
        for word in words:
            pos = lower.find(word)
            if pos >= 0:
                # A present keyword is a weak deterministic feature, not a negative label for absence.
                result[aspect] = {"value":1., "span":text[pos:pos+len(word)]}; break
    return validate_features(item,result)

class Catalog:
    def __init__(self, payload):
        self.payload = payload; self.items = payload["items"]
        ids = [i["id"] for i in self.items]
        if len(ids) != len(set(ids)): raise ValueError("Duplicate catalog identity")
        self.destinations = [i["destination"] for i in self.items]
        x = np.zeros((len(ids),len(ASPECTS))); mask = np.zeros_like(x)
        for j,item in enumerate(self.items):
            if item.get("features") != validate_features(item,item.get("features",{})):
                raise ValueError("Unvalidated feature record")
            for k,name in enumerate(ASPECTS):
                if name in item.get("features",{}):
                    x[j,k] = item["features"][name]["value"]; mask[j,k] = 1
        self.phi = x*mask
        # Presentation is known card order only. Never infer image quality from text.
        self.psi = np.column_stack(([2*i.get("card_variant",0)-1 for i in self.items],mask))
        # Grounded coarse taxonomy = country + source-linked aspect availability.
        self.distance = np.zeros((len(ids),len(ids)))
        for i in range(len(ids)):
            for j in range(i):
                union = np.maximum(mask[i],mask[j]).sum()
                jac = 1-np.minimum(mask[i],mask[j]).sum()/union if union else 0.
                self.distance[i,j] = self.distance[j,i] = .5*jac+.5*(self.items[i]["country"]!=self.items[j]["country"])
        self.hash = digest(payload)

    def context(self, context):
        # Confirmed budget cap and walkability request. Keep original observations' context.
        budget_cap, walk_min = context
        budget_map = {"low":1.,"medium":2.,"high":3.}
        walk_map = {"poor":0.,"bad":0.,"fair":1.,"okay":1.,"good":2.,"great":3.,"excellent":3.}
        z = np.zeros((len(self.items),2)); mask = np.zeros_like(z)
        for j,item in enumerate(self.items):
            a = item["attributes"]
            b = budget_map.get(str(a.get("budget","")).lower())
            w = walk_map.get(str(a.get("walkability","")).lower())
            if b is not None: z[j,0]=(budget_cap-b)/3; mask[j,0]=1
            if w is not None: z[j,1]=(w-walk_min)/3; mask[j,1]=1
        return np.column_stack((z*mask,mask))

    def feasible(self, context, purpose="planning", require_verified=False):
        if purpose not in {"planning","browsing"}: raise ValueError("Unknown purpose")
        if purpose == "browsing": return list(range(len(self.items)))
        chi = self.context(context)
        return [i for i in range(len(self.items)) if
                np.all(chi[i,2:] == 1) and np.all(chi[i,:2] >= 0) and
                (not require_verified or self.items[i].get("verified",False))]

    def subset(self, indices):
        return Catalog({**self.payload, "items":[self.items[i] for i in indices]})


def make_synthetic_catalog(n=30, seed=7):
    rng=np.random.default_rng(seed); items=[]
    for i in range(n):
        words=[ASPECTS[k] for k in range(len(ASPECTS)) if rng.random()<.6]
        text=" ".join(words) or "tourism"
        item={"id":str(i),"destination":f"d{i//2}","country":f"c{i//8}",
            "text":text,"source_id":f"synthetic:{i}","source_version":"fixture-v1",
            "card_variant":i%2,"verified":False,
            "attributes":{"budget":"Low","walkability":"Great"},"feature_source":"synthetic"}
        item["features"]=validate_features(item,{w:{"value":float(rng.uniform(.2,1)),"span":w} for w in words})
        items.append(item)
    return {"kind":"synthetic_fixture","items":items}
