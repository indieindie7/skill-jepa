"""ASTATE/MIN claim store (user capsule 11-03, schema v1): sources [id, sha256, data?], claims [id, s, p, o, e:[[src, start, len]+]], conflicts [[claim_id+]].
MIN-M ids = content-derived (sha256 of canonical s|p|o, 12 hex) so stores from different chats merge without dupes. validate() checks the invariants
(>=1 span, span in range, source hash, span bytes contain o, conflicts ref real claims); integrity != truth."""
import hashlib, json, re

def cid(s, p, o): return hashlib.sha256(json.dumps([s, p, o], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:12]
def src(path):
    b = open(path, "rb").read(); return {"id": path, "sha256": hashlib.sha256(b).hexdigest(), "_b": b}
def claim(s, p, o, spans): return {"id": cid(s, p, o), "s": s, "p": p, "o": o, "e": spans}

def validate(store):
    """-> list of (claim_id or '-', problem). Empty = all invariants hold."""
    bad, S = [], {x["id"]: x for x in store["sources"]}
    for x in S.values():
        b = x.get("_b") or open(x["id"], "rb").read()
        if hashlib.sha256(b).hexdigest() != x["sha256"]: bad.append(("-", f"source changed: {x['id']}"))
        x["_b"] = b
    ids = set()
    for c in store["claims"]:
        ids.add(c["id"])
        if c["id"] != cid(c["s"], c["p"], c["o"]): bad.append((c["id"], "id != content hash"))
        if not c["e"]: bad.append((c["id"], "no evidence span")); continue
        for sid, st, ln in c["e"]:
            b = S[sid]["_b"] if sid in S else None
            if b is None or st < 0 or st + ln > len(b): bad.append((c["id"], f"span out of range {sid}:{st}+{ln}")); continue
            if str(c["o"]).encode() not in b[st:st + ln]: bad.append((c["id"], f"span lacks value {c['o']!r}"))
    for k in store.get("conflicts", []):
        if not set(k) <= ids: bad.append(("-", f"conflict refs unknown claim {k}"))
    return bad

def merge(*stores):
    """MIN-M merge: same content id = same claim, evidence spans unioned; same (s, p) with different o -> conflict."""
    out = {"version": 1, "sources": {}, "claims": {}, "conflicts": []}
    for st in stores:
        for x in st["sources"]: out["sources"][x["id"]] = x
        for c in st["claims"]:
            k = out["claims"].setdefault(c["id"], {**c, "e": []}); k["e"] += [e for e in c["e"] if e not in k["e"]]
    by = {}
    for c in out["claims"].values(): by.setdefault((c["s"], c["p"]), []).append(c["id"])
    out["conflicts"] = [v for v in by.values() if len(v) > 1]
    return {**out, "sources": list(out["sources"].values()), "claims": list(out["claims"].values())}

def support(store, s, p, o, near=None):
    """Is the claim (s, p, o) backed? near(source_bytes, s) -> [(start, end)] scope windows for s; o must occur inside one. -> spans list ([] = unsourced)."""
    out = []
    for x in store["sources"]:
        b = x["_b"]; wins = near(b, s) if near else [(0, len(b))]
        for a, z in wins:
            for m in re.finditer(re.escape(str(o).encode()), b[a:z]): out.append([x["id"], a + m.start(), len(str(o).encode())])
    return out
