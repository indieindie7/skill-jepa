"""teacher_dump.py: dump a soft label distribution from the teacher (Sage-A1.5, llama-server 8093) for distillation into the AVB head (distill-1028, avb_rerank.py --distill).
Rows (cap --cap, default 2500, unique normalised prompts): 1 labelled train prompts, 2 none train prompts (time split first), 3 unlabelled prompts from ops/route.log col 4 (>= 4 words, not '<' system text,
not equal to any pairs.jsonl prompt); order inside a group seeded-shuffled; --reserve rows of the cap are kept for group 3.
Prompt = catalog of the 77 skills (data/skills.json) FIRST (prompt cache prefix), then the request; teacher answers JSON {"p":[[skill,prob]x8],"none":prob}. Final dist over 77+1 classes:
dist_final = (1-EPS)*normalised + EPS/78; stored sparsely: dist = {skill|"none": p} for the teacher's top-8 (+none), rest = per-class mass of every other skill.
Out: data/teacher_dump.jsonl (gitignored), row {q, sid, label (skill|"none"|null), split [tags], dist, rest, raw_top1, agree (labelled rows), secs}. Resumable: prompts already in the file (incl. failed rows) are skipped.
GPU: gpu.pick_gpu (never gpu 0 while a steamapps/games exe holds the GPU; most free VRAM >= 9.5 GB); Sage reused if already up (then not stopped), else started hidden by judge_fresh.start_model and ALWAYS stopped (finally + atexit).
  C:/llm/qlora/venv/Scripts/python.exe teacher_dump.py [--cap 2500] [--limit N] [--stub]     --stub: fake teacher (no model), plumbing test"""
import argparse, atexit, collections, hashlib, json, os, random, re, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "llm"))
OUT = os.path.join(D, "teacher_dump.jsonl"); ROUTE = os.path.join(HERE, "..", "ops", "route.log"); PORT, EPS, TOPN = 8093, 0.02, 8
nz = lambda t: " ".join(t.lower().split())


def build_rows(cap, reserve):
    import skilljepa as sj
    X, Y, lab, allp, sid, names, plab = sj.data("prevskill", "all"); pairs = [json.loads(l) for l in open(os.path.join(D, "pairs.jsonl"), encoding="utf-8")]; n = len(pairs)
    cnt, pos = collections.Counter(sid), np.zeros(n)
    c2 = collections.Counter()
    for i, s in enumerate(sid): pos[i] = c2[s] / cnt[s]; c2[s] += 1
    h = lambda s, salt: int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16) % 100
    te_s = {s for s in set(sid) if h(s, "test") < 30}; va_s = {s for s in set(sid) if s not in te_s and h(s, "val") < 20}
    info = {}
    for i, p in enumerate(pairs):
        tags = ["time_train" if pos[i] < .49 else "time_val" if pos[i] < .70 else "time_test", "sess_test" if sid[i] in te_s else "sess_val" if sid[i] in va_s else "sess_train"]
        e = info.setdefault(nz(p["q"]), {"q": p["q"], "sid": p["sid"], "label": names[lab[i]] if lab[i] >= 0 else "none", "split": set()}); e["split"].update(tags)
    trainable = lambda e: bool(e["split"] & {"time_train", "sess_train"})
    ents = list(info.values()); rnd = random.Random(0)
    g1 = [e for e in ents if e["label"] != "none" and trainable(e)]; g2a = [e for e in ents if e["label"] == "none" and "time_train" in e["split"]]
    g2b = [e for e in ents if e["label"] == "none" and "time_train" not in e["split"] and trainable(e)]
    g3, seen = [], set(info)
    for l in open(ROUTE, encoding="utf-8"):
        c = l.rstrip("\n").split("\t")
        if len(c) < 4 or len(c[3].split()) < 4 or c[3].startswith("<") or nz(c[3]) in seen: continue
        seen.add(nz(c[3])); g3.append({"q": c[3], "sid": "route.log", "label": None, "split": {"route_log"}})
    for g in (g1, g2a, g2b, g3): rnd.shuffle(g)
    k3 = min(len(g3), reserve); head = (g1 + g2a + g2b)[:cap - k3]; rows = head + g3[:cap - len(head)]
    print(f"plan: labelled-train {len(g1)} none-time-train {len(g2a)} none-sess-train {len(g2b)} unlabelled {len(g3)} -> cap {cap}: {len(head)} + {len(rows) - len(head)} = {len(rows)}", flush=True)
    return rows, names


def prompt(q, skills):
    cat = "\n".join(f"- {k}: {v}" for k, v in skills.items())
    return ("You route a request to the skills it needs. Skills:\n" + cat + "\n\nGive a probability for the up to 8 most likely skills and for none (no skill fits). "
            'Reply with JSON only, no other words: {"p":[["skill-name",0.55],["skill-name",0.2]],"none":0.1}. Probabilities of the listed skills plus none sum to at most 1.\n\nRequest: ' + q[:400])


def parse(text, names):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: j = json.loads(m.group(0))
    except ValueError: return None
    d = {}
    for it in j.get("p", []):
        try:
            k, v = it[0], float(it[1])
        except (IndexError, TypeError, ValueError): continue
        if k in names and v > 0: d[k] = d.get(k, 0.0) + v
    try: nn = max(0.0, float(j.get("none", 0)))
    except (TypeError, ValueError): nn = 0.0
    if sum(d.values()) + nn <= 0: return None
    d = dict(sorted(d.items(), key=lambda kv: -kv[1])[:TOPN]); return d, nn


def smooth(d, nn, names):
    tot = sum(d.values()) + nn; keep = tot if tot > 1 else 1.0   # tot < 1: the unassigned mass is spread over the other classes
    raw = {k: v / keep for k, v in d.items()}; raw["none"] = nn / keep; rem = max(0.0, 1 - sum(raw.values())); nrest = len(names) + 1 - len(raw)
    out = {k: (1 - EPS) * v + EPS / (len(names) + 1) for k, v in raw.items()}; rest = (1 - EPS) * rem / nrest + EPS / (len(names) + 1)
    s = sum(out.values()) + rest * nrest
    return {k: v / s for k, v in out.items()}, rest / s


def start_teacher():
    import localread as lr, gpu, judge_fresh as jf
    lr.SAGE_MB = 9500; g = gpu.pick_gpu(9500)
    if g < 0: sys.exit("teacher_dump: no GPU with >= 9.5 GB free (or game on gpu 0)")
    fr = gpu.free_vram_mb(g); cache = 2000 if fr >= 11500 else 500 if fr >= 10300 else 0   # --moe-cache-mib shrinks so the model fits the free VRAM
    lr.SAGE_EXTRA = [("%d" % cache if x == "2000" else x) for x in lr.SAGE_EXTRA]; print(f"teacher_dump: gpu {g} free {fr} MB, moe cache {cache}", flush=True)
    return jf.start_model()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=2500); ap.add_argument("--reserve", type=int, default=400); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stub", action="store_true"); ap.add_argument("--out", default=OUT); a = ap.parse_args()
    skills = json.load(open(os.path.join(D, "skills.json"), encoding="utf-8")); names = list(skills)
    rows, _ = build_rows(a.cap, a.reserve); done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: done.add(nz(json.loads(l)["q"]))
            except ValueError: pass
    todo = [r for r in rows if nz(r["q"]) not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"dumped already {len(done)}, todo {len(todo)}", flush=True)
    if not todo: return 0
    proc = lr = None; stop = lambda: None
    if a.stub:
        def ask(q): rnd = random.Random(q); k = rnd.sample(names, 3); return json.dumps({"p": [[k[0], .6], [k[1], .2], [k[2], .1]], "none": .1})
    else:
        lr, proc = start_teacher()
        def stop():
            if proc is not None and proc.poll() is None: subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True); print("teacher_dump: Sage stopped", flush=True)
        atexit.register(stop)
        def ask(q):
            for temp in (0.0, 0.7):
                x = lr.http(PORT, "/v1/chat/completions", {"messages": [{"role": "user", "content": prompt(q, skills)}], "temperature": temp, "max_tokens": 260, "cache_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
                t = lr.clean(x["choices"][0]["message"].get("content") or "")
                if parse(t, names): return t
            return ""
    t0 = time.time(); k = ag = nl = fails = 0
    try:
        with open(a.out, "a", encoding="utf-8", newline="\n") as f:
            for r in todo:
                ts = time.time(); pr = parse(ask(r["q"]), names)
                if pr is None: row = {"q": r["q"], "sid": r["sid"], "label": r["label"], "split": sorted(r["split"]), "fail": True}; fails += 1
                else:
                    dist, rest = smooth(*pr, names); top = max(dist, key=dist.get)
                    row = {"q": r["q"], "sid": r["sid"], "label": r["label"], "split": sorted(r["split"]), "dist": {kk: round(v, 6) for kk, v in dist.items()}, "rest": round(rest, 7), "raw_top1": top,
                           "secs": round(time.time() - ts, 2)}
                    if r["label"] is not None: row["agree"] = top == r["label"]; nl += 1; ag += row["agree"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); k += 1
                if k % 25 == 0: print(f"{k}/{len(todo)} {k / (time.time() - t0) * 3600:.0f} rows/h agree {ag}/{nl} fails {fails}", flush=True)
    finally:
        stop()
    el = time.time() - t0; print(f"teacher_dump: {k} rows in {el:.0f} s = {k / max(el, 1) * 3600:.0f} rows/h, labelled agree {ag}/{nl}, fails {fails}"); return 0


if __name__ == "__main__":
    sys.exit(main())
