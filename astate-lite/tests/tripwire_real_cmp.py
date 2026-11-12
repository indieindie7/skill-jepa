import json, re, collections
P = json.load(open("real_pred.json", encoding="utf-8"))
M = {int(m.group(1)): m.group(2) for m in (re.match(r"(\d+)\. (.*)", l) for l in open("real_msgs.txt", encoding="utf-8")) if m}
S = {int(m.group(1)): m.group(2) for m in (re.match(r"(\d+): ([a-z_]+)", l) for l in open("sage_labels.txt", encoding="utf-8")) if m}
print("labelled", len(S), "max n", max(S), "missing", [p["n"] for p in P if p["n"] not in S])
def near(p):
    r = p["reason"]
    return r.split(":", 1)[1] if r.startswith("unsure:") else ("remark" if r == "remark" else p["cls"])
un = [p for p in P if not p["escalated"] and p["n"] in S]
print("unescalated", len(un), "agree", sum(p["cls"] == S[p["n"]] for p in un))
for p in un:
    if p["cls"] != S[p["n"]]:
        print("  DIS", p["n"], "disp", p["cls"], "sage", S[p["n"]], "|", M[p["n"]][:110])
for kind in ("unsure", "remark", "model", "stop_word"):
    g = [p for p in P if p["escalated"] and p["reason"].split(":")[0] == kind and p["n"] in S]
    print(kind, len(g), "nearest==sage", sum(near(p) == S[p["n"]] for p in g), "sage:", dict(collections.Counter(S[p["n"]] for p in g)))
for p in P:
    if (p["cls"] == "cancel" or S.get(p["n"]) == "cancel"):
        print("  CANCEL", p["n"], p["cls"], p["reason"], "sage", S.get(p["n"]), "|", M[p["n"]][:110])
print("by session", dict(collections.Counter(p["session"][:8] for p in P)))
