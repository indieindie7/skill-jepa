"""freeze_index.py: snapshot every LIVE input arms.A0 reads -> data/a0_index-<date>.json (+ stable copy data/a0_index.json, sha256 line appended to data/holdout.sha256).
Live inputs: opsd.load_skills() dict, opsd._SK['idf'], ~/.claude/skills/<name>/SKILL.md bodies, usage.jsonl rows (prev-skill prior), len(opsd.brief(skill)) per skill.
  py -3.13 -I freeze_index.py [--force]   refuses to overwrite data/a0_index.json unless --force (the baseline must stay fixed)."""
import hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "Documents", "Tools", "ops"))
import opsd
opsd.LOG_ROUTES = False
SK = os.path.join(os.path.expanduser("~"), ".claude", "skills")


def main():
    stable = os.path.join(D, "a0_index.json"); sha_f = os.path.join(D, "holdout.sha256")
    if os.path.exists(stable) and "--force" not in sys.argv: sys.exit(f"refuse: {stable} exists (use --force to re-freeze)")
    skills = opsd.load_skills(); bodies = {}
    for nm in skills:
        try: bodies[nm] = open(os.path.join(SK, nm, "SKILL.md"), encoding="utf-8", errors="replace").read()
        except OSError: bodies[nm] = None   # A0 falls back to skill description
    use = []
    for l in open(os.path.join(SK, "usage.jsonl"), encoding="utf-8"):
        try: j = json.loads(l); use.append({"t": j["t"], "sid": j.get("sid"), "skill": j.get("skill")})
        except (ValueError, KeyError): continue
    snap = {"frozen_at": time.strftime("%F %T"), "skills": skills, "idf": opsd._SK["idf"], "bodies": bodies, "usage": use, "brief_chars": {nm: len(opsd.brief(sk)) for nm, sk in skills.items()}}
    txt = json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=0); dated = os.path.join(D, f"a0_index-{time.strftime('%Y%m%d')}.json")
    for p in (dated, stable): open(p, "w", encoding="utf-8", newline="\n").write(txt)
    h = hashlib.sha256(open(stable, "rb").read()).hexdigest()
    lines = [l for l in open(sha_f, encoding="utf-8") if l.strip() and l.split()[1] != "a0_index.json"] + [f"{h}  a0_index.json\n"]
    open(sha_f, "w", encoding="utf-8", newline="\n").write("".join(l if l.endswith("\n") else l + "\n" for l in lines))
    print(f"froze {len(skills)} skills, {len(use)} usage rows -> {dated} (+ a0_index.json) sha256 {h}")


if __name__ == "__main__":
    main()
