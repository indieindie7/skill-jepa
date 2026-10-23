"""Mine (prompt -> skill) pairs from Claude Code transcripts.

A user prompt is labelled with the skills the assistant touched before the next user prompt:
Skill tool calls, reads of skills/<name>/..., and skill-brief API calls (/api/skill/<name>).
Prompts that touched no skill get the label "none" (the router should abstain on those).
"skills_clean" keeps only real use: Skill tool calls, skill-brief API calls and Read/Grep/Glob of skill files. It drops
edits (Edit/Write/Bash writing a skill file), which are mostly end-of-turn skill upkeep, not the skill the prompt needed.

  py -3.13 build_data.py [--projects DIR] [--skills DIR] [--out data/pairs.jsonl]
Writes one JSON line per prompt: {"sid", "t", "q", "skills": [...]} and data/skills.json {name: description}.
The data stays local (data/ is git-ignored): prompts are private.
"""
import argparse, glob, json, os, re

HOME = os.path.expanduser("~")
ap = argparse.ArgumentParser()
ap.add_argument("--projects", default=os.path.join(HOME, ".claude", "projects"))
ap.add_argument("--skills", default=os.path.join(HOME, ".claude", "skills"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pairs.jsonl"))
a = ap.parse_args()

skills = {}
for f in glob.glob(os.path.join(a.skills, "*", "SKILL.md")):
    head = open(f, encoding="utf-8", errors="replace").read(3000)
    m = re.search(r"^description:\s*(.+)$", head, re.M)
    skills[os.path.basename(os.path.dirname(f))] = (m.group(1).strip().strip('"') if m else "")
names = sorted(skills, key=len, reverse=True)
path_re = re.compile(r"skills[\/]+([A-Za-z0-9_.-]+)[\/]")
api_re = re.compile(r"/api/skill/([A-Za-z0-9_.-]+)")
NOISE = ("<command-", "<local-command", "<task-notification", "This session is being continued", "[Request interrupted",
         "Caveat:", "<system-reminder>", "<side-sessions-event", "<ci-monitor-event")


def text_of(content):
    if isinstance(content, str): return content
    if any(b.get("type") == "tool_result" for b in content if isinstance(b, dict)): return None
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def touched(block):
    """(skills touched, used) where used = False for an edit of a skill file."""
    if block.get("type") != "tool_use": return [], True
    inp = block.get("input") or {}
    if block.get("name") == "Skill": return [str(inp.get("skill", "")).split(":")[-1]], True
    s = json.dumps(inp); name = block.get("name")
    used = name in ("Read", "Grep", "Glob") or (name in ("Bash", "PowerShell") and not re.search(r">>|> *\S*SKILL|open\([^)]*['\"][wa]", s))
    return path_re.findall(s) + api_re.findall(s), used


def clean(q):
    q = re.sub(r"<system-reminder>.*?</system-reminder>", "", q, flags=re.S)
    return re.sub(r"\s+", " ", q).strip()


n = 0
with open(a.out, "w", encoding="utf-8") as out:
    for f in glob.glob(os.path.join(a.projects, "*", "*.jsonl")):
        cur = None
        def flush():
            global n
            if cur and 3 <= len(cur["q"]) <= 2000:
                cur["skills"] = [s for s in dict.fromkeys(cur["skills"]) if s in skills] or ["none"]
                cur["skills_clean"] = [s for s in dict.fromkeys(cur["skills_clean"]) if s in skills] or ["none"]
                out.write(json.dumps(cur, ensure_ascii=False) + "\n"); n += 1
        for line in open(f, encoding="utf-8", errors="replace"):
            if '"type":"user"' not in line and '"type":"assistant"' not in line and '"type": "user"' not in line \
                    and '"type": "assistant"' not in line: continue
            try: j = json.loads(line)
            except ValueError: continue
            if j.get("isSidechain") or j.get("isCompactSummary") or j.get("isMeta"): continue
            msg = j.get("message") or {}
            if j.get("type") == "user":
                q = text_of(msg.get("content", ""))
                if q is None: continue
                q = clean(q)
                if not q or q.startswith(NOISE): continue
                flush()
                cur = {"sid": j.get("sessionId", os.path.basename(f))[:8], "t": j.get("timestamp", ""), "q": q, "skills": [], "skills_clean": []}
            elif j.get("type") == "assistant" and cur is not None and isinstance(msg.get("content"), list):
                for b in msg["content"]:
                    if isinstance(b, dict):
                        t, used = touched(b); cur["skills"] += t
                        if used: cur["skills_clean"] += t
        flush()
json.dump(skills, open(os.path.join(os.path.dirname(a.out), "skills.json"), "w", encoding="utf-8"), indent=1)
print(f"{n} prompts, {len(skills)} skills -> {a.out}")
