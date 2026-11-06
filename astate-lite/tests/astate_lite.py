"""astate_lite.py: reference implementation of the mechanics that the astate-lite skill describes in prose
(SKILL.md sections 1-5 and the four templates). It exists only so the prose can be tested; where the prose
leaves a choice open, the choice made here is marked GAP-n and listed in TESTS.md.

CLI:  py -3.13 -I astate_lite.py regen <LOG.md> <STATE.md> [--project NAME] [--date YYYY-MM-DD]
      py -3.13 -I astate_lite.py check <dir>     (validates <dir>/.state/LOG.md + STATE.md, prints JSON report)
      py -3.13 -I astate_lite.py gate <contract.md> <STATE.md> "<user message>"
"""
import re, hashlib, os, sys, json, time, datetime as dt

try:
    import msvcrt
except ImportError:      # POSIX
    msvcrt = None
    import fcntl

EM = "—"                      # em dash: the separator the claim grammar uses
TAGS = ("SRC", "SYN", "ø")    # [SRC] seen in a source, [SYN] inferred, [ø] unknown / not checked
CLAIM_STATUSES = ("confirmed", "hypothesis", "rejected", "unknown")
LOG_TYPES = ("decision", "event", "blocker", "metric")
LOG_STATUSES = CLAIM_STATUSES   # GAP-1 fixed 11-05: SKILL.md:38 lists rejected; only templates/LOG.md omitted it
STATE_CAP = 2000     # tokens, "STATE <= ~2k tokens"
TASK_CAP = 300       # tokens, "~300 tokens"
EXPLORE_BUDGET = 5   # "explore <= 5 steps"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CLAIM_RE = re.compile(r"^- \[(SRC|SYN|ø)\] (.+?) — status: (confirmed|hypothesis|rejected|unknown) — evidence: (.+?) — since: (\S+)$")
CONFLICT_RE = re.compile(r"^conflict: (.+?) vs (.+?) — (.+?) — (.+)$")      # "conflict: A vs B — sources — next"
METRIC_RE = re.compile(r"^rediscoveries=(\d+) tokens_open=(\d+) cold_test=(\d+)/5$")
TAG_RE = re.compile(r"^\[(SRC|SYN|ø)\]\s*(.*)$")
# A pointer into a source: README.md:9, src/app.py:3-9, README.md#L9, src/app.py line 3, Makefile:12. Skeptic round 11-05:
# any '.ext:N' also matched 127.0.0.1:8080 / example.com:80 / ver 3.13:5, so the extension must be a source/doc kind.
SRC_EXT = r"(?:py|md|txt|rst|toml|json|ya?ml|ini|cfg|conf|env|ps1|sh|bat|cmd|js|ts|rs|c|h|cpp|hpp|cs|gd|tscn|tres|java|go|rb|lua|sql|html|css|xml|uc|csv)"
FILE_LINE_RE = re.compile(r"(?:[^\s,;()]+\." + SRC_EXT + r"|(?:^|[\s/\\])(?:Makefile|Dockerfile|LICENSE|Justfile|Rakefile))"
                          r"(?::\d+|#L\d+|\s+line\s+\d+)\b", re.IGNORECASE)
STATE_HEADS = {"standing questions": "questions", "claims": "claims", "conflicts": "conflicts", "procedures": "procedures",
               "open blockers": "blockers", "receipts": "receipts"}


CJK_RE = re.compile(r"[⺀-鿿가-퟿豈-﫿＀-￯]")   # CJK, Kana, Hangul, full-width forms


def tokens(text):
    """~4 chars per token; the skill says 'tokens' without naming a tokenizer (GAP-2). CJK/Kana/Hangul run about one
    token per character, so they count whole (skeptic 11-06: 400 CJK chars read as 100 tokens and slipped under TASK_CAP)."""
    cjk = len(CJK_RE.findall(text))
    return (len(text) - cjk + 3) // 4 + cjk


def norm(s):
    return " ".join(s.lower().replace(EM, "-").split())


# ---------------------------------------------------------------- claims (section 4)
def parse_claim(line):
    m = CLAIM_RE.match(line.rstrip("\r\n"))
    if not m:
        return None
    tag, text, status, ev, since = m.groups()
    return {"tag": tag, "text": text, "status": status, "evidence": ev, "since": since}


def render_claim(c):
    return f"- [{c['tag']}] {c['text']} {EM} status: {c['status']} {EM} evidence: {c['evidence']} {EM} since: {c['since']}"


# ---------------------------------------------------------------- LOG lines (template LOG.md)
def _log_fields(raw):
    """Returns the five fields or an error string. Right split keeps a | inside 'what happened' (GAP-3: no escaping rule)."""
    if raw.count("|") < 4:
        return None, f"expected 5 fields separated by |, got {raw.count('|') + 1}"
    date, typ, rest = [p.strip() for p in raw.split("|", 2)]
    what, ev, st = [p.strip() for p in rest.rsplit("|", 2)]
    if not DATE_RE.match(date):
        return None, f"bad date {date!r}"
    if typ not in LOG_TYPES:
        return None, f"bad type {typ!r}"
    if not ev.startswith("evidence:") or not st.startswith("status:"):
        return None, "4th field must start with 'evidence:', 5th with 'status:'"
    return (date, typ, what, ev[len("evidence:"):].strip(), st[len("status:"):].strip()), None


def _metric(what):
    m = METRIC_RE.match(what)
    if not m:
        return None, "metric line must read rediscoveries=<n> tokens_open=<n> cold_test=<k>/5"
    if int(m[3]) > 5:
        return None, "cold_test k must be <= 5"
    return {"rediscoveries": int(m[1]), "tokens_open": int(m[2]), "cold_test": int(m[3])}, None


def parse_log(line, strict=True):
    """date | type | what happened | evidence: <pointer> | status: <s>. Returns (entry, error); error 'skip' for headers/blank."""
    raw = line.rstrip("\r\n").lstrip("\ufeff")       # 11-06: PowerShell 5.1 can leave a BOM after an append (s8 item 8)
    if not raw.strip() or raw.startswith("#") or raw.startswith("("):
        return None, "skip"
    fields, err = _log_fields(raw)
    if err:
        return None, err
    date, typ, what, ev, status = fields
    if status not in (LOG_STATUSES if strict else CLAIM_STATUSES):
        return None, f"bad status {status!r}"
    e = {"date": date, "type": typ, "what": what, "evidence": ev, "status": status, "raw": raw}
    if typ == "metric":
        e["metric"], err = _metric(what)
        if err:
            return None, err
    return e, None


def render_log(date, typ, what, evidence, status):
    return f"{date} | {typ} | {what} | evidence: {evidence} | status: {status}"


def _decode(data):
    """Bytes -> (text, encoding), never raises (s8 item 8, 11-06). PowerShell 5.1 `>` writes UTF-16LE with a BOM; its `>>`
    appends BOM-less UTF-16LE to whatever is there (NULs after the first newline = mixed: UTF-8 head + UTF-16 tail);
    NULs before the first newline = a BOM-less UTF-16LE file. CRLF is folded to LF as text mode did."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return _nl(data.decode("utf-16", errors="replace")), "utf-16"
    i = data.find(b"\x00")
    if i < 0:
        return _nl(data.decode("utf-8-sig", errors="replace")), "utf-8"
    s = _tail_start(data, i)
    if not _utf16_like(data[s:]):   # a stray NUL inside UTF-8 text (skeptic 11-06): keep the text, parse_log rejects that line
        return _nl(data.decode("utf-8-sig", errors="replace")), "utf-8"
    if s == 0:
        return _nl(data.decode("utf-16-le", errors="replace")), "utf-16-le"
    return _nl(data[:s].decode("utf-8-sig", errors="replace") + data[s:].decode("utf-16-le", errors="replace")), "mixed utf-8 + utf-16"


def _utf16_like(b):
    """UTF-16LE text is about half NUL bytes (ASCII); one third is the bar, so a stray NUL or two never qualifies."""
    return len(b) >= 2 and b.count(b"\x00") * 3 >= len(b)


def _tail_start(data, i):
    """Start of the UTF-16LE run whose first NUL byte sits at i: i-1 when that NUL is the high byte of an ASCII unit (the
    usual case), i when the run opens with a unit like U+0100 (skeptic 11-06). The aligned candidate is the one whose
    LOW bytes are rarely NUL; ties keep i-1."""
    best = None
    for s in (i - 1, i):
        if s < 0 or (len(data) - s) % 2:
            continue
        score = data[s::2].count(0)
        if best is None or score < best[0]:
            best = (score, s)
    return best[1] if best else max(i - 1, 0)


def _nl(text):
    """Text mode folded CRLF and lone CR to LF; binary reads must too (skeptic 11-06: a CR-only LOG lost its entries)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_log(path, strict=True):
    entries, errors = [], []
    if not os.path.exists(path):
        return entries, errors
    with open(path, "rb") as f:
        text = _decode(f.read())[0]
    for n, line in enumerate(text.split("\n"), 1):
        e, err = parse_log(line, strict)
        if e:
            entries.append(e)
        elif err != "skip":
            errors.append((n, err, line.rstrip("\r\n")))
    return entries, errors


def _lock(f, on):
    """Cross-process lock for LOG appends. Measured 11-05: 8 sessions x 500 unlocked appends lost ~20% of the lines on
    Windows (results/unit.json concurrent_8x500_appends), so the lock is part of what 'append-only' has to mean."""
    f.seek(0)
    if msvcrt is None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if on else fcntl.LOCK_UN)
        return
    if not on:
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return
    for _ in range(20000):
        try:
            return msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            time.sleep(0.0005)
    raise OSError("LOG lock timeout")


SNIFF = 8192


def _sniff(f):
    """Encoding of an open binary file from its first and last SNIFF bytes (skeptic 11-06: a 2-byte BOM check let a
    BOM-less UTF-16 or a mixed LOG take a UTF-8 line). Leaves the file pointer anywhere; callers seek after it."""
    size = f.seek(0, 2)
    f.seek(0)
    head = f.read(SNIFF)
    f.seek(max(0, size - SNIFF))
    tail = f.read(SNIFF)
    for enc in (_decode(head)[1], _decode(tail)[1]):
        if enc != "utf-8":
            return enc
    return "utf-8"


def append_log(path, line, lock=True):
    """Append-only write: the file is only ever opened for appending; lock=False shows the unlocked loss in the stress test."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # unbuffered (11-06, s8 item 8): a BufferedRandom read-ahead leaves the OS pointer past byte 0, so the unlock after the
    # UTF-16 guard raised hit the wrong byte (EACCES) and close() rewound to a negative offset (EINVAL).
    with open(path, "a+b", buffering=0) as f:
        if lock:
            _lock(f, True)
        try:
            enc = _sniff(f)
            if enc != "utf-8":
                raise ValueError(f"{path} is not UTF-8 ({enc}): re-save it as UTF-8 before appending (s8 item 8)")
            f.seek(0, 2)
            f.write((line.rstrip("\r\n") + "\n").encode("utf-8"))
            f.flush()
        finally:
            if lock:
                _lock(f, False)


def read_text(path):
    with open(path, "rb") as f:
        return _decode(f.read())[0]


def _encoding_problems(*paths):
    """`check` names a .state file that is not UTF-8 (UTF-16 from PowerShell 5.1 `>`, or mixed after a `>>`)."""
    out = []
    for p in paths:
        with open(p, "rb") as f:
            enc = _decode(f.read())[1]
        if enc != "utf-8":
            out.append(f"{os.path.basename(p)} not UTF-8 ({enc}): re-save as UTF-8")
    return out


def append_only_ok(before, after):
    """'never edit old lines': the new text must start with the old text, line for line."""
    b = before.splitlines()
    a = after.splitlines()
    return len(a) >= len(b) and a[: len(b)] == b


# ---------------------------------------------------------------- STATE regeneration (section 2, Close)
def split_tag(what, evidence=""):
    m = TAG_RE.match(what)
    if m:
        return m[1], m[2].strip()
    if FILE_LINE_RE.search(evidence):
        return "SRC", what    # GAP-4 fixed 11-05: untagged fact with a file:line pointer = seen in a source
    return "ø", what          # untagged, no source pointer = not checked


def _drop(parts, key):
    for d in (parts["claims"], parts["conflicts"], parts["procedures"], parts["blockers"], parts["receipts"]):
        for k in [k for k in d if norm(k) == key or norm(str(d[k])) == key]:
            d.pop(k, None)
    parts["questions"][:] = [q for q in parts["questions"] if norm(q) != key]


def _apply_prefixed(parts, w):
    """conflict: / procedure: / question: / receipt: / drop: lines pick a STATE section (GAP-5: the skill does not say
    how a LOG line maps to a STATE section; this is the convention under test). Returns False for a plain claim."""
    low = w.lower()
    if low.startswith("conflict:"):
        parts["conflicts"][norm(w)] = w
    elif low.startswith("procedure:"):
        name, _, steps = w[len("procedure:"):].partition(":")
        parts["procedures"][name.strip()] = steps.strip()
    elif low.startswith("question:"):
        q = w[len("question:"):].strip()
        if q and q not in parts["questions"]:
            parts["questions"].append(q)
    elif low.startswith("receipt:"):
        body = w[len("receipt:"):].strip()
        parts["receipts"][body.split(f" {EM} ")[0]] = body
    elif low.startswith("drop:"):
        _drop(parts, norm(w[len("drop:"):]))
    else:
        return False
    return True


def _apply_blocker(parts, e):
    key = norm(e["what"])
    if e["status"] == "unknown":
        parts["blockers"][key] = {"text": e["what"], "since": e["date"], "owner": e["evidence"]}
    else:
        parts["blockers"].pop(key, None)


def _apply_claim(parts, e):
    tag, text = split_tag(e["what"], e["evidence"])
    key = norm(text)
    c = parts["claims"].get(key)
    if c is None:
        parts["claims"][key] = {"tag": tag, "text": text, "status": e["status"], "evidence": e["evidence"], "since": e["date"],
                                "last": e["date"], "n": len(parts["claims"])}
    else:
        c.update(tag=tag, status=e["status"], evidence=e["evidence"], last=e["date"])   # promote / demote; since = first date


def derive(entries):
    """LOG -> STATE parts: claims (latest line wins), conflicts, procedures, open blockers (type blocker, status unknown;
    a later line with the same text and another status closes it), questions (max 5), receipts, metrics."""
    parts = {"claims": {}, "conflicts": {}, "procedures": {}, "blockers": {}, "questions": [], "receipts": {}, "metrics": []}
    for e in entries:
        if e["type"] == "metric":
            parts["metrics"].append(e)
        elif _apply_prefixed(parts, e["what"]):
            pass
        elif e["type"] == "blocker":
            _apply_blocker(parts, e)
        else:
            _apply_claim(parts, e)
    parts["questions"] = parts["questions"][:5]
    return parts


def render_state(parts, project="<project>", date=None):
    date = date or dt.date.today().isoformat()
    out = [f"# STATE {EM} {project}", f"(regenerated from LOG on {date}; keep ≤ ~2k tokens)", "", "Standing questions:"]
    out += [f"{i}. {q}" for i, q in enumerate(parts["questions"], 1)]
    out += ["", "Claims:"] + [render_claim(c) for c in sorted(parts["claims"].values(), key=lambda c: c["n"])]
    out += ["", "Conflicts, both sides kept:"] + list(parts["conflicts"].values())
    out += ["", "Procedures, the how-to that keeps getting forgotten:"] + [f"{k}: {v}" for k, v in parts["procedures"].items()]
    out += ["", "Open blockers:"] + [f"{b['text']} {EM} since {b['since']} {EM} {b['owner']}" for b in parts["blockers"].values()]
    out += ["", "Receipts:"] + list(parts["receipts"].values())
    return "\n".join(out) + "\n"


DROP_ORDER = ("rejected", "unknown", "hypothesis", "confirmed")


def _trim_claims(parts, cap, project, date):
    """Drop claims in DROP_ORDER (then oldest 'last' date, then insertion order) until the STATE fits the cap.
    Size is tracked arithmetically (one claim line = its render + newline), so this is one sort, not n renders."""
    cl = parts["claims"]
    order = sorted(cl, key=lambda k: (DROP_ORDER.index(cl[k]["status"]), cl[k]["last"], cl[k]["n"]))
    size, dropped = len(render_state(parts, project, date)), []
    for k in order:
        if size <= cap * 4:          # tokens(text) = (len + 3) // 4 <= cap  <=>  len <= 4 * cap
            break
        dropped.append(cl.pop(k))
        size -= len(render_claim(dropped[-1])) + 1
    return dropped


def regenerate(entries, project="<project>", date=None, cap=STATE_CAP):
    """Returns (state_text, report). Cap policy (GAP-6, the skill says only 'respect the cap'):
    drop claims by status rejected -> unknown -> hypothesis -> confirmed, oldest 'last' date first;
    conflicts, procedures, blockers, questions and receipts are never dropped by the cap."""
    parts = derive(entries)
    dropped = _trim_claims(parts, cap, project, date)
    text = render_state(parts, project, date)
    rep = {"tokens": tokens(text), "cap": cap, "over_cap": tokens(text) > cap, "claims": len(parts["claims"]),
           "dropped": [f"{c['status']}: {c['text']}" for c in dropped], "metrics": len(parts["metrics"])}
    rep.update({k: len(parts[k]) for k in ("conflicts", "procedures", "blockers", "questions")})
    return text, rep


def _state_head(s):
    low = s.lower().rstrip(":")
    hit = next((v for k, v in STATE_HEADS.items() if low.startswith(k)), None)
    return hit if hit and (s.endswith(":") or "," in s) else None


def parse_state(text):
    """STATE.md -> dict of sections (claims parsed with the grammar, others kept as lines)."""
    sec = None
    out = {"questions": [], "claims": [], "bad_claims": [], "conflicts": [], "procedures": [], "blockers": [], "receipts": [], "header": []}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        head = _state_head(s)
        if head:
            sec = head
        elif sec is None:
            out["header"].append(s)
        elif sec == "claims":
            c = parse_claim(s)
            out["claims" if c else "bad_claims"].append(c or s)
        else:
            out[sec].append(s)
    return out


def confirmed_claims(state_text):
    return [c["text"] for c in parse_state(state_text)["claims"] if c["status"] == "confirmed"]


# ---------------------------------------------------------------- the gate (section 3) + contract (template)
CONTRACT_KEYS = ("requires", "reads", "produces", "effects", "conflicts")


class ContractError(ValueError):
    pass


def _front_matter(text):
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ContractError("no front matter")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ContractError("front matter not closed")
    return lines[1:end]


def _yaml_val(v):
    if v == "":
        return ""
    if v.startswith("[") and v.endswith("]"):
        return [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
    return v.strip("'\"")


def _fm_item(s, cur, target):
    if cur is None:
        raise ContractError("list item without key")
    old = target.get(cur)
    target[cur] = (old if isinstance(old, list) else []) + [s[2:].strip()]


def _fm_key(s, indent, in_meta, doc, meta):
    """One 'key: value' line. Returns (current key, inside metadata block)."""
    key, _, val = s.partition(":")
    cur = key.strip()
    if indent == 0:
        if cur == "metadata":
            return cur, True
        doc[cur] = _yaml_val(val.strip())
        return cur, False
    if not in_meta:
        raise ContractError(f"unexpected indented key {cur!r}")
    meta[cur] = _yaml_val(val.strip())
    return cur, True


def _fm_lines(lines):
    """Minimal YAML: top keys, 'metadata:' block with indented keys, '- item' lists. Returns (doc, meta)."""
    doc, meta, cur, in_meta = {}, {}, None, False
    for raw in lines:
        line = raw.rstrip() if raw.lstrip().startswith("- ") else raw.split("#", 1)[0].rstrip()
        s = line.strip()
        if not s:
            continue
        indent = len(line) - len(line.lstrip())
        if s.startswith("- "):
            _fm_item(s, cur, meta if in_meta and indent >= 2 else doc)
        else:
            cur, in_meta = _fm_key(s, indent, in_meta, doc, meta)
    return doc, meta


def _check_keys(doc, meta):
    missing = [k for k in CONTRACT_KEYS if k not in meta]
    if missing:
        raise ContractError(f"metadata missing {missing}")
    stray = [k for k in CONTRACT_KEYS if k in doc]
    if stray:
        raise ContractError(f"{stray} must sit under metadata:")
    if not doc.get("name"):
        raise ContractError("name missing")


def _as_list(v):
    if v in ("", None):
        return []
    return [v] if isinstance(v, str) else v


def parse_contract(text):
    """'In a real Cowork skill these five keys sit under metadata: or the loader rejects the skill.'"""
    doc, meta = _fm_lines(_front_matter(text))
    _check_keys(doc, meta)
    for k in CONTRACT_KEYS:
        meta[k] = _as_list(meta[k])
    doc["metadata"] = meta
    return doc


def item_met(item, confirmed, message):
    """One requires item is met when it equals a confirmed claim (normalised) or appears as a whole phrase in the message.
    GAP-7 fixed 11-05: substring matching met negations ('port is not 8080') and one-letter items; now exact / word-bounded."""
    it = _key(item)
    if not it:
        return False
    # skeptic round 11-05: '.' between word chars continues the token (v1.2 is not met by v1.2.3); a final '.' does not
    if re.search(r"(?<!\w)(?<!\w\.)" + re.escape(it) + r"(?!\w)(?!\.\w)", norm(message)):
        return True
    return any(it == _key(c) for c in confirmed)


def _key(s):
    """norm without trailing sentence punctuation: 'Use py.' and 'use py' are one claim (skeptic round 11-05)."""
    return norm(s).rstrip(" .;:!")


def gate(contract, confirmed, message):
    """Returns (load, unmet). GAP-8 fixed 11-05: an empty `requires` never loads (no claim to match = no match)."""
    req = contract["metadata"]["requires"]
    if not req:
        return False, ["requires is empty"]
    unmet = [r for r in req if not item_met(r, confirmed, message)]
    return (not unmet), unmet


def _clash_index(passing):
    """Lookup tables over the passing candidates: name or effect -> contracts, conflict item -> contracts."""
    by_name, by_conflict = {}, {}
    for x in passing:
        for k in (x["name"], *x["metadata"]["effects"]):
            by_name.setdefault(_key(k), []).append(x)
        for k in x["metadata"]["conflicts"]:
            by_conflict.setdefault(_key(k), []).append(x)
    return by_name, by_conflict


def _clash(c, by_name, by_conflict):
    """Conflict items between c and the other passing candidates, both directions: c's `conflicts` naming another's
    name or effect, or another's `conflicts` naming c or one of c's effects. Items match through _key like the gate's
    own matching (skeptic 11-06: conflicts ['b'] did not catch a candidate named 'B')."""
    clash = [k for k in c["metadata"]["conflicts"] if any(x is not c for x in by_name.get(_key(k), []))]
    for k in (c["name"], *c["metadata"]["effects"]):
        clash += [x["name"] for x in by_conflict.get(_key(k), []) if x is not c]
    return sorted(set(clash))


def gate_many(contracts, confirmed, message):
    """Several candidates: each passes the gate on its own; a clash between two PASSING candidates holds both out
    ('ask the user'). Order-independent. GAP-9 decided 11-06 (s8 item 7): SKILL.md says conflicts are never silently
    resolved, so there is no first-wins tie rule; a candidate that fails its own gate holds nobody out."""
    verdicts = [gate(c, confirmed, message) for c in contracts]
    by_name, by_conflict = _clash_index([c for c, v in zip(contracts, verdicts) if v[0]])
    out = []
    for c, (load, unmet) in zip(contracts, verdicts):
        clash = _clash(c, by_name, by_conflict) if load else []
        why = f"mutual conflict with {clash}: ask the user" if clash else None
        out.append({"name": c["name"], "load": load and not clash, "unmet": unmet, "why": why})
    return out


# ---------------------------------------------------------------- TASK capsule, budget, receipts, metric
BLOCKER_NOTE = ("LOG blocker line: the question for the user goes in the what-text; "
                "status stays unknown until a later line with the same what-text closes it")   # GAP-11, templates/TASK.md


def task_capsule(goal, done, date, loaded="USER, STATE, skills: none", frontier=(), blockers=()):
    t = [f"# TASK {EM} {goal}", date, f"budget: explore ≤ {EXPLORE_BUDGET} steps", f"done = {done}", f"loaded: {loaded}", "", "frontier:"]
    t += [f"- [ ] {x}" for x in frontier] + ["", f"blockers ({BLOCKER_NOTE}):"] + [f"- {x}" for x in blockers]
    t += ["", "close: append LOG → regenerate STATE → receipts"]
    return "\n".join(t) + "\n"


class Budget:
    def __init__(self, steps=EXPLORE_BUDGET):
        self.left, self.used = steps, 0

    def explore(self):
        if self.left <= 0:
            return False
        self.left -= 1
        self.used += 1
        return True

    @property
    def exhausted(self):
        return self.left <= 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt_line(path, archived=False, root="."):
    """`path` is recorded as given (relative to the project root); the hash is taken from root/path. `check` verifies it
    against the file now (s8 item 9, 11-06)."""
    return f"receipt: {path} {EM} {sha256_file(os.path.join(root, path))} {EM} archived {'yes' if archived else 'no'}"


def metric_line(date, rediscoveries, tokens_open, cold_test):
    return render_log(date, "metric", f"rediscoveries={rediscoveries} tokens_open={tokens_open} cold_test={cold_test}/5", "weekly", "confirmed")


def metric_due(entries, today):
    last = [e["date"] for e in entries if e["type"] == "metric"]
    if not last:
        return True
    return (dt.date.fromisoformat(today) - dt.date.fromisoformat(max(last))).days >= 7


# ---------------------------------------------------------------- whole-project check (used by the dogfood checker)
def _state_problems(ps, state):
    out = []
    if ps["bad_claims"]:
        out.append(f"{len(ps['bad_claims'])} STATE claim lines off-grammar")
    if tokens(state) > STATE_CAP:
        out.append(f"STATE {tokens(state)} tokens > cap {STATE_CAP}")
    if len(ps["questions"]) > 5:
        out.append("more than 5 standing questions")
    if ps["questions"] and "open" not in ps["questions"][0].lower():
        out.append("first standing question is not 'How do I open X?'")
    return out


def _unbacked(ps, entries):
    """Provenance: STATE claims with no LOG line of the same (normalised) text ('never edit STATE without a LOG line').
    Exact match since 11-05 (substring let paraphrases pass; TESTS.md s8 item 3)."""
    logtexts = {norm(split_tag(e["what"])[1]) for e in entries}
    return [c["text"] for c in ps["claims"] if norm(c["text"]) not in logtexts]


def _check_state(state, entries, rep):
    ps = parse_state(state)
    rep.update(state_tokens=tokens(state), state_claims=len(ps["claims"]), state_bad_claims=ps["bad_claims"][:20],
               state_conflicts=len(ps["conflicts"]), state_procedures=len(ps["procedures"]), state_questions=len(ps["questions"]))
    rep["problems"] += _state_problems(ps, state)
    unbacked = _unbacked(ps, entries)
    rep["unbacked_claims"] = unbacked[:20]
    if unbacked:
        rep["problems"].append(f"{len(unbacked)} STATE claims with no LOG line")


CAPSULES = ("USER.md", "TASK.md")


def _capsule_problems(root):
    """USER / TASK capsules when they exist as files (.state/<name> first, else <root>/<name>): SKILL.md's table says
    ~300 tokens each (TASK_CAP); over the cap is reported like STATE over STATE_CAP (s8 item 6, 11-06)."""
    out = []
    for name in CAPSULES:
        for p in (os.path.join(root, ".state", name), os.path.join(root, name)):
            if os.path.isdir(p):   # skeptic 11-06: a folder named TASK.md crashed check with PermissionError
                out.append(f"{name} is a folder, not a file")
                break
            if os.path.isfile(p):
                n = tokens(read_text(p))
                if n > TASK_CAP:
                    out.append(f"{name[:-3]} {n} tokens > cap {TASK_CAP}")
                break
    return out


RECEIPT_RE = re.compile(r"^(.*?) — ([0-9a-fA-F]{64})(?: — .*)?$")   # path — sha256 — archived yes|no; the path may hold " — "


def _receipt_key(path):
    return os.path.normpath(path.strip()).replace("\\", "/").lower()


def _receipt_problems(root, entries):
    """Receipts (SKILL.md Close: sha256 of the sources relied on): the latest LOG receipt per path must match the file
    now; a source edited after its receipt is reported, never silently trusted (s8 item 9, 11-06). Skeptic 11-06: the
    sha is found from the right (paths with ' — ' split wrong), './f' and 'f' are one path, hex case is ignored, a
    folder is reported instead of crashing sha256_file."""
    latest = {}
    for e in entries:
        if e["what"].lower().startswith("receipt:"):
            m = RECEIPT_RE.match(e["what"][len("receipt:"):].strip())
            if m:
                latest[_receipt_key(m.group(1))] = (m.group(1).strip(), m.group(2).lower())
    out = []
    for path, sha in latest.values():
        p = os.path.join(root, path)
        if os.path.isdir(p):
            out.append(f"receipt names a folder, not a file: {path}")
        elif os.path.isfile(p) and sha256_file(p) != sha:
            out.append(f"receipt mismatch: {path} (source edited after its receipt; re-read it and log a new receipt)")
    return out


def check_project(root, strict=True):
    sd = os.path.join(root, ".state")
    rep = {"root": root, "ok": True, "problems": []}
    lp, sp = os.path.join(sd, "LOG.md"), os.path.join(sd, "STATE.md")
    rep["problems"] += [f"missing {os.path.basename(p)}" for p in (lp, sp) if not os.path.exists(p)]
    rep["problems"] += [f"{os.path.basename(p)} is a folder, not a file" for p in (lp, sp) if os.path.isdir(p)]
    if rep["problems"]:
        rep["ok"] = False
        return rep
    entries, errors = read_log(lp, strict)
    rep.update(log_entries=len(entries), log_errors=errors[:20])
    if errors:
        rep["problems"].append(f"{len(errors)} malformed LOG lines")
    _check_state(read_text(sp), entries, rep)
    rep["problems"] += _capsule_problems(root) + _receipt_problems(root, entries) + _encoding_problems(lp, sp)
    rep["ok"] = not rep["problems"]
    return rep


def write_state(path, text):
    """Atomic STATE write (tmp + os.replace): a reader never sees a torn file; with two closers the last writer wins."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _cmd_regen(argv):
    log, state = argv[1], argv[2]
    project = argv[argv.index("--project") + 1] if "--project" in argv else "<project>"
    date = argv[argv.index("--date") + 1] if "--date" in argv else None
    entries, errors = read_log(log, strict=False)
    text, rep = regenerate(entries, project, date)
    write_state(state, text)
    rep["log_errors"] = errors[:10]
    return rep


def _cmd_gate(argv):
    c = parse_contract(read_text(argv[1]))
    conf = confirmed_claims(read_text(argv[2]))
    load, unmet = gate(c, conf, argv[3])
    return {"name": c["name"], "load": load, "unmet": unmet}


def main(argv):
    cmds = {"regen": _cmd_regen, "check": lambda a: check_project(a[1]), "gate": _cmd_gate}
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    print(json.dumps(cmds[argv[0]](argv), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
