# Tripwire post-mortem (2026-11-18): turned off, it failed

Tripwire was the mid-task steer system built 2026-11-10 to 2026-11-17 (`hooks/tripwire.py`,
`hooks/tripwire_dispatch.py`, design in `tree/tripwire-print.md`). On 2026-11-18 all four hooks were
removed from the live settings. The code stays in this repo as a record; it is not recommended for use.

## What it was meant to do
Catch a correction the user gives in the middle of a task, store it as a note, show the note again right
before the matching file edit, and refuse to let the turn end while a note is still open.

## What happened in real use
Counts from the hook's own log, all chats, 2026-11-10 to 2026-11-18:

| Measure | Count |
|---|---|
| User prompts the dispatcher judged | 1878 |
| Judged "ambiguous" (handed back to the model to decide) | 1096 (58 %) |
| Notes stored | 86 |
| Notes that fired on a matching edit | 10 |
| Notes marked applied | 71 |
| Notes waived | 9 |
| Turn-end blocks | 199 |
| Blocks still open at the second stop (the block changed nothing) | 189 |

## Why it failed
- The sorter could not sort. More than half of the prompts came back as "you judge", so the model paid
  tokens to read the hook line and then did the judging anyway.
- The trigger almost never matched. 10 of 86 notes fired on an edit; the other notes were closed by hand
  or never closed.
- The stop block added cost without changing behaviour. 189 of 199 blocks ended with the note still open.
- Every part of the loop cost model output: the injected line, the `applied` / `waive` command, the extra
  turn after a block. The steer was already in the conversation, so the stored copy only added a second
  reminder of something the model had already read.
- The user's summary: instead of being done, the notes kept being stored and repeated, burning output
  tokens; none of it was plain code output that would have cost nothing.

## Likely cause
Probably a misunderstanding of the goal or of the architecture, not a tuning problem. The design assumed
a cheap script could tell a steer from a new task or a remark, and that enforcement by hooks would be
close to free. Neither held: the classification needs judgement, and each enforcement step is paid for
in model tokens. With Claude Fable 5.2 those tokens are too expensive for the gain.

Root cause found 11-19: persistence fired on non-completion. A note that did not close got
stored, repeated and blocked again, so each failure made the mechanism spend more. The failure fed itself:
a self-amplifying loop, not a trigger. Any revival needs a retry cap, backoff on failure, and must never
re-fire because of its own failure.

## Status
Shelved. To be picked up again when models are smarter and cheaper. The earlier scores in the README and
in `tree/tripwire-print.md` (9 of 9 eval cases, the 127-message score, the retrain) are kept as history;
they measured the parts, not whether the whole system saved anything.
