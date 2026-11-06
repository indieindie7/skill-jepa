---
name:
description:
metadata:
  requires:    # every item a confirmed STATE claim or stated by the user
  reads:       # files or state it only reads
  produces:    # artifacts or claims it writes
  effects:     # sends, process execution, file writes
  conflicts:   # skills or effects it must not run alongside
---

Front matter with name, description, requires (every item a confirmed STATE claim or stated by the user), reads (files or state it only reads), produces (artifacts or claims it writes), effects (sends, process execution, file writes), conflicts (skills or effects it must not run alongside). In a real Cowork skill these five keys sit under `metadata:` or the loader rejects the skill.
