# nudger measurements 2026-10-27 18:41 (nudger_measure.py, read-only over route.log 1866 rows, usage.jsonl 934 reads, 78 skills)

## A4 refute templates
- live rows with col 7: 4; refuted: 3
- retro (template applied to every historical row), conf x class -> refuted count:
  | conf | class | refuted | n |
  |---|---|---|---|
  | high | brief_only | False | 329 |
  | high | full_read | False | 132 |
  | high | full_read | True | 223 |
  | high | miss | True | 173 |
  | low | brief_only | True | 113 |
  | low | full_read | False | 21 |
  | low | miss | False | 32 |
  | medium | brief_only | False | 355 |
  | medium | full_read | False | 56 |
  | medium | full_read | True | 135 |
  | medium | miss | True | 284 |
  | none | none | False | 7 |
  | none | none+read | True | 6 |
- cells where refuted is NOT determined by (conf,class): [('medium', 'full_read'), ('high', 'full_read')]; full_read rows with >1 skill read (only place a template could add info): 372/567
- high/medium rows: refuted == (class==miss) on 0.788 of 1687 rows

## A3 exploration slot
- live explore rows: 0; explore_hit: 0
- rows with a read where some read skill is outside top-1: 867/1062 = 0.816
- rows with a read where some read skill is outside top-3: 630/1062 = 0.593
- rows with a read where some read skill is outside top-5: 630/1062 = 0.593
- retro least-routed-14d candidate read in window: 17/1866 = 0.0091
- expected hit rate of a uniform-random explore candidate: 0.0136
- skills read but outside top-3 (top 10): ops-manager 180, local-gpu-ai 157, goobi-brawler 76, video-notes 69, swat4-mod 62, archive-digests 44, video-grab 33, hydrophobia-mod 33, avalon-bevy 30, deep-research 19

## A6 autonomy baseline (no nudger)
  | day | routes | distinct top-1 | entropy bits |
  |---|---|---|---|
  | 2026-10-17 | 227 | 51 | 4.36 |
  | 2026-10-18 | 325 | 29 | 2.62 |
  | 2026-10-19 | 130 | 12 | 2.38 |
  | 2026-10-20 | 151 | 29 | 3.14 |
  | 2026-10-21 | 109 | 8 | 1.62 |
  | 2026-10-22 | 149 | 9 | 1.34 |
  | 2026-10-23 | 181 | 9 | 1.93 |
  | 2026-10-24 | 122 | 10 | 1.96 |
  | 2026-10-25 | 228 | 10 | 1.52 |
  | 2026-10-26 | 150 | 8 | 1.99 |
  | 2026-10-27 | 94 | 7 | 1.70 |
- novel top-1 (never routed before) over all rows: 66/1866; skills ever top-1: 66/78
- top-1 share, top 5: video-notes 0.55, ops-manager 0.10, goobi-brawler 0.09, local-gpu-ai 0.09, swat4-mod 0.03
