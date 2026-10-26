#!/bin/bash
cd /c/Users/john/AppData/Local/Temp/claude/C--/a65df56e-768a-48c0-afa4-3bf9b0548cb5/scratchpad/jobs
LOG=log.txt; echo "start b $(date +%H:%M)" >> $LOG
D=/c/Users/john/AppData/Local/Temp/claude/C--/a65df56e-768a-48c0-afa4-3bf9b0548cb5/scratchpad/dig
run() { n=$1; shift; t0=$(date +%s); py -3.13 -I lr4070.py "$@" --model strata --keep > $n.md 2> $n.err; rc=$?; echo "$n rc=$rc $(( $(date +%s) - t0 ))s bytes=$(wc -c < $n.md) $(date +%H:%M)" >> $LOG; }
Q=$(grep '^Q=' $D/run5.sh | sed "s/^Q='//; s/'\$//")
export LR_TWEET=300 LR_MAXTOK=350
run warmup skills_index.txt "How many skills are listed? Answer with the number only." --lines 1-5
export LR_TWEET=1200 LR_MAXTOK=800
run drill_es "$D/ws1.sel.txt" "Report the decider-2b / Evolution Strategies / EGGROLL idea in full: what model, what fitness or rule labels, why ES instead of QLoRA/GRPO, exact paths, commands, numbers, results, and the user's decision about it. Cite lines." --lines 6650-6950 --chunk 6000
run drill_tfidf "$D/ws2.sel.txt" "Report the router rerank design in full: TF-IDF word+trigram centroids, floor 0.15 -> 0.25, exact file paths, commands, Hit@1/Hit@3 numbers before and after, and the user's decisions. Cite lines." --lines 1100-1350 --chunk 6000
export LR_TWEET=1500 LR_MAXTOK=900
run dig_ws1_a "$D/ws1.sel.txt" "$Q" --lines 1-2500 --chunk 6000
run dig_ws1_b "$D/ws1.sel.txt" "$Q" --lines 2501-5000 --chunk 6000
run dig_ws1_c "$D/ws1.sel.txt" "$Q" --lines 7501-10000 --chunk 6000
run dig_ws2_b "$D/ws2.sel.txt" "$Q" --lines 2501-5000 --chunk 6000
run dig_ws2_a "$D/ws2.sel.txt" "$Q" --lines 1-2500 --chunk 6000
run dig_ws2_c "$D/ws2.sel.txt" "$Q" --lines 5001-7560 --chunk 6000
export LR_TWEET=30000 LR_MAXTOK=5000
run eval_all_draft eval_inputs.txt "TASK (ignore the NONE rule; output code only): write a complete Python 3.13 script skilljepa/eval_all.py that runs the production router eval (ops/treeroute_eval.py, including --sid-only) and the skilljepa.py eval on the existing data/eval-*.json splits (prompt, session, time, prevskill variants) via subprocess, collects Hit@1, Hit@3, abstain accuracy and latency per split into one dict, writes skilljepa/data/eval-<YYYYMMDD>.json and prints a markdown table; --compare PATH prints the delta against an older json. Reuse the CLI flags and output formats you see in the code; do not invent data formats." --chunk 9000
powershell -NoProfile -ExecutionPolicy Bypass -File stop.ps1 >> $LOG
echo "end b $(date +%H:%M) server stopped" >> $LOG; touch DONE4070b
