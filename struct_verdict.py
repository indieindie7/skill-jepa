"""struct_verdict.py T1|T3: re-verify holdout hash, print VERDICT {gain,note} from data/struct-arms-20261029.json (autorun ideas.py convention). CPU only."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import holdout_check as hc
ok, msg = hc.verify(); r = json.load(open(os.path.join(HERE, "data", "struct-arms-20261029.json")))
arm = sys.argv[1]
if not ok: v = {"gain": 0, "note": "holdout hash mismatch " + msg}
elif arm == "T1":
    o = r["T1"]["oracle_skill"]; v = {"gain": round(o["branch_hit3_bm25"] - o["branch_listed_in_prod_brief8"], 3), "note": f"branch top3 BM25 {o['branch_hit3_bm25']:.2f} vs prod 8-listed {o['branch_listed_in_prod_brief8']:.2f}; hit1 {o['branch_hit1_bm25']:.2f} vs {o['branch_hit1_prod_brief_order']:.2f}; n={o['n']} (transcript GT, small)"}
else:
    t = r["T3"]; v = {"gain": round(1 - t["tokens_top3_lines"] / t["tokens_brief_top1_A0"], 3), "note": f"tokens {t['tokens_top3_lines']:.0f} vs brief {t['tokens_brief_top1_A0']:.0f}; quality proxy {t['proxy_ref_hits_opened_branch']} n={t['proxy_n']} only: NO quality ground truth, needs review"}
print("VERDICT " + json.dumps(v))
