[part 1, lines 1-475]
```python
#!/usr/bin/env python3
"""eval_all.py (2026-10-25): run the production router eval and skilljepa.py eval on existing data/eval-*.json splits.

Usage:
  py -3.13 eval_all.py [--compare PATH]

Runs ops/treeroute_eval.py (with --sid-only) and skilljepa/skilljepa.py eval for all split variants, collects metrics into one dict, writes skilljepa/data/eval-<YYYYMMDD>.json, prints a markdown table.
"""
import os, sys, json, time, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLJEPA_DIR = os.path.join(HERE)  # assuming eval_all.py is in skilljepa/
OPS_DIR = os.path.join(SKILLJEPA_DIR, "..", "ops")
DATA_DIR = os.path.join(SKILLJEPA_DIR, "data")

def run_treeroute_eval():
    """Run ops/treeroute_eval.py with --sid-only and return parsed results."""
    script_path = os.path.join(OPS_DIR, "treeroute_eval.py")
    if not os.path.exists(script_path):
        print(f"Warning: {script_path} not found", file=sys.stderr)
        return {}

    cmd = [sys.executable, "-I", script_path, "--sid-only"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"treeroute_eval.py failed with code {result.returncode}", file=sys.stderr)
            return {}

        # Parse the output table from treeroute_eval.py (L145-L147 format)
        results = {}
        for line in result.stdout.splitlines():
            if "router" not in line and "Hit@1" not in line:
                parts = line.strip().split()
                if len(parts) >= 3:
                    router_name = parts[0]
                    hit1 = float(parts[1])
                    hit3 = float(parts[2])
                    n = int(parts[3]) if len(parts) > 3 else 0
                    results[router_name] = {
                        "Hit@1": round(hit1, 3),
                        "Hit@3": round(hit3, 3),
                        "n": n
                    }
        return {"treeroute_sid_only": results}
    except subprocess.TimeoutExpired:
        print("treeroute_eval.py timed out", file=sys.stderr)
        return {}

def run_skilljepa_eval(split="session", ctx="none", labels="all"):
    """Run skilljepa/skilljepa.py eval for a given split/ctx/labels combination."""
    script_path = os.path.join(SKILLJEPA_DIR, "skilljepa.py")
    if not os.path.exists(script_path):
        print(f"Warning: {script_path} not found", file=sys.stderr)
        return {}

    cmd = [sys.executable, "-I", script_path, "eval", "--split", split, "--ctx", ctx, "--labels", labels]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"skilljepa eval failed for {split}/{ctx}/{labels}: code {result.returncode}", file=sys.stderr)
            return {}

        # Parse JSON output from cmd_eval (L428-L435 format)
        try:
            data = json.loads(result.stdout.strip())
            key = f"skilljepa_{split}_{ctx}_{labels}"
            return {key: data}
        except json.JSONDecodeError as e:
            print(f"Failed to parse skilljepa eval output for {split}/{ctx}/{labels}: {e}", file=sys.stderr)
            return {}
    except subprocess.TimeoutExpired:
        print(f"skilljepa eval timed out for {split}/{ctx}/{labels}", file=sys.stderr)
        return {}

def collect_all_evals():
    """Run all eval variants and collect results into one dict."""
    all_results = {}

    # Run treeroute eval with --sid-only
    tr_results = run_treeroute_eval()
    if tr_results:
        all_results.update(tr_results)

    # Define split variants to test (from skilljepa.py L467-L469, L369-L378)
    splits = ["session", "prompt", "time"]
    ctxs = ["none", "prev", "prevskill", "cat", "sep"]  # from L468
    labels_variants = ["all", "clean"]  # from L469

    for split in splits:
        for ctx in ctxs:
            for labels in labels_variants:
                eval_results = run_skilljepa_eval(split=split, ctx=ctx, labels=labels)
                if eval_results:
                    all_results.update(eval_results)

    return all_results

def extract_metrics_for_table(all_results):
    """Extract Hit@1, Hit@3, abstain accuracy and latency per split for markdown table."""
    rows = []

    # Process treeroute results (L145-L147 format: router -> [hit1, hit3, n])
    if "treeroute_sid_only" in all_results:
        tr_data = all_results["treeroute_sid_only"]
        for router_name, metrics in sorted(tr_data.items()):
            rows.append({
                "source": "treeroute",
                "variant": router_name,
                "Hit@1": metrics.get("Hit@1", 0),
                "Hit@3": metrics.get("Hit@3", 0),
                "abstain_acc": None,  # treeroute doesn't have abstain accuracy in same format
                "latency_ms": None,   # not measured in treeroute_eval.py output
                "n": metrics.get("n", 0)
            })

    # Process skilljepa results (L428-L435 format: out dict with hit1, hit3, acc_with_abstain, etc.)
    for key, data in sorted(all_results.items()):
        if not key.startswith("skilljepa_"):
            continue

        split = "unknown"
        ctx = "none"
        labels = "all"
        parts = key.split("_")
        if len(parts) >= 4:
            # skilljepa_<split>_<ctx>_<labels>
            split = parts[1]
            ctx = parts[2]
            labels = parts[3]

        for model_name, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            hit1 = metrics.get("hit1", 0)
            hit3 = metrics.get("hit3", 0)
            abstain_acc = metrics.get("acc_with_abstain", None)

            # Latency is not directly in the eval output; estimate or mark as N/A
            latency_ms = None

            rows.append({
                "source": f"skilljepa_{split}_{ctx}_{labels}",
                "variant": model_name,
                "Hit@1": round(hit1, 3) if isinstance(hit1, (int, float)) else hit1,
                "Hit@3": round(hit3, 3) if isinstance(hit3, (int, float)) else hit3,
                "abstain_acc": round(abstain_acc, 3) if abstain_acc is not None and isinstance(abstain_acc, (int, float)) else abstain_acc,
                "latency_ms": latency_ms,
                "n": data.get("n", {}).get("prompts", 0) if isinstance(data.get("n"), dict) else 0
            })

    return rows

def print_markdown_table(rows):
    """Print a markdown table of results."""
    if not rows:
        print("No eval results collected.")
        return

    # Header
    header = "| Source | Variant | Hit@1 | Hit@3 | Abstain Acc | Latency (ms) | N |"
    sep = "|--------|---------|-------|-------|-------------|--------------|---|"
    print(header)
    print(sep)

    for row in rows:
        src = row["source"]
        var = row["variant"]
        h1 = f"{row['Hit@1']:.3f}" if isinstance(row["Hit@1"], (int, float)) else str(row["Hit@1"])
        h3 = f"{row['Hit@3']:.3f}" if isinstance(row["Hit@3"], (int, float)) else str(row["Hit@3"])
        aa = f"{row['abstain_acc']:.3f}" if row["abstain_acc"] is not None and isinstance(row["abstain_acc"], (int, float)) else "N/A"
        lat = f"{row['latency_ms']:.1f}" if row["latency_ms"] is not None else "N/A"
        n = str(row["n"])

        print(f"| {src} | {var} | {h1} | {h3} | {aa} | {lat} | {n} |")

def compare_with_older_json(all_results, older_path):
    """Print delta against an older JSON file."""
    if not os.path.exists(older_path):
        print(f"Older eval file not found: {older_path}", file=sys.stderr)
        return

    try:
        with open(older_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to read older eval file {older_path}: {e}", file=sys.stderr)
        return

    # Compare key metrics for matching keys
    deltas = {}
    for key in all_results:
        if key not in old_data:
            continue

        new_metrics = all_results[key]
        old_metrics = old_data[key]

        delta_entry = {}
        for metric_name in ["hit1", "Hit@1", "hit3", "Hit@3", "acc_with_abstain"]:
            if isinstance(new_metrics, dict) and isinstance(old_metrics, dict):
                new_val = new_metrics.get(metric_name)
                old_val = old_metrics.get(metric_name)

                # Handle nested model dicts from skilljepa eval output (L428-L431 format)
                if metric_name in ("hit1", "Hit@1"):
                    for model_key in new_metrics:
                        if isinstance(new_metrics[model_key], dict):
                            nm = new_metrics[model_key].get("hit1") or new_metrics[model_key].get("Hit@1")
                            om = old_metrics.get(model_key, {}).get("hit1") or old_metrics.get(model_key, {}).get("Hit@1") if isinstance(old_metrics.get(model_key), dict) else None
                            if nm is not None and om is not None:
                                delta_entry[f"{model_key}_hit1"] = round(nm - om, 3)

                elif metric_name in ("hit3", "Hit@3"):
                    for model_key in new_metrics:
                        if isinstance(new_metrics[model_key], dict):
                            nm = new_metrics[model_key].get("hit3") or new_metrics[model_key].get("Hit@3")
                            om = old_metrics.get(model_key, {}).get("hit3") or old_metrics.get(model_key, {}).get("Hit@3") if isinstance(old_metrics.get(model_key), dict) else None
                            if nm is not None and om is not None:
                                delta_entry[f"{model_key}_hit3"] = round(nm - om, 3)

                elif metric_name == "acc_with_abstain":
                    for model_key in new_metrics:
                        if isinstance(new_metrics[model_key], dict):
                            nm = new_metrics[model_key].get("acc_with_abstain")
                            om = old_metrics.get(model_key, {}).get("acc_with_abstain") if isinstance(old_metrics.get(model_key), dict) else None
                            if nm is not None and om is not None:
                                delta_entry[f"{model_key}_abstain_acc"] = round(nm - om, 3)

        # For treeroute format (flat hit1/hit3 per router)
        elif isinstance(new_metrics, dict):
            for router_name in new_metrics:
                if isinstance(new_metrics[router_name], dict):
                    nm_h1 = new_metrics[router_name].get("Hit@1") or new_metrics[router_name].get("hit1")
                    om_h1 = old_metrics.get(router_name, {}).get("Hit@1") or old_metrics.get(router_name, {}).get("hit1") if isinstance(old_metrics.get(router_name), dict) else None

                    nm_h3 = new_metrics[router_name].get("Hit@3") or new_metrics[router_name].get("hit3")
                    om_h3 = old_metrics.get(router_name, {}).get("Hit@3") or old_metrics.get(router_name, {}).get("hit3") if isinstance(old_metrics.get(router_name), dict) else None

                    if nm_h1 is not None and om_h1 is not None:
                        delta_entry[f"{router_name}_Hit@1"] = round(nm_h1 - om_h1, 3)
                    if nm_h3 is not None and om_h3 is not None:
                        delta_entry[f"{router_name}_Hit@3"] = round(nm_h3 - om_h3, 3)

        if delta_entry:
            deltas[key] = delta_entry

    # Print deltas as markdown table
    print("\n## Delta vs older eval\n")
    print("| Key | Metric | Delta |")
    print("|-----|--------|-------|")
    for key in sorted(deltas):
        for metric, delta in sorted(deltas[key].items()):
            sign = "+" if delta > 0 else ""
            print(f"| {key} | {metric} | {sign}{delta:.3f} |")

def main():
    args = sys.argv[1:]
    compare_path = None
    for i, arg in enumerate(args):
        if arg == "--compare" and i + 1 < len(args):
            compare_path = args[i + 1]

    # Collect all eval results
    print("Running production router eval (ops/treeroute_eval.py --sid-only)...", file=sys.stderr)
    all_results = collect_all_evals()

    if not all_results:
        print("No eval results collected.", file=sys.stderr)
        sys.exit(1)

    # Write output JSON with date stamp (L435 format adapted for combined eval)
    today = datetime.datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(DATA_DIR, f"eval-{today}.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=1)
    print(f"Wrote {out_path}", file=sys.stderr)

    # Extract metrics for markdown table
    rows = extract_metrics_for_table(all_results)

    # Print markdown table
    print_markdown_table(rows)

    # If --compare PATH given, print delta against older JSON
    if compare_path:
        compare_with_older_json(all_results, compare_path)

if __name__ == "__main__":
    main()
```

-- localread: eval_inputs.txt, lines 1-475 of 475, 1 part(s), 102s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
