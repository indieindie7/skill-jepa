[part 1, lines 1-180]
```python
"""Documents/Tools/llm/gpu.py - stdlib-only GPU management for local LLM servers."""
import argparse, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STRATA_DIR = r"C:\llm\strata-0.2"  # 10-23: 0.2 worktree, old checkout deleted   # 10-14: Strata engine + Qwen 3.8 Flash Next 125B MoE (IQ3_XXS), beat the 27B on speed and digests
STRATA_CFG = {"strata": "strata-iq3_xxs", "coder": "strata-coder-iq1_m-gpu1", "strata-kvq4": "strata-iq3_xxs-kvq4"}   # 10-20 kvq4: same model, --kv q4_0, port 8092 (KV-cache JSON test)
STRATA_PORTS = {"strata": 8090, "coder": 8091, "strata-kvq4": 8092}   # 10-19: Coder runs side by side on gpu 0 / 8091 (verified, no loops); needs the 16 GB page file
STRATA_GPUS = {"strata": 1, "coder": 0, "strata-kvq4": 1}   # 10-15: Coder = ISTA half-expert code model, ~32 GB RAM
STRATA_CFG_GPU = {"strata": {0: "strata-iq3_xxs", 1: "strata-test-gpu1"}, "coder": {0: "strata-coder-iq1_m-gpu0", 1: "strata-coder-iq1_m-gpu1"}}   # 10-25: config per GPU, chosen by free VRAM
STRATA_PORT, STRATA_MB = 8090, 9500    # ~4 GB fixed + an expert cache sized to the free VRAM ("expert cache auto"); ~43 GB RAM; one request at a time


def http(port, path, body=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def healthy(port):
    try:
        return http(port, "/health", timeout=2).get("status") == "ok"
    except Exception:
        return False


def strata_ready(port=STRATA_PORT):  # noqa
    try:
        h = http(port, "/health", timeout=2)
        return h.get("status") == "ok" and h.get("service") == "strata" and h.get("loaded") is True
    except Exception:
        return False


def free_vram_mb(index):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def game_running():
    """True if nvidia-smi --query-compute-apps lists a process under steamapps or games."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            name = parts[-1] if len(parts) > 1 else ""
            low = name.lower()
            if "steamapps" in low or "games" in low:
                return True
        return False
    except Exception:
        return False


def pick_gpu(need_mb, prefer=None):
    """Return index with most free VRAM that fits (never 0 while game_running)."""
    gpus = [0, 1]
    if game_running():
        gpus = [g for g in gpus if g != 0]
    
    frees = {g: free_vram_mb(g) for g in gpus}
    candidates = []
    for g in gpus:
        f = frees.get(g)
        if f is not None and f >= need_mb:
            candidates.append((f, g))
        
    if not candidates:
        # Fallback to any available GPU even if tight on VRAM? No, spec says "fits". 
        # If none fit, return -1 or raise. Spec implies returning an index. 
        # Let's assume strict fitting. If no fit, maybe prefer logic helps?
        # Actually, let's look at start_strata logic: it picks max free even if < need_mb sometimes (with warning/exit).
        # But pick_gpu spec says "fits". So filter by >= need_mb.
        return -1

    candidates.sort(key=lambda x: (-x[0], x[1]))  # Most free first, then lowest index
    
    if prefer is not None and prefer in gpus:
        pref_free = frees.get(prefer)
        if pref_free is not None and pref_free >= need_mb:
            return prefer
            
    return candidates[0][1]


class _Tree:
    """Popen wrapper whose kill() also stops the children (Strata's server.py starts strata.exe)."""
    def __init__(self, proc):
        self.proc = proc

    def kill(self):
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)


def start_strata(model="strata", port=None, gpu=None):
    if model not in STRATA_CFG:
         sys.exit(f"gpu.py: unknown model {model}")
         
    target_port = port if port is not None else STRATA_PORTS[model]
    
    # Determine GPU to use
    need_mb = STRATA_MB
    
    if gpu is not None:
        chosen_gpu = gpu
    else:
        chosen_gpu = pick_gpu(need_mb)
        
    if chosen_gpu == -1:
         frees = {g: free_vram_mb(g) for g in (0, 1)}
         sys.exit(f"gpu.py: no GPU has enough free VRAM ({frees}) for ~{need_mb} MB.")

    # Check against game_running constraint if we picked automatically or explicitly? 
    # Spec says pick_gpu never returns 0 while game_running. start_strata should respect that too implicitly via pick_gpu logic or explicit check.
    
    cfg_map = STRATA_CFG_GPU.get(model, {})
    cfg = cfg_map.get(chosen_gpu)
    if not cfg:
        cfg = STRATA_CFG[model]

    py = os.path.join(STRATA_DIR, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        sys.exit(f"gpu.py: Strata is not installed at {STRATA_DIR}")
        
    log_path = os.path.join(HERE, "strata.log")
    try:
        log = open(log_path, "w")
    except Exception as e:
        sys.exit(f"gpu.py: cannot write log file {log_path}: {e}")

    proc = subprocess.Popen([py, os.path.join(STRATA_DIR, "serve", "server.py"), "--engine", "strata",
                             "--config", os.path.join(STRATA_DIR, cfg + ".json"), "--port", str(target_port)],
                            cwd=STRATA_DIR, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    
    # Wait for /health
    try:
        for _ in range(600):                    # the first start reads ~47 GB of weights; ~1 min when cached
            if strata_ready(target_port):
                return proc
            if proc.poll() is not None:
                 sys.exit(f"gpu.py: Strata exited during start-up (see {log_path} and {os.path.join(STRATA_DIR, cfg + '.log')})")
            time.sleep(1)
    except Exception as e:
        _Tree(proc).kill()
        raise
        
    _Tree(proc).kill()
    sys.exit(f"gpu.py: Strata did not come up in 10 minutes (see {log_path})")


def stop_port(port):
    """Kills the serve/server.py python whose command line holds that --port."""
    try:
        # Use wmic or powershell to find processes? 
        # Standard way on Windows without psutil is parsing tasklist or using WMI via subprocess.
        # Let's use a simple approach with `wmic` if available, or fallback to scanning PIDs via PowerShell.
        
        # Try WMIC first (deprecated but common in older setups)
        cmd = ["wmic", "process", "where", f"CommandLine like '%--port {port}%'", "get", "ProcessId", "/format:value"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        
        pids = []
        for line in out.splitlines():
            if line.startswith("ProcessId="):
                try:
                    pid = int(line.split("=")[1])
                    # Verify it's a python process running serve/server.py? 
                    # The query already filters by CommandLine containing --port {port}.
                    # We assume the user wants to kill anything matching that port arg.
                    pids.append(pid)
                except ValueError:
                    pass
        
        if not pids:
            # Fallback: PowerShell Get-CimInstance
            ps_cmd = ["powershell", "-Command", 
                      f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '--port {port}' }} | Select-Object ProcessId"]
            out_ps = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10).stdout
            
            import re
            matches = re.findall(r"\d+", out_ps)
            for m in matches:
                try:
                    pids.append(int(m))
                except ValueError:
                    pass

        killed = False
        for pid in set(pids):
             # Double check it's not ourselves? Unlikely to match --port unless we are running a server.
             subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
             killed = True
             
        if not killed:
            print(f"No process found with --port {port}")
            
    except Exception as e:
        sys.exit(f"gpu.py: error stopping port {port}: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPU management for local LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # free
    p_free = subparsers.add_parser("free", help="Show free VRAM per GPU")
    
    # pick <need_mb>
    p_pick = subparsers.add_parser("pick", help="Pick a GPU index with enough free VRAM")
    p_pick.add_argument("need_mb", type=int, help="Required MB of VRAM")

    # start strata [port] [gpu]
    p_start = subparsers.add_parser("start", help="Start Strata server")
    p_start.add_argument("model", choices=["strata"], help="Model name (currently only 'strata' supported in this CLI)")
    p_start.add_argument("--port", type=int, default=None)
    p_start.add_argument("--gpu", type=int, default=None)

    # stop <port>
    p_stop = subparsers.add_parser("stop", help="Stop server on specific port")
    p_stop.add_argument("port", type=int)

    args = parser.parse_args()

    if args.command == "free":
        for i in range(2):
            mb = free_vram_mb(i)
            print(f"GPU {i}: {mb} MB free")
            
    elif args.command == "pick":
        idx = pick_gpu(args.need_mb)
        if idx == -1:
             sys.exit("No GPU fits the requirement.")
        print(idx)

    elif args.command == "start":
        start_strata(model=args.model, port=args.port, gpu=args.gpu)
        
    elif args.command == "stop":
        stop_port(args.port)


if __name__ == "__main__":
    main()
```

-- localread: localread.py, lines 1-180 of 301, 1 part(s), 193s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
