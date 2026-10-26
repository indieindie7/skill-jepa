Based on the provided notes from Parts 5, 9, 11, and 12, here is the consolidated report regarding the requested topics. Note that specific mentions of a "skill-jepa router," GRPO/distillation training loops, or a trainable mixture-of-experts model for diverse thinking are **absent** in these extracts; however, related components like local routers (Effort Pacer), small judge models (Decider/Recall Judge), and GPU usage rules are present.

### 1. Local Models, Ports, GPUs, and Routers
*   **GPU & Environment:** RTX 4070 Ti (12 GB VRAM) on Win10 [L3521-L3523]. Rule: No gaming while GPU jobs run [Part 9]. Rule: Switch models if they use less VRAM with equal scores (`gpu-lower-vram-swap-rule`) [Part 9].
*   **Ports:**
    *   `8084`: CPU embedding server (PID 19076) [L3358-L3371]; required for data generation in recall judge [L4875, L4902-L4907].
    *   `8083`: Gemma 12B [Part 9].
    *   `8085`: Used for Mimo/Gemma benchmarks via `bench.py` and later for `mmsearch.py` with EmbeddingGemma 2 [L3342-L3360, Part 9].
*   **Effort Pacer (Local Router/Usage Control):**
    *   **Design:** Scheduled task reads usage via `mcp__ccd_session_mgmt__get_usage`, lists sessions, runs `pacer.py plan` to set effort on *other* sessions. Cannot set own session effort [L3362].

-- localread: ws1.sel.txt, lines 2501-5000 of 10050, 13 part(s), 207s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
