The user proposed using Evolution Strategies (ES) and EGGROLL for fine-tuning a local **decider-2b** model on a 12 GB card, citing papers 1703.03864, 2509.24372, and 2511.16652 (L6757-L6760). The method uses Gaussian weight noise or LoRA-shaped low-rank perturbations with scalar rewards from **"right or wrong" rule labels** in the **`action field`**, requiring no backprop/gradients (L6881-L6892, L6772-L6774).

ES/EGGROLL was chosen over QLoRA/GRPO because it fits better on a 12 GB card than GRPO and suits non-differentiable game sim fitness for the TRM enemy brain (L6770-L6772, L6839-L6840). Performance benchmarks showed EGGROLL at **91** vs PPO **34** vs OpenES **0.41**, with Countdown accuracy of **35%** (EGGROLL) vs **23%** (GRPO), and a 14B model gaining **+17 AIME24/+26 AIME25** in 12h on 32 GPUs (L6757-L6760).

The user decided to **"lets experiment es and eggroll"** (L6843-L6851).

-- localread: ws1.sel.txt, lines 6650-6950 of 10050, 2 part(s), 52s, Strata qwen3.8-flash-next-iq3_xxs on port 8090. Verify cited lines before relying on them.
