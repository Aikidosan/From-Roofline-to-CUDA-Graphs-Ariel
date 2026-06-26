# Tasks — From Roofline to CUDA Graphs

Checklist of the assignment work across the three notebooks. Status reflects the
committed notebooks: every implementation cell is filled (no `NotImplementedError`
left), every code cell is executed, all writeups answered, and all `results/`
artifacts produced.

Legend: `[x]` done · `[ ]` open

---

## HW1 — Roofline & Arithmetic Intensity
[`hw1_roofline.ipynb`](hw1_roofline.ipynb) · L1 §06 (Roofline) · L1 §03 (GPU memory hierarchy)

- [x] **1a** `lowest_ai_fn` — lowest-arithmetic-intensity elementwise op
- [x] **1b** `make_compute_fn` — tunable-K kernel, works eager **and** compiled
- [x] **2** `benchmark_fn` — correct CUDA-event timing with warmup
- [x] **3** `compute_metrics` — measurements → (AI, TFLOP/s, GB/s)
- [x] **4** Writeup Q1 (memory- vs compute-bound from the plot)
- [x] **4** Writeup Q2 (why AI rises with N for an N×N×N matmul)
- [x] Self-checks pass
- [x] Roofline run on GPU → [`results/hw1/roofline.png`](results/hw1/roofline.png), [`results/hw1/roofline_data.json`](results/hw1/roofline_data.json)

## HW2 — Profile & Optimize an Autoregressive Decode Loop
[`hw2_decode_optimization.ipynb`](hw2_decode_optimization.ipynb) · L1 §07 (Profiling) · L2 §01 (KV cache) · L2 §03 (engine opts)

- [x] **1** `profile` — wrap a loop in `torch.profiler`, print table, export Chrome trace
- [x] **2** `optimized_loop` — fast greedy decode, fp32-identical tokens to V0
- [x] **3** `generate_optimized` — build, warm up, time the optimized generation
- [x] **4** Writeup Q1 (what dominates the baseline trace)
- [x] **4** Writeup Q2 (per-optimization speedup table: +KV cache → +compile → …)
- [x] **4** Writeup Q3 / Q4
- [x] Self-checks pass (fp32 correctness vs V0)
- [x] Speedup target met — end-to-end ≈7.8× vs V0 on H100 (target ≥4.0× = excellent)
- [x] Traces exported → [`results/hw2/`](results/hw2/) (`trace_baseline.json`, `trace_optimized.json`, `trace_check.json`)

## HW3 — `torch.compile` & CUDA Graphs
[`hw3_compile_cuda_graphs.ipynb`](hw3_compile_cuda_graphs.ipynb) · L2 §03 (compilation, CUDA graphs) · L1 §07 (profiling)

- [x] **1** `sweep_elementwise_times` + `estimate_launch_overhead_us` (per-launch overhead ≈6.8 µs on H100)
- [x] **2** `fixed_decode_step` — same math as the broken step, zero graph breaks (`fullgraph=True`)
- [x] **3** `make_graphed_callable` — manual CUDA-graph capture + replay
- [x] **4** Writeup Q1 (`.item()` / Python `if` / `print` graph breaks; why `.item()` is doubly bad)
- [x] **4** Writeup Q2 (why replay helps small-batch decode; capture/replay constraints)
- [x] **4** Writeup Q3 (when `torch.compile` doesn't help, and how to detect it)
- [x] Self-checks pass
- [x] Final 256-step benchmark on GPU — manual CUDA graph ≈4.0× over eager (H100)
- [x] Plots produced → [`results/hw3/launch_overhead.png`](results/hw3/launch_overhead.png), [`results/hw3/decode_step_latency.png`](results/hw3/decode_step_latency.png)

---

## Submission
- [x] All notebooks run top-to-bottom on GPU with outputs included
- [x] `results/` artifacts committed
- [x] [`README.md`](README.md) summarizing hw1–hw3

> Notebooks run on the `py313torch` kernel. (Avoid the 3.14 env — `torch.compile`
> is broken there.)

---

## GPU VM — Nebius H100 (2026-06-25)

Reconnected the GPU box and stood up a remote environment for running the
notebooks on real hardware.

- [x] **SSH access** — fresh instance reachable via `ssh h100` (`~/.ssh/config`)
  - User `arielmit`, key `~/.ssh/nebius_h100` (ed25519), IP `195.242.30.120`
  - Old hosts (`nebius-gpu-vm` 89.169.122.93, `hw2` 89.169.122.70) are deprovisioned
- [x] **Lesson learned** — Nebius/cloud-init injects SSH keys only at *first boot*;
  editing keys on a running VM + reboot does **not** rewrite `authorized_keys`.
  Fix: recreate the instance with the key in cloud-init, and the user-data MUST
  start with a literal `#cloud-config` line or the `users:` block is ignored.
- [x] **Hardware** — 1× H100 80GB SXM, driver 580.159.04, image
  `ubuntu24.04-cuda13.0` (CUDA 13.0.3), Python 3.12, 16 vCPU / 196GB RAM / 1.2TB
- [x] **Environment** — venv `~/venvs/torch` ready: PyTorch 2.12.1+cu130 (CUDA
  avail), numpy 2.5.0, matplotlib 3.11.0, jupyterlab 4.6.0, ipykernel 7.3.0,
  transformers 5.12.1. Also installed `python3.12-dev` so Inductor/torch.compile
  finds `Python.h` (without it the compiled cells fall back silently).
- [x] **Re-run notebooks on the H100** — executed headless via
  `jupyter nbconvert --execute` over SSH (notebooks staged in `~/repo`), refreshed
  all `results/` artifacts and copied them back. Final rigorous run had GPU clocks
  locked to 1980 MHz (`sudo nvidia-smi -pm 1; -lgc 1980`) for reproducible timing,
  plus a warmup/iters bump in HW2 `generate_optimized`. H100 figures:
  HW1 launch overhead ≈6.8 µs (matmul ≈787 TFLOP/s) · HW2 ≈7.8× vs V0 · HW3 manual
  CUDA graph ≈4.0× over eager. Clock-locking confirmed the GPU was already at its
  boost ceiling — numbers are at the hardware limit, not throttled (run-to-run
  spread ~3–6%). Lower multipliers than the earlier run reflect newer PyTorch's
  ~6.8 µs (vs ~19 µs) launch overhead making the baselines faster, not worse work.
- [ ] **Remote Jupyter for VS Code** *(optional)* — not needed for the re-run
  above (headless execution covered it). To work interactively: register venv
  kernel, launch JupyterLab on `localhost:8888`, SSH-forward 8888, connect VS Code
  → *Existing Jupyter Server*.
