# From Roofline to CUDA Graphs — Ariel Mitiushkin

A hands-on assignment on **measuring and optimizing GPU inference**, from raw kernel
performance up to a fully launch-bound autoregressive decode loop. Nebius Academy.

The work is split across three notebooks, each building on the last: first you learn
to *see* where a kernel sits relative to the hardware limits, then you fix a slow
generation loop, then you remove the last bottleneck — kernel launch overhead — at
batch size 1.

---

## HW1 — Roofline analysis

[`hw1_roofline.ipynb`](hw1_roofline.ipynb)

Build kernels spanning a range of **arithmetic intensities** (FLOPs per byte), time
them with **CUDA events**, and plot the achieved throughput on the **roofline** for
your card to see what is memory-bound vs compute-bound.

- An elementwise op (very low AI) → memory-bound, far left under the bandwidth ceiling.
- A `compute-K` kernel swept across K (rising AI) → walks up the diagonal.
- A `matmul-N` swept across N (high AI) → approaches the compute ceiling.

It also contrasts **eager vs `torch.compile`**: at the same arithmetic intensity, the
compiled kernels achieve dramatically higher throughput (fusion removes memory
round-trips), pushing points up toward the roofline.

Outputs: [`results/hw1/roofline.png`](results/hw1/roofline.png),
[`results/hw1/roofline_data.json`](results/hw1/roofline_data.json).

## HW2 — Profile & optimize an autoregressive decode loop

[`hw2_decode_optimization.ipynb`](hw2_decode_optimization.ipynb)

Profile a deliberately slow greedy-decode baseline (**V0**) and rewrite it into a
fast, **numerically identical** version.

The baseline's three sins:
- **No KV cache** — recomputes the whole sequence every step.
- **fp32 eager** — no fusion, full-precision math.
- **A host sync every step** — `.item()` stalls the CPU on the GPU each token.

The optimized loop fixes all three: a **KV cache** (process one new token per step),
**fewer host syncs**, **`torch.compile`**, and **bf16**. Part 1 wraps both loops in
`torch.profiler` and exports Chrome traces (view at
[ui.perfetto.dev](https://ui.perfetto.dev)); Part 2 writes the correct fast loop;
Part 3 builds and times it against V0.

Outputs: Chrome traces in [`results/hw2/`](results/hw2/)
(`trace_baseline.json`, `trace_optimized.json`, `trace_check.json`).

## HW3 — `torch.compile` & CUDA Graphs

[`hw3_compile_cuda_graphs.ipynb`](hw3_compile_cuda_graphs.ipynb)

At batch size 1 a decode step is hundreds of tiny kernels, each finishing faster than
the CPU can launch the next — so generation speed is set by **launch overhead**, not
FLOPs or bandwidth. Measure that overhead and remove it two ways.

- **Part 1** — measure per-launch overhead by sweeping an elementwise op across sizes;
  find the knee below which a kernel is effectively free (launch-bound).
- **Part 2** — `torch.compile`: what graph breaks are, how to find them with
  `torch._dynamo.explain`, and compiling a decode step into a **single graph with no
  graph breaks**.
- **Part 3** — build a **CUDA graph by hand**: capture the decode step's exact kernel
  sequence once, then replay it with a single launch (N launches → 1).

Outputs: [`results/hw3/launch_overhead.png`](results/hw3/launch_overhead.png),
[`results/hw3/decode_step_latency.png`](results/hw3/decode_step_latency.png).

---

## Results (1× H100 80GB SXM, PyTorch 2.12.1+cu130)

Headline figures from the latest end-to-end run on the Nebius H100, with the GPU
clocks **locked to 1980 MHz** (`nvidia-smi -lgc`) for reproducible timing:

| HW | Metric | Result |
|----|--------|--------|
| HW1 | Per-launch kernel overhead | ≈6.8 µs |
| HW1 | Matmul peak (N=4096) | ≈787 TFLOP/s (~80% of bf16 peak) |
| HW2 | Optimized decode vs baseline **V0** | ≈7.8× (target ≥4× = excellent) |
| HW3 | Manual CUDA graph vs eager decode step | ≈4.0× (fixed step compiles `fullgraph=True`, 0 graph breaks) |

Exact numbers vary with PyTorch version and GPU (run-to-run spread is ~3–6% even
with locked clocks); rerun the notebooks to refresh the `results/` artifacts for
your hardware. Note newer PyTorch has lower kernel-launch overhead (≈6.8 µs here
vs ≈19 µs on older builds), which makes the eager/V0 baselines faster and so
*shrinks* the CUDA-graph speedup multipliers — a faster baseline, not worse work.

---

## Running

Notebooks run on a CUDA GPU with a recent PyTorch. Open each `.ipynb` and run top to
bottom; the "DO NOT EDIT" harness cells set up the model, correctness checks, and
timing helpers, and the parts you implement are marked inline.
