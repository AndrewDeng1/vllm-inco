# Inco take-home — Track A (Engine & Model)

**Short-form code generation on Qwen3-30B-A3B-Instruct, one H100 80GB, vLLM.**

Two stacked optimizations against a measured baseline: **REAP 50% expert
pruning** to turn dead weight memory into KV cache, then a **DFlash2
speculative-decoding drafter** trained on-policy against the pruned target.

| | baseline | + REAP 50% | + REAP + DFlash2 |
|---|---|---|---|
| weights | 58.3 GiB | 31.3 GiB | 31.3 GiB |
| KV cache | 12 GiB | 38 GiB | 32 GiB |
| max batch | 85 | 270 | 270 |
| peak tok/s/gpu, ISL 1024 / OSL 512 | 3779 (c=80) | **7738** (c=256) | — |
| peak tok/s/gpu, code-eval workload | 3776 (c=96) | 5004 (c=96) | **11,580** (c=96) |
| HumanEval / HumanEval+ pass@1 | 0.9146 / 0.8720 | 0.9146 / 0.8659 | unchanged (lossless) |

**2.05x on the target workload from pruning alone; 3.1x end to end on the code
benchmarks, for ±0.0 pp HumanEval and −0.6 pp HumanEval+.** The cost is
out-of-domain — ARC-Challenge drops 22.9 pp — which is the code calibration
corpus showing through, and for this deployment it is the trade we want.

Below: the workload, what makes the benchmark trustworthy, then every
experiment with the command that reproduces it.

## Workload

| | |
|---|---|
| Task | short-form code generation — autocomplete, short edits |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` — 30.5B total / 3.3B active MoE, bf16, TP=1 |
| GPU | one H100 80GB |
| Shape | ISL **1024** / OSL **512**, `/v1/chat/completions`, streaming |
| Load | client concurrency swept `1,2,4,8,16,24,32,40,48,56,64,72,80` |
| Axes | x = tokens/s/user (interactivity), y = tokens/s/gpu (throughput) |

## What makes the numbers trustworthy

1. **The server launches once** at max batch size; only *client* concurrency is
   swept, so one set of CUDA graphs and one KV cache back the whole curve.
2. **OSL is pinned** with `ignore_eos:true`, thinking mode off. Otherwise
   tokens/s has no fixed denominator.
3. **The prefix cache is flushed between points** via `/reset_prefix_cache`.
4. **The server is audited before it is measured.** The client reads the
   effective `VllmConfig` from `/server_info` and refuses to run if CUDA graphs
   or async scheduling are off, or `max_num_seqs` is below the top of the
   sweep.
5. **8 waves per point**, warmup `max(16, concurrency)` — warmup runs *at* the
   concurrency it warms, so a fixed count below it leaves the target width's
   first-touch cost inside the sample.
6. **Integrity checks are automatic**: fewer than 4 waves at any point, or
   tok/s/user rising with concurrency (physically impossible), fails the run.

Standing flags on every measured run:

| flag | value | why |
|---|---|---|
| `--kv-cache-memory-bytes` | 12 GiB (131,072 tokens) | pinned: Modal's `gpu="H100"` returns either an 80GB HBM3 or a 94GB NVL, which at util 0.90 leaves ~12.5 vs ~25 GiB of KV and doubles servable concurrency between runs |
| `--gpu-memory-utilization` | 0.90 | |
| `--max-model-len` | 4096 | |
| `--max-num-batched-tokens` | 8192 | |
| `--async-scheduling`, `--enable-prefix-caching` | on | audited via `/server_info` before measuring |
| `ignore_eos:true` | on | pins OSL exactly |
| `enable_thinking` | false | via `--default-chat-template-kwargs` |

Concurrencies are drawn from vLLM's `cudagraph_capture_sizes`, so no point pays
graph padding.

**For any A/B, run baseline and variant in the same container.** Measured
cross-instance variation is 2–5%, plus the H100 variant lottery on top.
`gpu_model` is recorded in every `manifest.json` — check it matches before
comparing two runs.

---

# Experiments

Every experiment below is one script in `inco/scripts/experiments/`. They take no
arguments, resolve their own paths, and forward any extra flags to the
underlying Modal entrypoint. Shared paths (repo root, `modal` CLI, the pruned
checkpoint) live in `experiments/_env.sh`.

Modal volumes: `inco-results` (sweeps), `inco-reap` (checkpoints, evals,
probes), `inco-spec` (drafters).

## Setup (once)

```bash
bash inco/scripts/experiments/0-setup.sh
```

Creates the venv, authenticates Modal (browser OAuth), clones the two
gitignored dependencies, and caches 61GB of weights.

Server and client share one container, so aiperf measures the engine rather
than the network. The image installs **this fork** via
`VLLM_USE_PRECOMPILED=1` — the harness depends on config fields and server
flags that only exist in this tree.

`inco/reap/` and `inco/speculators/` are gitignored clones. Each needs its own
virtualenv — both install their *own* vLLM and would otherwise overwrite the
engine under test.

On a bare-metal GPU host instead of Modal: `bash inco/scripts/install.sh`, then
`serve_baseline.sh` in one terminal and `run_baseline.sh` in another.

## 1. Baseline sweep — ISL 1024 / OSL 512

Max batch, from the pinned KV cache:

```
KV cache  12.0 GiB           = 131,072 tokens
sequence  1024 + 512         =   1,536 tokens
max batch 131,072 / 1,536    =      85.3
```

85 is the ceiling; concurrency steps by 8, so the sweep caps at 80.

```bash
bash inco/scripts/experiments/1-baseline.sh
```

~19 min. Defaults are this workload, so a bare `modal run modal_baseline.py`
reproduces it. Plot: `results/chat-1024-512/pareto.png`.

| c | 1 | 8 | 16 | 32 | 48 | 64 | 72 | 80 |
|---|---|---|---|---|---|---|---|---|
| tok/s/user | 210.2 | 110.3 | 88.7 | 70.9 | 61.0 | 55.4 | 50.8 | 49.2 |
| tok/s/gpu | 205 | 851 | 1354 | 2162 | 2805 | 3390 | 3517 | **3779** |

Two observations drive everything after this:

* **Throughput gains are slow early.** Each added request routes to *new*
  experts, so MoE weight traffic grows with batch size instead of staying
  fixed. Quantified in §2 and §3d.
* **Never saturates.** Still climbing at c=80. KV, not compute, is what stops
  it — a 30B bf16 model on 80GB leaves little room for KV cache. Saturation is
  reachable at a shorter shape (§2b), so this is a capacity limit, not an
  engine limit.

## 2. Analysis — is the slow early scaling really the MoE?

`Qwen3-4B-Instruct-2507` is dense, so its weight traffic per step is
batch-independent by construction. Same ISL/OSL, same GPU, as a control.

```bash
bash inco/scripts/experiments/2-dense-control.sh
# -> results/moe-vs-dense-ratio-plots/moe-vs-dense-ratio.png
```

Raw adjacent-concurrency ratios are not comparable because the steps are not
uniform (x2 early, x1.11 late), so the second panel divides by the concurrency
step: 1.00 means adding requests bought proportional throughput.

* The MoE's per-step scaling efficiency bottoms out at **0.733** around
  c=8→16; the dense model is smooth at ~0.95 through the same region.
* They converge from c≈24 on, which is where nearly every expert is being
  touched every pass and the MoE's ratios line up with the dense model's.

So the early deficit is specific to sparse routing, not to batching. Only the
*shape* of the dense curve is used, never its absolute tok/s.

### 2b. Saturation is reachable at a shorter shape

At 256 tokens/request there is far more room for KV cache, so the plateau
becomes visible.

```bash
bash inco/scripts/experiments/2b-short-shape.sh
```

Diminishing returns from **BS ≈ 320**, saturating at **~11,000 tok/s/gpu**.
Marginal gain per added request decays 39.5 → 13.5 → 7.9 → 4.2 → 0.2.

## 3. REAP expert pruning

MoE lets transformers add capacity without adding proportional compute, and
nearly every frontier model is now MoE — but the expert weights are the bulk of
the parameters and they all have to be resident, which is what makes these
models hard to deploy. Experts are not equally useful for every task, so:
**if we prune X% of experts across all MoE layers, how do we preserve as much
performance as possible on one task area?**

REAP (router-weighted expert activation pruning) tracks a saliency score per
expert — the router weight times the magnitude of the expert activation,
averaged over the tokens that activate it:

$$S_j = \frac{1}{|\mathcal{X}_j|} \sum_{x \in \mathcal{X}_j} g_j(x) \cdot \|f_j(x)\|_2$$

Higher saliency means the expert matters more for the calibration data. Per MoE
layer the bottom X% are deleted along with their columns in the router matrix.
`top_k` is unchanged, so the model still routes 8-of-64.

**Calibration: codealpaca, 1024 sequences, seed 42** — chosen because the task
is short-form code generation.

```bash
bash inco/scripts/experiments/3-reap-prune.sh
```

~90 min: ~109 s per decoder block x 48 blocks of calibration, plus a model
reload and checkpoint write. Observations are cached and reused, so a second
prune at a different ratio with the same sample count and seed skips
calibration entirely (~3 min).

`batch_size` is not a free speed knob — the observer materialises activations
for *all* 128 experts, `num_experts x (batch_size x model_max_length) x
hidden_dim x 4`. The default 8 x 2048 is 17.2 GB; 32 x 2048 is 68.7 GB and
OOMs on an 80GB card.

### 3a. Saliency heatmaps

The observer writes every metric it accumulated next to the checkpoint, one
tensor of 128 experts per layer. Pruning consumes only the per-layer ranking,
so the dump is the whole decision and plots without a GPU.

```bash
bash inco/scripts/experiments/3a-saliency.sh
# -> results/saliency/reap-saliency-heatmap.png  (4 panels)
```

**Absolute saliency is not comparable across layers** — the median expert grows
105x from layer 0 to layer 47, because activation norms grow with depth.
Normalizing by the layer *max* flattens layers 1–3 (each holds one expert worth
16–81x its layer median) to near-white, so the plot divides by the layer median
on a log colour scale.

* **Pruning is not index-structured.** Every index quartile retains 49.6–50.2%
  of its experts; no index survives in all 48 layers or dies in all 48. The
  mask panel looks like noise because it is — saliency is a per-layer property,
  not a property of an expert slot.
* **The kept half carries ~2/3 of the saliency mass** (mean 0.669). Deleting
  half the experts removes about a third of the measured saliency, which is the
  premise REAP trades on.
* **Deep layers are more concentrated**: p90/p10 spread widens from ~3.2 at
  layer 0 to ~5.2 at layer 47; 15 of the 23 experts above 4x their layer median
  live in layers 42–47.

### 3b. Impact on max batch size

95% of the parameters are expert weights, which is why pruning experts is the
lever:

| component | params | GiB | share |
|---|---|---|---|
| experts (128 x 48 layers) | 28.99B | 54.00 | **95.0%** |
| attention | 0.91B | 1.69 | 3.0% |
| embed + lm_head | 0.62B | 1.16 | 2.0% |
| router | 0.01B | 0.02 | 0.0% |
| total | 30.53B | 56.87 | |

Freed weight memory does **not** become KV automatically — re-pin it:

| prune | experts kept | weights | freed | KV | KV tokens | max batch |
|---|---|---|---|---|---|---|
| 0% | 128 | 58.3 GiB | – | 12.0 GiB | 131,072 | 85 |
| 38% | 79 | 37.6 GiB | 20.7 | 32.7 GiB | 356,864 | 232 (2.7x) |
| 50% | 64 | 31.3 GiB | 27.0 | 39.0 GiB | 425,984 | 277 (3.3x) |

### 3c. Impact on throughput

```bash
bash inco/scripts/experiments/3c-reap-sweep.sh
```

Same GPU model (H100 80GB HBM3) as the baseline, zero integrity warnings, zero
preemption in either. Plot: `results/reap50-1024-512/pareto.png`.

| | baseline | REAP 50% |
|---|---|---|
| weights | 58.3 GiB | 31.3 GiB |
| KV cache | 12 GiB | 38 GiB |
| max batch | 85 | **270** |
| peak tok/s/gpu | 3779 @ c=80 | **7738 @ c=256** |

**Peak throughput more than doubled (2.05x).** Still does not reach saturation,
but at much higher throughput. The gain separates into two effects:

* **Faster steps: ~+12%, only at high concurrency.** At c=1–2 it is −1%
  (noise) — a token routes to top-8 experts whether the model has 128 or 64, so
  the weight read per step is identical. The gain grows with batch size,
  plateauing near +12.5% by c=64.
* **More batch: 3.2x ceiling.** The dominant effect. Every point from c=96 to
  c=256 is throughput the unpruned model cannot produce at any latency.

So the 2x is mostly capacity, not speed.

### 3d. Why the speed gain only appears at large batch

```bash
bash inco/scripts/experiments/3d-expert-activation.sh
# -> results/experts/expert-activation.png
```

One token routes to `top_k` experts, so a batch of B decoding sequences touches
at most `B * top_k`. A single forward pass over 256 sequences yields every
layer's router logits; decode at batch B is emulated by subsampling B
sequences' final-position decisions over 64 draws, so the whole curve costs one
forward pass rather than one per batch size.

Mean distinct experts activated per MoE layer (48 layers, top-8):

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| unpruned (128 experts) | 8.0 | 14.5 | 24.8 | 40.7 | 59.7 | 78.1 | 94.0 | 104.7 | 112.6 |
| REAP 50% (64 experts) | 8.0 | 14.0 | 22.7 | 34.7 | 46.3 | 54.7 | 59.7 | 61.6 | 62.3 |

At batch 1 both models read exactly 8 experts per layer, so there is no
throughput gain. As batch grows the unpruned model walks toward all 128 while
the pruned one saturates at ~62 — so pruning 50% of experts saves not only
~50% of weight memory but, at large batch, roughly half the MoE memory traffic
and compute per forward pass.

### 3e. Impact on the interactivity floor

Autocomplete needs fast per-user generation, so per-user latency matters a lot
and the honest comparison is SLO-matched. REAP still provides large gains — and
the gain *grows* with the floor you can afford, which is §3d showing up in the
throughput numbers: a higher floor means a larger batch, and a larger batch
means more of the pruned experts would have been read.

| interactivity floor | baseline | REAP | gain |
|---|---|---|---|
| ≥ 50 tok/s/user (BS ~100) | 3517 | 4756 | **+35%** |
| ≥ 40 tok/s/user | 3779 | 6326 | **+67%** |
| ≥ 35 tok/s/user (BS ~200) | 3779 | 6861 | **+82%** |

### 3f. Pruning degradation

Code via **EvalPlus** (chat template, tree-sitter sanitization, expanded test
suites — lm-eval's `humaneval`/`mbpp_plus` do none of these and understate an
instruct model badly). MC via lm-eval, 0-shot, seed 42, `acc_norm`.

```bash
bash inco/scripts/experiments/3f-evals.sh         # detached; one dataset per container
bash inco/scripts/experiments/3f-evals-report.sh  # once they finish
```

| domain | benchmark | unpruned | REAP 50% | delta | verdict |
|---|---|---|---|---|---|
| reasoning | arc_challenge | 0.6263 | 0.3976 | −22.9 pt | significant |
| knowledge | openbookqa | 0.4480 | 0.3140 | −13.4 pt | significant |
| reasoning | winogrande | 0.7388 | 0.6298 | −10.9 pt | significant |
| inference | rte | 0.7726 | 0.6895 | −8.3 pt | significant |
| code | MBPP | 0.8889 | 0.8571 | −3.2 pt | borderline (p=.058) |
| code | MBPP+ | 0.7328 | 0.7169 | −1.6 pt | noise |
| code | HumanEval+ | 0.8720 | 0.8659 | −0.6 pt | noise |
| code | HumanEval | 0.9146 | 0.9146 | ±0.0 pt | noise |

**Degradation is extremely low on code and much higher everywhere else.** That
is the calibration dataset: calibrating on codealpaca preserved the "coding
experts" and pruned the non-coding ones. Since the purpose of this model is
short-form code generation, this is no problem for us — but it means **the
calibration corpus, not the compression ratio, is the design decision.**

MBPP at p=.058 is the row to watch; MBPP+ on the same generations is flat,
which is the more informative of the pair.

## 4. DFlash2 speculative decoding

| | |
|---|---|
| target | REAP-50% pruned Qwen3-30B-A3B-Instruct |
| data | codealpaca completions **distilled from the pruned target**, on-policy |
| draft | 5 layers, 3 hidden states extracted from the target |
| samples / epochs / seq len / block | 5000 / 5 / 2048 / 8 |
| speculative tokens | 7 |
| trained with | `vllm-project/speculators`; served with vLLM |

**Distilling from the pruned target, not the unpruned one, is essential** —
pruning modifies the output distribution, so a drafter trained on the dense
model's completions predicts the wrong model.

```bash
bash inco/scripts/experiments/4-train-drafter.sh
```

~65 min: ~19 min regenerating responses, a few minutes of data prep and
hidden-state extraction, ~42 min of training.

**`--max-model-len` must exceed `--seq-length`.** `prepare-data` renders with
`truncate_prompt_tokens=seq_length` and vLLM budgets
`max_tokens = max_model_len - input_length`, so equal values leave zero
headroom and every conversation that hits the truncation limit is dropped with
`max_tokens must be at least 1, got 0`. The step still exits 0, so at scale the
loss is silent. `max_model_len` now defaults to `2 * seq_length`.

### 4a. Serving — acceptance and throughput

vLLM wants the *drafter* checkpoint as the served model: a speculators config
makes it swap in `verifier.name_or_path` and derive `method=dflash`.

```bash
bash inco/scripts/experiments/4a-acceptance.sh
bash inco/scripts/experiments/4-specdec-report.sh  # plots this against §4b
```

Eval protocol: temperature 0; warmup 16 per concurrency point; 128 prompts per
point (64 HumanEval+, 64 MBPP+); at least 8 waves per point, so the pool
repeats until that holds (c=1 → 128 requests, one pass through the pool; c=16 →
the pool repeats). **Not the 1024/512 shape** — prefill is ~15% of tokens here,
so these tok/s do not belong on the same axes as the sweeps above.

Three settings this measurement is wrong without, each found by getting a wrong
answer first:

* **`--no-async-scheduling` in both phases.** It defaults on but is blocked for
  `method="dflash"`, so omitting it gives the no-draft phase a scheduler the
  drafted phase is not allowed to have. Worth 1.51x to the baseline at c=1.
* **A graph ceiling sized for the drafted batch.** The verify pass carries
  `concurrency x (1 + spec_tokens)` rows; the 512 default is exactly 64 x 8, so
  every point above c=64 ran the draft eager and manufactured a fake peak.
* **`--no-enable-prefix-caching`.** The prompt pool repeats within a point, so
  with caching those repeats read their prefills out of the cache.

| c | REAP draft | REAP plain | ratio | acc_len | unpruned draft | unpruned plain | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 400 | 168 | 2.38x | 3.894 | 382 | 165 | 2.32x |
| 8 | 1940 | 726 | 2.67x | 3.956 | 1530 | 660 | 2.32x |
| 32 | 5920 | 2062 | **2.87x** | 3.924 | 4261 | 1625 | 2.62x |
| 48 | 7861 | 2846 | 2.76x | 3.918 | 6001 | 2198 | 2.73x |
| 64 | 9457 | 3623 | 2.61x | 3.921 | 7218 | 2770 | 2.61x |
| 96 | 11,580 | 5004 | 2.31x | 3.922 | 9314 | 3776 | 2.47x |

* **The drafter is worth 2.2–2.9x on both models**, peaking at c=32–48. Below
  that the draft's fixed per-step cost is unamortised; above it, rejected
  tokens start costing real compute.
* **Pruning is worth ~1.25x on top, and the two stack**: pruned+drafted 11,580
  against unpruned undrafted 3776 is **3.1x**.
* **Acceptance is flat in load** — 3.89–3.96 pruned, 3.99–4.02 unpruned, across
  a 96x range of concurrency.
* Speculative decoding is **lossless**: the target verifies every drafted
  token, so §3f is unchanged by this stage.

One measurement per point, so 0.1–0.2x differences between the two models'
speedup curves are inside run-to-run variation. The acceptance gap, 2% in the
same direction at all ten points, is the more trustworthy of the two.

### 4b. How does REAP pruning affect spec-decode performance?

A second drafter trained against the **unpruned** target with the same setup —
including its own on-policy data generation, so each draft sees the
distribution of the model it will predict.

```bash
bash inco/scripts/experiments/4b-train-dense-drafter.sh
bash inco/scripts/experiments/4b-acceptance-dense.sh  # after training completes
```

Validation expected acceptance length, per epoch
(`checkpoints/{epoch}/val_metrics.json`):

| epoch | dense eal | pruned eal | dense/pruned |
|---|---|---|---|
| 0 | 2.469 | 2.542 | 97.1% |
| 1 | 2.868 | 2.953 | 97.1% |
| 2 | 3.138 | 3.245 | 96.7% |
| 3 | 3.376 | 3.476 | 97.1% |
| 4 | 3.524 | **3.637** | 96.9% |

**The drafter for the pruned target slightly outperforms the one for the
unpruned target**, by ~3% at every epoch. Could be that with fewer experts
there is less variety in the generated tokens, so there is less for the draft
to model.

Two caveats: it does not survive into serving, where the pruned drafter lands
2% *behind* (3.92 vs 4.01), so the claim that holds is the weak one —
**pruning does not measurably change draftability**. And neither model had
converged: every metric was still improving monotonically at epoch 4, so more
epochs or more samples are the obvious next lever.

---

# Harness

## Output

Per run, in `inco/results/<label>/`:

| file | contents |
|---|---|
| `pareto.csv` | one row per concurrency: both axes, TTFT/ITL/e2e, error rate |
| `pareto.png` | the Pareto curve, annotated with concurrency |
| `summary.md` | markdown table + throughput at each interactivity SLO |
| `manifest.json` | workload, sweep config, every aiperf command, effective `VllmConfig`, `gpu_model` |
| `concurrency*/` | raw aiperf artifacts, per-request records included |

A Pareto curve is not one number, so the comparison that matters is
**throughput at an interactivity SLO** — "at ≥50 tok/s/user, how many tok/s
does one GPU deliver?" It prevents claiming a win that was really just trading
latency for batch size. An optimization is real when it moves the frontier up
and/or right, not when it moves one point along the existing curve.

Also checked per point: `output_sequence_length` must equal 512 (else
`ignore_eos` was not honoured and the run is invalid), `error_rate`, and
`input_sequence_length` to confirm the tokenizer agreed with the server. A
point whose aiperf run fails, or exits clean but writes no export, is recorded
as a failure rather than dropped; the sweep exits 1.

## Layout

```
inco/
├── bench/
│   ├── config.py        workload + sweep config, the single source of truth
│   ├── server.py        readiness, /server_info capture, perf-feature audit
│   ├── aiperf.py        aiperf command construction
│   ├── collect.py       aiperf JSON export -> tidy sweep points
│   ├── report.py        CSV / markdown / Pareto plot / throughput-at-SLO
│   ├── sweep.py         the driver:  python -m bench.sweep
│   ├── compare.py       before/after overlay:  python -m bench.compare a b
│   ├── eval_compare.py  per-task eval deltas with a significance gate
│   └── plot_specdec.py  drafted-vs-not frontier and acceptance panels
├── modal/
│   ├── modal_baseline.py     serve + sweep on one Modal H100
│   ├── modal_reap.py         prune, EvalPlus, lm-eval, expert-activation probe
│   └── modal_speculators.py  DFlash2 training pipeline + acceptance A/B
├── scripts/
│   ├── experiments/     one script per experiment in this README
│   └── ...              workload.env, serve/run/install, the three plot scripts
├── tests/               pytest suite, no GPU required
└── results/             generated artifacts (gitignored)
```

Every knob is also an env var (`INCO_MODEL`, `INCO_OSL`, `INCO_CONCURRENCIES`,
…); `scripts/workload.env` keeps the server flags and the aiperf flags in sync.

## Tests

No GPU required — the measurement loop runs against a fake aiperf and a fake
server, so parsing, the audit gate, failure handling, the Pareto math, eval
comparison and the plotting code are all covered.

```bash
bash inco/scripts/experiments/test.sh
# 97% statement coverage on bench/
```

## Tools used

Written with Claude Code (Opus 5). REAP from
[CerebrasResearch/reap](https://github.com/CerebrasResearch/reap); DFlash2 from
[vllm-project/speculators](https://github.com/vllm-project/speculators); aiperf
flag names and export schema from the NVIDIA AIPerf docs; vLLM flags verified
against this fork's source rather than assumed.
