# DQC Repo Guide (`README_H.md`)

This is a developer-oriented map of the `dqc/` directory and a step-by-step guide to
**finetuning a DQC verifier** (the chunk-critic Q-function that RoboMonkey uses to
score/rank candidate action chunks). It is meant to complement the upstream
[README.md](README.md), which covers the original OGBench reproduction of the
*Decoupled Q-Chunking* paper.

---

## 1. What this repo is

There are really **two codebases living here**:

1. **Upstream DQC** — the published *Decoupled Q-Chunking* (DQC) offline-RL code, built on
   [horizon-reduction](https://github.com/seohongpark/horizon-reduction). This is what
   [main.py](main.py) + [agents/dqc.py](agents/dqc.py) run, on OGBench tasks like
   `cube-quadruple`. See [README.md](README.md) for the paper-reproduction commands.

2. **The RoboMonkey verifier fork** — a custom layer added on top to train an
   **image + language conditioned chunk-critic** on real-robot data (BridgeV2 + SOAR).
   This is the "DQC verifier": a learned Q-function `Q(image, language, action_chunk)`
   that scores how good a chunk of actions is for a given visual observation and task
   instruction. RoboMonkey samples many candidate chunks from a base policy and uses
   this verifier to pick the best (best-of-N). **This is the part you almost certainly
   care about and the focus of §4 below.**

Everything with `bridge_soar` in its name belongs to layer (2).

---

## 2. Directory layout

```
dqc/
├── main.py                         # Upstream DQC trainer (OGBench). Layer (1).
├── README.md                       # Upstream paper README + an eval snippet at the bottom.
├── requirements.txt                # jax/flax/distrax + torch/pyarrow/imageio + TF-hub (MUSE).
│
├── agents/
│   ├── __init__.py                 # Registry: {dqc, bridge_calql_chunk}.
│   ├── dqc.py                      # Upstream DQC agent (OGBench). Layer (1).
│   ├── bridge_calql_chunk.py       # CAL-QL chunk-critic verifier + ResNet34-FiLM encoder.
│   └── bridge_dqc_chunk.py         # DQC-style chunk-critic verifier (no CQL term).  ← main verifier
│
├── experiments/
│   ├── train_bridge_soar_dqc.py    # Train the DQC verifier (bridge_dqc_chunk). ← MAIN entrypoint
│   ├── train_bridge_soar_chunk.py  # Train the CAL-QL verifier (bridge_calql_chunk).
│   ├── visualize_bridge_soar_dqc_q.py    # Eval/viz: Q over episodes + noise scatter (DQC).
│   ├── visualize_bridge_soar_chunk_q.py  # Same, for the CAL-QL verifier.
│   ├── analyze_bridge_soar_model_q_correlations.py  # Correlate learned Q vs MC returns.
│   ├── compare_caption_infinite_horizon_q.py        # Compare Q under different captions.
│   ├── reproduce.py / reproduce-sensitivity.py      # Print upstream OGBench commands.
│   ├── generate.py                                  # Upstream data/command generation helper.
│   ├── configs/
│   │   └── bridge_soar_chunk_data_config.py  # Dataset config (sources, horizon, batch). ← edit me
│   └── scripts/
│       ├── launch_bridge_soar_dqc.sh         # Full DQC verifier run (1M steps). ← main launcher
│       ├── launch_bridge_soar_dqc_debug.sh   # 1k-step smoke test (tiny batch/workers).
│       ├── launch_bridge_soar_chunk_calql.sh # Full CAL-QL verifier run.
│       └── launch_bridge_soar_chunk_calql_debug.sh
│
├── utils/
│   ├── bridge_lerobot_dataset.py   # Streaming BridgeV2+SOAR LeRobot loader. ← the data pipeline
│   ├── text_processing.py          # MUSE (universal-sentence-encoder) language embedder.
│   ├── flax_utils.py               # save_agent / restore_agent (pickled flax state dicts).
│   ├── networks.py                 # MLP, ensemblize.
│   ├── encoders.py, datasets.py, log_utils.py, evaluation.py  # Mostly upstream helpers.
│   └── bridge_lerobot_dataset.py
│
├── envs/                           # OGBench env helpers (upstream, layer 1 only).
├── exp_data/                       # Released paper results (dqc-exp-data.pkl) + sanity notebook.
├── assets/                         # Figures for the upstream README.
│
├── job.sh                          # SLURM: submit CAL-QL verifier training (4 GPU).
├── job_dqc.sh                      # SLURM: submit DQC verifier training (4 GPU).
└── job_resume.sh                   # SLURM: resume CAL-QL training from a checkpoint.
```

---

## 3. Key utilities, explained

### Data: [utils/bridge_lerobot_dataset.py](utils/bridge_lerobot_dataset.py)
Streams transitions from one or more **LeRobot-format** datasets (parquet actions/metadata +
mp4 videos) and turns each episode into chunk-critic training transitions.

- `make_bridge_soar_chunk_dataset(**data_config)` → `BridgeSoarChunkDataset`. Call
  `.iterator()` to get an infinite stream of dict batches via a PyTorch `DataLoader`.
- `BRIDGE_ACTION_LOW` / `BRIDGE_ACTION_HIGH` — fixed 7-D action normalization bounds
  (xyz + rpy + gripper). Actions are normalized to `[-1, 1]` and clipped by `action_clip_eps`.
  These bounds are written to `action_normalization.json` in every checkpoint and
  **re-validated at eval time** — do not change them without retraining.
- Per episode it builds, for each timestep `t`:
  - `observations.image` (current frame), `high_value_next_observations.image`
    (frame `t + backup_horizon`),
  - `high_value_action_chunks` — the `backup_horizon`-long action chunk flattened to
    `backup_horizon * action_dim`,
  - `high_value_rewards` — n-step discounted reward, `high_value_masks`,
    `high_value_backup_horizon`, `valids`, `mc_returns`.
- **Rewards.** BridgeV2 (success) episodes get a shaped ramp reward
  (`_bridge_rewards`: linear 0→1 over the first 95%, then 1). SOAR episodes are relabeled
  via a JSON `label_path` (`failure` → all-zero reward; `successful`/`already_complete` →
  ramp). Episodes with blank language instructions are dropped.
- `infinite_horizon_backup=True` treats every transition as non-terminal (mask=1), which
  the production launchers use.

### Language: [utils/text_processing.py](utils/text_processing.py)
`make_text_processor("muse_embedding")` loads Google's Multilingual Universal Sentence
Encoder from TF-Hub and produces a **512-D embedding** per instruction. The image encoder
is FiLM-conditioned on this embedding. (Runs on CPU; TF GPUs are disabled in the trainers.)

### Verifier networks: [agents/bridge_dqc_chunk.py](agents/bridge_dqc_chunk.py) and [agents/bridge_calql_chunk.py](agents/bridge_calql_chunk.py)
Shared encoder (`BridgeFilmResNet34` + `LanguageImageEncoder`, defined in the CAL-QL file
and imported by the DQC file): a ResNet-34 with GroupNorm + swish, **FiLM-conditioned on
the language embedding** after every block, with optional spatial-coordinate channels and
avg pooling.

- **`BridgeDQCChunkAgent`** (DQC verifier) trains three heads:
  - `chunk_critic` — Q over the *full* `backup_horizon` chunk (the verifier's scoring head),
  - `action_critic` + `value` — distill the chunk-Q down to a single-step critic / value
    (expectile/quantile backups; `distill_expectile`, `backup_quantile`, `backup_loss`),
  - `actor_bc` — a flow-matching BC policy (not needed for scoring, but trained jointly).
  No CQL term. Ensemble of `num_qs` critics aggregated by `q_agg` (`min`/`mean`).
- **`BridgeCalQLChunkAgent`** (CAL-QL verifier) trains a `chunk_critic` + Gaussian `actor`
  with a Cal-QL conservative penalty (`cql_alpha`, `cql_temp`, `cql_n_actions`,
  `cql_max_target_backup`, MC-return lower bound).
- **Both expose `score_chunks(observations, language_embedding, action_chunks, q_agg)`** —
  the JIT-compiled inference entry point RoboMonkey calls to rank candidate chunks.

### Checkpoints: [utils/flax_utils.py](utils/flax_utils.py)
`save_agent(agent, save_dir, epoch)` pickles the flax state dict to `params_{epoch}.pkl`.
`restore_agent(agent, restore_path, restore_epoch)` globs `restore_path` (must match
exactly one dir) and loads `params_{restore_epoch}.pkl`. Each run dir also contains
`flags.json` (full config) and `action_normalization.json`.

---

## 4. Finetuning a DQC verifier (the main workflow)

> "Finetune" here means **(re)train the chunk-critic** — either from scratch on your data
> or continuing from an existing checkpoint with `--restore_path`. There is one entrypoint
> per verifier flavor: [experiments/train_bridge_soar_dqc.py](experiments/train_bridge_soar_dqc.py)
> (DQC) and [experiments/train_bridge_soar_chunk.py](experiments/train_bridge_soar_chunk.py)
> (CAL-QL). The DQC one is the default verifier.

### Step 0 — Environment
```bash
cd dqc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # jax/flax + torch + pyarrow + imageio + tensorflow_hub/_text
```
The launchers assume a `.venv/` in the repo root and JAX with GPU support. The first run
downloads the MUSE encoder from TF-Hub (set `XDG_CACHE_HOME`/`TFHUB_CACHE_DIR` if offline).

### Step 1 — Point the data config at your datasets
Edit [experiments/configs/bridge_soar_chunk_data_config.py](experiments/configs/bridge_soar_chunk_data_config.py).
Each `sources` entry needs:
- `name`, `data_dir` — path to a LeRobot dataset (expects `meta/episodes.jsonl`,
  `meta/tasks.jsonl`, `data/chunk-XXX/episode_*.parquet`, `videos/.../episode_*.mp4`),
- `weight` — sampling weight across sources,
- `video_key` — which camera stream to read,
- `label_path` (optional) — JSON of success/failure relabels (used by SOAR).

Key knobs: `backup_horizon` (chunk length, default 8), `action_chunk_size` (**must be 1**),
`discount`, `infinite_horizon_backup`, `batch_size`, `num_workers`, `normalize_actions`.

> ⚠️ `backup_horizon`, `action_chunk_size`, and `batch_size` **must match** between the
> `--data.*` and `--agent.*` configs, or the trainer raises in `_prepare_configs`. The
> code currently only supports `policy_chunk_size=1` and (for CAL-QL) `use_chunk_critic=True`.

### Step 2 — Smoke test (1000 steps, tiny batch)
```bash
bash experiments/scripts/launch_bridge_soar_dqc_debug.sh
```
This runs `train_bridge_soar_dqc.py` for 1k steps with `batch_size=8`, `num_workers=2`.
Use it to confirm your data paths, videos, and labels load and that `_check_batch` passes.

### Step 3 — Full training run
Locally / on a node with GPUs:
```bash
bash experiments/scripts/launch_bridge_soar_dqc.sh
```
On SLURM (4×GPU, 32 CPU, 960G — see header):
```bash
sbatch job_dqc.sh        # DQC verifier   (calls launch_bridge_soar_dqc.sh)
# or
sbatch job.sh            # CAL-QL verifier (calls launch_bridge_soar_chunk_calql.sh)
```
The DQC launcher trains for `1,000,000` steps, logging every 1k and checkpointing every
100k to:
```
<save_dir>/dqc/<run_group>/<run_name>/params_<step>.pkl
                                      ├── flags.json
                                      ├── action_normalization.json
                                      └── train.csv
```
Important flags (all overridable on the command line, `--agent.*` / `--data.*`):

| Flag | Meaning | Default in launcher |
|------|---------|---------------------|
| `--agent.backup_horizon` | chunk length the critic scores | 8 |
| `--agent.policy_chunk_size` | must be 1 | 1 |
| `--agent.num_critics` | Q-ensemble size | 2 |
| `--agent.discount` | γ | 0.98 |
| `--agent.lr` | Adam LR | 3e-4 |
| `--agent.target_update_rate` | Polyak τ | 5e-3 |
| `--agent.backup_quantile` / `--agent.distill_expectile` | value/distill backup shape | 0.9 / 0.8 |
| `--agent.best_of_n` | default N for downstream best-of-N | 32 |
| `--data.infinite_horizon_backup` | treat all transitions as non-terminal | True (prod) |
| `--agent.batch_size` / `--data.batch_size` | must match | 256 |

CAL-QL adds `--agent.cql_alpha`, `--agent.cql_temp` (the conservatism strength).

### Step 4 — Resume / continue-finetune from a checkpoint
Pass `--restore_path` (a glob matching exactly one run dir) and `--restore_epoch`. See
[job_resume.sh](job_resume.sh) for a worked example:
```bash
.venv/bin/python experiments/train_bridge_soar_chunk.py \
  --restore_path=".../sd000s_124194.0.20260526_221347" \
  --restore_epoch=100000 \
  --offline_steps=899999 \
  ...   # same --agent.* / --data.* as the original run
```
To continue-finetune the **DQC** verifier, add the same `--restore_path`/`--restore_epoch`
flags to `launch_bridge_soar_dqc.sh` (the DQC trainer already supports them).
**Re-use the original config** — restore relies on identical network shapes.

### Step 5 — Evaluate / visualize the trained verifier
```bash
python experiments/visualize_bridge_soar_dqc_q.py \
    --checkpoint_dir="<save_dir>/dqc/<run_group>/<run_name>/" \
    --restore_epoch=-1 \          # -1 = latest params_*.pkl
    --output_dir=exp/eval_videos/dqc \
    --k_episodes=10 \             # per-episode Q-vs-reward videos
    --k_states=64 \               # noisy-chunk scatter (Q drops as chunks deviate from expert)
    --q_agg=min --fps=10 --score_batch_size=64
```
This produces, per episode, an mp4 overlaying the model's chunk-Q against the discounted
return, plus a `bridge_noise_q_scatter.{csv,png}` showing that Q decreases as candidate
chunks move away from the expert chunk — the core sanity check that the verifier learned a
useful ranking. `analyze_bridge_soar_model_q_correlations.py` and
`compare_caption_infinite_horizon_q.py` give further diagnostics (Q vs. MC-return
correlation; sensitivity to the language caption). The eval scripts validate that the
checkpoint's `action_normalization.json` matches the loader constants before scoring.

### Step 6 — Serve the verifier to RoboMonkey
The trained checkpoint is loaded by the inference server outside this directory
(`monkey-verifier/src/infer_server.py` / `verifier_client.py` in the parent repo), which
calls `agent.score_chunks(...)` to rank policy-sampled action chunks at deploy time.

---

## 5. DQC vs. CAL-QL verifier — which to finetune?

| | DQC (`bridge_dqc_chunk`) | CAL-QL (`bridge_calql_chunk`) |
|---|---|---|
| Conservatism | implicit (expectile/quantile distill) | explicit CQL penalty (`cql_alpha`) |
| Extra heads | action-critic + value + flow BC actor | Gaussian actor |
| Entrypoint | `train_bridge_soar_dqc.py` | `train_bridge_soar_chunk.py` |
| Launcher | `launch_bridge_soar_dqc.sh` / `job_dqc.sh` | `launch_bridge_soar_chunk_calql.sh` / `job.sh` |
| Tuning sensitivity | quantile/expectile knobs | `cql_alpha` (over-conservative if too high) |

Both expose the same `score_chunks` API, so they are drop-in interchangeable as verifiers.
Default to the **DQC** verifier unless you specifically want CQL-style conservatism.

---

## 6. Gotchas

- **`policy_chunk_size` must be 1** and `data.action_chunk_size` must be 1 — the Bridge/SOAR
  path is hard-coded for single-step policy chunks with a longer-horizon critic chunk.
- **Config mismatches throw early.** `backup_horizon`, `action_chunk_size`, `batch_size`
  must agree across `--agent.*` and `--data.*`.
- **Action normalization is frozen** to `BRIDGE_ACTION_LOW/HIGH`. Changing it invalidates
  existing checkpoints (eval will refuse to load them).
- **MUSE download** happens on first run; cache it for offline/headless SLURM nodes.
- **Blank-language episodes are silently dropped**; the loader prints how many per source.
- The hard-coded `save_dir` / `data_dir` paths in the launchers point at the original
  `socialrl`/`weirdlab` cluster — override them for your environment.
