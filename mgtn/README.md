# `mgtn/` — MGTN model and training/eval pipeline

`mgtn` is the model proposed in the paper: it keeps the MeshGraphNets
encoder/decoder and edge construction, but replaces the message-passing
processor with a graph-aware Transformer processor (multi-head self-attention
with an adjacency/attention bias and a geometric positional encoding derived
from mesh node coordinates).

The pipeline (data reading, encode–process–decode framework, online
normalization, CFD/cloth wrappers, rollout evaluation, plotting, dataset
download) is adapted from
[MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
(Apache 2.0); see the top-level `NOTICE`.

## Module map

| File | Role | Origin |
|---|---|---|
| `run_model.py` | Entry point: `--mode=train` / `--mode=eval`, checkpointing, rollout dump | MGN, adapted |
| `core_model.py` | Encode–process–decode graph net; processor stack now uses `TransformerBlock` | MGN, modified |
| `TransformerBlock.py` | One pre-norm Transformer encoder layer over node tokens (self-attention + FFN + residual + LayerNorm), with optional additive attention bias | **new (MGTN)** |
| `MultiHeadSelfAttention.py` | Scaled dot-product multi-head self-attention with optional `[N, N]` bias | **new (MGTN)** |
| `gpe.py` | `GeometricPositionalEncoding`: MLP projection of mesh node coordinates into model-dim positional features | **new (MGTN)** |
| `cfd_model.py`, `cfd_eval.py`, `plot_cfd.py` | `cylinder_flow` / `airfoil` model wrapper, rollout error, plotting | MGN, adapted |
| `cloth_model.py`, `cloth_eval.py`, `plot_cloth.py` | `flag_simple` / sphere model wrapper, rollout error, plotting | MGN, adapted |
| `dataset.py` | TFRecord parsing, windowing, `add_targets`, `split_and_preprocess` | MGN |
| `normalization.py` | `Normalizer` — online accumulation of feature mean/std | MGN |
| `common.py` | `NodeType` enum, triangle→edge helpers | MGN |
| `download_dataset.sh` | Fetches `{meta.json,train,valid,test}.tfrecord` from the public MeshGraphNets bucket | MGN |
| `requirements.txt` | Minimal deps (TF 1.15, Sonnet 1); see top-level `requirements-mgn.txt` for the pinned env used in the paper | — |

## Quick start

```bash
# from the repository root
python -m mgtn.run_model --mode=train --model=cloth \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple \
    --num_training_steps=200000

python -m mgtn.run_model --mode=eval --model=cloth \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple \
    --rollout_path=${DATA}/rollout_flag.pkl --num_rollouts=100

python -m mgtn.plot_cloth --rollout_path=${DATA}/rollout_flag.pkl
```

Use `--model=cfd` with `cylinder_flow` / `airfoil` and `mgtn.plot_cfd` for the
CFD domain. `run_cfd_m.sh` / `run_cloth_m.sh` run short smoke tests; the
top-level `job_s_mgn_vs_mgt_*.sh` are the full SLURM jobs used for the paper.
