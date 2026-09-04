# Adjacency-Aware Graph Transformers for Learning Mesh Dynamics

> 2026 IEEE 5th International Conference on Robotics, Automation, Artificial-Intelligence and Internet-of-Things (RAAICON)
> 18-19 September 2026, Jashore University of Science and Technology (JUST), Jashore, Bangladesh

## Overview

This repository accompanies the paper *"Adjacency-Aware Graph Transformers for
Learning Mesh Dynamics"*. It builds on
[MeshGraphNets](https://arxiv.org/abs/2010.03409) (Pfaff et al., ICLR 2021)
and replaces its message-passing processor with a graph-aware Transformer
processor (**MGTN**): edge features are still built from sender/receiver node
features as in MeshGraphNets, but the node update is performed by a
Transformer block with multi-head self-attention, an adjacency/attention bias,
and a geometric positional encoding derived from mesh node coordinates
(`mgtn/core_model.py`, `mgtn/TransformerBlock.py`,
`mgtn/MultiHeadSelfAttention.py`, `mgtn/gpe.py`).

The model is evaluated on the same simulation domains as MeshGraphNets:

- `cylinder_flow` — CFD (`mgtn/cfd_model.py`, `mgtn/cfd_eval.py`, `mgtn/plot_cfd.py`)
- `flag_simple` — cloth (`mgtn/cloth_model.py`, `mgtn/cloth_eval.py`, `mgtn/plot_cloth.py`)

and benchmarked against the original MeshGraphNets baseline (referred to as
`mgn` throughout this repo) to compare rollout accuracy and qualitative
trajectory behavior.

> **Baseline dependency:** the SLURM job scripts also invoke
> `python -m meshgraphnets.run_model` to train the original MeshGraphNets
> baseline (`mgn`) alongside MGTN for comparison. Only the MGTN implementation
> (`mgtn/`) is included in this repository — the `meshgraphnets` package must
> be installed separately (as a sibling importable module) to reproduce the
> baseline runs.

## Repository layout

```
.
├── mgtn/                          # MGTN model, training/eval pipeline (see mgtn/README.md)
├── plot_three_panel.py            # Ground truth / MGN / MGTN comparison figure (flag_simple)
├── plot_cylinder_three_panel.py   # Ground truth / MGN / MGTN comparison figure (cylinder_flow)
├── job_s_mgn_vs_mgt_cfd.sh        # SLURM job: train+rollout MGN vs MGTN on cylinder_flow
├── job_s_mgn_vs_mgt_cloth.sh      # SLURM job: train+rollout MGN vs MGTN on flag_simple
├── requirements-mgn.txt           # Pinned dependencies used for the experiments
├── figures/                       # Generated comparison figures
├── LICENSE                        # Apache License 2.0
└── NOTICE                         # Attribution for code adapted from MeshGraphNets
```

## Setup

Prepare the environment and install dependencies:

```bash
conda create -n mgn python=3.7
conda activate mgn
pip install -r requirements-mgn.txt
```

Download a dataset (same datasets as MeshGraphNets):

```bash
mkdir -p ${DATA}
bash mgtn/download_dataset.sh flag_simple ${DATA}
```

Available dataset names: `airfoil`, `cylinder_flow`, `deforming_plate`,
`flag_minimal`, `flag_simple`, `flag_dynamic`, `flag_dynamic_sizing`,
`sphere_simple`, `sphere_dynamic`, `sphere_dynamic_sizing`. `--model=cfd`
covers both `airfoil` and `cylinder_flow`; `--model=cloth` covers the
flag/sphere domains.

## Running the model

Train MGTN on the cloth domain:

```bash
python -m mgtn.run_model --mode=train --model=cloth \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple
```

Generate trajectory rollouts:

```bash
python -m mgtn.run_model --mode=eval --model=cloth \
    --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple \
    --rollout_path=${DATA}/rollout_flag.pkl
```

Plot a trajectory:

```bash
python -m mgtn.plot_cloth --rollout_path=${DATA}/rollout_flag.pkl
```

Use `--model=cfd` with a CFD dataset (`cylinder_flow` or `airfoil`) and
`mgtn.plot_cfd` for the CFD domain.

### Quick smoke tests

`mgtn/run_cfd_m.sh` and `mgtn/run_cloth_m.sh` run a short train → rollout →
plot cycle to sanity-check the pipeline before launching a full job:

- `mgtn/run_cfd_m.sh` — 10 training steps on `DATA/cylinder_flow`, 1 rollout,
  then `mgtn.plot_cfd`.
- `mgtn/run_cloth_m.sh` — 50,000 training steps on `Data/flag_simple`, 50
  rollouts, then `mgtn.plot_cloth`. It also calls `test.plot_gt_vs_pred`,
  which is a cluster-local helper script not included in this repository —
  ignore that step (or replace it with `plot_three_panel.py` below) if it's
  missing.

### Full training on a SLURM cluster

`job_s_mgn_vs_mgt_cfd.sh` and `job_s_mgn_vs_mgt_cloth.sh` are the SLURM
batch scripts used for the paper's experiments. Each one:

1. Activates the `mgn` conda environment and requests a single GPU
   (`quad_rtx_8000` partition, 16 CPUs, 64 GB RAM).
2. Trains the MeshGraphNets baseline for 200,000 steps
   (`python -m meshgraphnets.run_model --mode=train ...`), then rolls out
   100 trajectories, saving to `Data/<dataset>/mgn/checkpoint` and
   `Data/<dataset>/mgn/rollout.pkl`.
3. Trains MGTN the same way (`python -m mgtn.run_model --mode=train ...`)
   for 200,000 steps and 100 rollouts, saving to
   `Data/<dataset>/mgtn/checkpoint` and `Data/<dataset>/mgtn/rollout.pkl`.

`job_s_mgn_vs_mgt_cfd.sh` runs on the `cylinder_flow` dataset with
`--model=cfd`; `job_s_mgn_vs_mgt_cloth.sh` runs on `flag_simple` with
`--model=cloth`. Set `CONDA_ROOT` and `PROJECT_DIR` (both default to `$HOME/...`)
for your machine, adjust the partition/resources, then submit with
`sbatch job_s_mgn_vs_mgt_cfd.sh` / `sbatch job_s_mgn_vs_mgt_cloth.sh`; logs are
written to `<job-name>_<jobid>.out` / `.err` in the submission directory.

### MGN vs. MGTN comparison figures

Once rollouts for both models are available under `Data/<dataset>/mgn/` and
`Data/<dataset>/mgtn/`, generate the side-by-side ground truth / MGN / MGTN
panels used in the paper:

```bash
python plot_three_panel.py \
    --mgn  Data/flag_simple/mgn/rollout.pkl \
    --ours Data/flag_simple/mgtn/rollout.pkl \
    --traj 0 --step 50 --out figures/

python plot_cylinder_three_panel.py \
    --mgn  Data/cylinder_flow/mgn/rollout.pkl \
    --ours Data/cylinder_flow/mgtn/rollout.pkl \
    --traj 0 --step 50 --out figures/
```

## Citation

If you use this code, please cite the paper accepted at RAAICON 2026:

    @inproceedings{arafat2026aagt,
      title={Adjacency-Aware Graph Transformers for Learning Mesh Dynamics},
      author={Md Yasir Arafat and
              Wissem Inoubli and
              Said Jabbour},
      booktitle={2026 IEEE 5th International Conference on Robotics, Automation,
                 Artificial-Intelligence and Internet-of-Things (RAAICON)},
      year={2026},
      address={Jashore University of Science and Technology (JUST), Jashore, Bangladesh}
    }

This work builds on MeshGraphNets; please also cite:

    @inproceedings{pfaff2021learning,
      title={Learning Mesh-Based Simulation with Graph Networks},
      author={Tobias Pfaff and
              Meire Fortunato and
              Alvaro Sanchez-Gonzalez and
              Peter W. Battaglia},
      booktitle={International Conference on Learning Representations},
      year={2021}
    }

## License

Released under the [Apache License 2.0](LICENSE). Portions of the data
pipeline and model framework are adapted from DeepMind's
[MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
(also Apache 2.0); see [`NOTICE`](NOTICE) for details. The MeshGraphNets
datasets are distributed by DeepMind under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
