#!/bin/bash
#SBATCH --job-name=cloth_FlagSimple_200
#SBATCH --output=cloth_FlagSimple_200_%j.out
#SBATCH --error=cloth_FlagSimple_200_%j.err
#SBATCH --time=80:00:00
#SBATCH --partition=quad_rtx_8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ---------------------------
# Safety & debug
# ---------------------------
set -e
set -x

echo "Job started on $(hostname)"
echo "Start time: $(date)"

# ---------------------------
# Load conda safely (NO .bashrc)
#   Set CONDA_ROOT and PROJECT_DIR for your own cluster/machine.
# ---------------------------
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/aagt-mesh}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda activate mgn || {
  echo "Conda activation failed"
  exit 1
}

# Sanity checks (important for CF debugging)
which python
python --version
nvidia-smi

# ---------------------------
# Go to project directory
#   The meshgraphnets baseline package must be importable from here.
# ---------------------------
cd "${PROJECT_DIR}" || exit 1

# ---------------------------
# Paths
# ---------------------------
DATA_DIR="Data/flag_simple"
ROLLOUT_PATH="${DATA_DIR}/mgn/rollout.pkl"
CHK_DIR="${DATA_DIR}/mgn/checkpoint"
SAVE_PATH="test/Result"

mkdir -p "${CHK_DIR}"


# ---------------------------
# Train MeshGraphNets (cloth)
# ---------------------------
echo "MGN Training started"

python -m meshgraphnets.run_model \
  --model=cloth \
  --mode=train \
  --checkpoint_dir="${CHK_DIR}" \
  --dataset_dir="${DATA_DIR}" \
  --num_training_steps=200000

# ---------------------------
# Rollout generation
# ---------------------------
echo "MGN Rollout started"

python -m meshgraphnets.run_model \
  --model=cloth \
  --mode=eval \
  --checkpoint_dir="${CHK_DIR}" \
  --dataset_dir="${DATA_DIR}" \
  --rollout_path="${ROLLOUT_PATH}" \
  --num_rollouts=100



# for aagt-mesh (ours)
ROLLOUT_PATH_aagt="${DATA_DIR}/aagt_mesh/rollout.pkl"
CHK_DIR_aagt="${DATA_DIR}/aagt_mesh/checkpoint"

mkdir -p "${CHK_DIR_aagt}"

echo "aagt-mesh training started"
python -m aagt_mesh.run_model \
  --model=cloth \
  --mode=train \
  --checkpoint_dir="${CHK_DIR_aagt}" \
  --dataset_dir="${DATA_DIR}" \
  --num_training_steps=200000

echo "aagt-mesh rollout started"

python -m aagt_mesh.run_model \
  --model=cloth \
  --mode=eval \
  --checkpoint_dir="${CHK_DIR_aagt}" \
  --dataset_dir="${DATA_DIR}" \
  --rollout_path="${ROLLOUT_PATH_aagt}" \
  --num_rollouts=100

# python -m meshgraphnets.plot_cloth \
#   --rollout_path="${ROLLOUT_PATH}"

# python -m test.plot_gt_vs_pred\
#   --rollout_path="${ROLLOUT_PATH}" \
#   --save_path="${SAVE_PATH}"
# # ---------------------------
# # Cleanup
# # ---------------------------

# set -e

# # Display commands being run.
# set -x

# #TMP_DIR=`mktemp -d`
# TMP_DIR='Data/flag_simple'

#python3.11 -m venv "${TMP_DIR}/env"
#source "${TMP_DIR}/env/bin/activate"

#conda create -n meshgraphnets python=3.7
#conda activate meshgraphnets
# Install dependencies.
# pip install --upgrade -r meshgraphnets/requirements.txt

# # Download minimal dataset
# DATA_DIR="${TMP_DIR}"
# #bash meshgraphnets/download_dataset.sh flag_minimal ${TMP_DIR}
# SAVE_PATH="test/Result"

# mkdir -p "${SAVE_PATH}"
# # Train for a few steps.
# # echo "Training started"
# CHK_DIR="${TMP_DIR}/checkpoint"
# # python -m aagt_mesh.run_model --model=cloth --mode=train --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --num_training_steps=50000

# echo "Generate a rollout trajectory"
# # Generate a rollout trajectory
# ROLLOUT_PATH="${TMP_DIR}/rollout.pkl"
# python -m aagt_mesh.run_model --model=cloth --mode=eval --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --rollout_path=${ROLLOUT_PATH} --num_rollouts=50

# echo "Plot the rollout trajectory"
# # Plot the rollout trajectory
# python -m aagt_mesh.plot_cloth --rollout_path=${ROLLOUT_PATH}

# echo "plot ground truth vs prediction"

# python -m test.plot_gt_vs_pred\
#   --rollout_path="${ROLLOUT_PATH}" \
#   --save_path="${SAVE_PATH}"
# Clean up.
#rm -r ${TMP_DIR}
echo "Test run complete."

conda deactivate

echo "Job finished at: $(date)"
