#!/bin/bash
#SBATCH --job-name=cfd_airfoil
#SBATCH --output=cfd_airfoil_%j.out
#SBATCH --error=cfd_airfoil_%j.err
#SBATCH --time=70:00:00
#SBATCH --partition=quad_rtx_8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ---------------------------
# Load environment
# ---------------------------
source /home/cril/arafat/miniconda3/etc/profile.d/conda.sh
conda activate mgn

# Safety check
# which python
# python --version

# ---------------------------
# Go to project directory
# ---------------------------
cd /home/cril/arafat/MGN-C || exit 1

TMP_DIR='Data/airfoil'
DATA_DIR="${TMP_DIR}"
ROLLOUT_PATH="${TMP_DIR}/mgn/rollout.pkl"
CHK_DIR="${TMP_DIR}/mgn/checkpoint"

mkdir -p "${CHK_DIR}"
# ---------------------------
# Run training
# ---------------------------
echo "training started"
python -m meshgraphnets.run_model \
  --model=cfd \
  --mode=train \
  --checkpoint_dir=${CHK_DIR} \
  --dataset_dir=${DATA_DIR} \
  --num_training_steps=200000


# Generate a rollout trajectory

echo "Rollout started"
python -m meshgraphnets.run_model \
  --model=cfd --mode=eval \
  --checkpoint_dir=${CHK_DIR} \
  --dataset_dir=${DATA_DIR} \
  --rollout_path=${ROLLOUT_PATH} \
  --num_rollouts=100



# visualize

# echo "visulaize and save"
# python -m meshgraphnets.plot_cfd \
#   --rollout_path=${ROLLOUT_PATH} 

# python -m test.plot_cfd_gt_vs_pred.py \
#     --rollout_path='Data/airfoil/mgn/rollout.pkl' \
#     --save_path='test/Result'

ROLLOUT_PATH_mgtn="${TMP_DIR}/mgtn/rollout.pkl"
CHK_DIR_mgtn="${TMP_DIR}/mgtn/checkpoint"

mkdir -p "${CHK_DIR_mgtn}"

echo "training started"
python -m mgtn.run_model \
  --model=cfd \
  --mode=train \
  --checkpoint_dir=${CHK_DIR_mgtn} \
  --dataset_dir=${DATA_DIR} \
  --num_training_steps=200000


# Generate a rollout trajectory

echo "Rollout started"
python -m mgtn.run_model \
  --model=cfd --mode=eval \
  --checkpoint_dir=${CHK_DIR_mgtn} \
  --dataset_dir=${DATA_DIR} \
  --rollout_path=${ROLLOUT_PATH_mgtn} \
  --num_rollouts=100
#Deactivate conda environment
conda deactivate

echo "Job finished at: $(date)"