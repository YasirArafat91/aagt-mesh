#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e

# Display commands being run.
set -x

#TMP_DIR=`mktemp -d`
TMP_DIR='DATA/cylinder_flow'

#python3.11 -m venv "${TMP_DIR}/env"
#source "${TMP_DIR}/env/bin/activate"

#conda create -n meshgraphnets python=3.7
#conda activate meshgraphnets
# Install dependencies.
# pip install --upgrade -r meshgraphnets/requirements.txt

# Download minimal dataset
DATA_DIR="${TMP_DIR}"
#bash meshgraphnets/download_dataset.sh flag_minimal ${TMP_DIR}

# Train for a few steps.
CHK_DIR="${TMP_DIR}/checkpoint"
python -m meshgraphnets.run_model --model=cfd --mode=train --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --num_training_steps=10

# Generate a rollout trajectory
ROLLOUT_PATH="${TMP_DIR}/rollout.pkl"
python -m meshgraphnets.run_model --model=cfd --mode=eval --checkpoint_dir=${CHK_DIR} --dataset_dir=${DATA_DIR} --rollout_path=${ROLLOUT_PATH} --num_rollouts=1

# Plot the rollout trajectory
python -m meshgraphnets.plot_cfd --rollout_path=${ROLLOUT_PATH}

# Clean up.
#rm -r ${TMP_DIR}
echo "Test run complete."
