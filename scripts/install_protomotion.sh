#!/bin/bash

conda create -n protomotions_interpose python=3.8

source ~/miniconda3/etc/profile.d/conda.sh

conda activate protomotions_interpose

git lfs fetch --all

# Install IsaacGym

wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4

pip install -e isaacgym/python

# Install ProtMotions
pip install -e .
pip install -r requirements_isaacgym.txt
pip install -e isaac_utils
pip install -e poselib