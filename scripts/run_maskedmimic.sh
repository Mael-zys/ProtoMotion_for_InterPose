#!/bin/bash  -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate protomotions_interpose

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

model_path=$1
control_setting=$2
input_folder=$3
input_smpl_path=$4
input_control_path=$5
output_path=$6

python data/scripts/convert_amass_to_isaac.py ${input_folder} --humanoid-type=smpl


python protomotions/eval_agent.py \
+robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/${control_setting} \
+motion_file=${input_smpl_path} \
+checkpoint=${model_path} \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat +save_motion=True \
hydra.run.dir=${output_path} \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","masked_gt_err","masked_gr_err","masked_gr_err_degrees"] \
+env.config.masked_mimic.user_control_folder=${input_control_path} \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=0.5