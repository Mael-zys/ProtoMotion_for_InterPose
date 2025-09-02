#!/bin/sh

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0

model_path=$1
training_data=$2
control_setting=$3
success_threshold=$4

## compute metrics

# ### amass_test
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/${control_setting} \
+motion_file=data/Dataset/amass_data/amass_test_smpl.pt \
+checkpoint=${model_path} +agent.config.augment_rotation=False \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat \
hydra.run.dir=outputs/${training_data}_${control_setting}_amass_${success_threshold}/$(date +"%Y-%m-%d-%H-%M-%S") \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","masked_gt_err","masked_gr_err","masked_gr_err_degrees"] \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=${success_threshold}

# ### omomo_test
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/${control_setting} \
+motion_file=data/Dataset/omomo_data/omomo_test_smpl.pt \
+checkpoint=${model_path} +agent.config.augment_rotation=False \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat \
hydra.run.dir=outputs/${training_data}_${control_setting}_omomo_${success_threshold}/$(date +"%Y-%m-%d-%H-%M-%S") \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","masked_gt_err","masked_gr_err","masked_gr_err_degrees"] \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=${success_threshold}


### behave_test
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/${control_setting} \
+motion_file=data/Dataset/behave_data/behave_test_smpl.pt \
+checkpoint=${model_path} +agent.config.augment_rotation=False \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat \
hydra.run.dir=outputs/${training_data}_${control_setting}_behave_${success_threshold}/$(date +"%Y-%m-%d-%H-%M-%S") \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","masked_gt_err","masked_gr_err","masked_gr_err_degrees"] \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=${success_threshold}
