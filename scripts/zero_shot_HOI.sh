export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0

model_path=$1
training_data=$2
control_setting=$3
success_threshold=$4

### omomo_interpolation
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/${control_setting} \
+motion_file=data/Dataset/omomo_data/omomo_test_smpl.pt \
+checkpoint=${model_path} \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat +save_motion=True \
hydra.run.dir=outputs/${training_data}_${control_setting}_omomo_${success_threshold}_interpolation \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","masked_gt_err","masked_gr_err","masked_gr_err_degrees"] \
+env.config.masked_mimic.user_control_folder=data/Dataset/omomo_data/test_set_hands_control_interpolation_position \
+env.config.masked_mimic.user_control_yaml=data/Dataset/omomo_data/omomo_test_smpl.yaml \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=${success_threshold}

### behave_test_interpolation
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/${control_setting} \
+motion_file=data/Dataset/behave_data/behave_test_smpl.pt \
+checkpoint=${model_path} \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat +save_motion=True \
hydra.run.dir=outputs/${training_data}_${control_setting}_behave_${success_threshold}_interpolation \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","masked_gt_err","masked_gr_err","masked_gr_err_degrees"] \
+env.config.masked_mimic.user_control_folder=data/Dataset/behave_data/test_set_hands_control_interpolation_position \
+env.config.masked_mimic.user_control_yaml=data/Dataset/behave_data/behave_test_smpl.yaml \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=${success_threshold}
