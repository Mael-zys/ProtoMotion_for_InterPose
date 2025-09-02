export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0,1,2,3

# first stage
python protomotions/train_agent.py \
+exp=full_body_tracker/transformer_flat_terrain +robot=smpl \
+simulator=isaacgym motion_file=data/Dataset/interpose_data/merged_interpose_amass_train.pt \
+experiment_name=full_body_tracker_merged +terrain=flat \
ngpu=4 eval_overrides.ngpu=4 num_envs=2048 agent.config.batch_size=8192 +opt=wandb \
agent.config.eval_metrics_every=1000 training_max_steps=100000000000

# second stage
python protomotions/train_agent.py +exp=masked_mimic/flat_terrain +robot=smpl \
+simulator=isaacgym motion_file=data/Dataset/interpose_data/merged_interpose_amass_train.pt \
agent.config.expert_model_path=results/full_body_tracker_merged \
+terrain=flat +experiment_name=masked_mimic_merged ngpu=4 \
eval_overrides.ngpu=4 num_envs=1024 agent.config.batch_size=4096 +opt=wandb training_max_steps=100000000000 \
agent.config.eval_metrics_every=1000 \
env.config.mimic_reward_config.component_weights.pow_rew_w=1.0e-4 \
+env.config.mimic_reward_config.component_weights.fc_rew_w=0.5 \
+env.config.mimic_reward_config.component_coefficients.fc_rew_c=-0.5 \
agent.config.vae.kld_schedule.start_epoch=6000 \
agent.config.vae.kld_schedule.end_epoch=12000
