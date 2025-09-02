# ProtoMotions: Physics-based Character Animation

- [What is this?](#what-is-this)
- [Installation guide](#installation)
- [Data preparation](#data-preparation)
- [Training built-in agents](#training-your-agent)
- [Evaluating your agent](#evaluationvisualization)

# What is this?

This codebase is based on [ProtoMotions](https://github.com/NVlabs/ProtoMotions/tree/main). We use MaskedMimic as a baseline to conduct experiments for spatial control and zero-shot human-object interaction with our proposed dataset [InterPose](https://mael-zys.github.io/InterPose/).

> **Important:**</br>
> This codebase builds heavily on [Hydra](https://hydra.cc/) and [OmegaConfig](https://omegaconf.readthedocs.io/).<br>
> It is recommended to familiarize yourself with these libraries and how config composition works.

# Installation

This codebase supports IsaacGym.

First run `git lfs fetch --all` to fetch all files stored in git-lfs.

Run the installation in one script:
```bash
bash scripts/install_protomotion.sh
```

or follow this step-by-step guide below:
<details>
<summary>IsaacGym</summary>

1. Install [IsaacGym](https://developer.nvidia.com/isaac-gym)
```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```
2. Once IG and PyTorch are installed, from the repository root install the ProtoMotions package and its dependencies with:
```bash
pip install -e .
pip install -r requirements_isaacgym.txt
pip install -e isaac_utils
pip install -e poselib
```
Set the `PYTHON_PATH` env variable (not really needed, but helps the instructions stay consistent between sim and gym).
```bash
alias PYTHON_PATH=python
```

### Potential Issues

If you have python errors:

```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
```

If you run into memory issues -- try reducing the number of environments by adding to the command line `num_envs=1024`

</details>


# Data Preparation


## Download SMPL parameters
Download the [SMPL](https://smpl.is.tue.mpg.de/) v1.1.0 parameters and place them in the `data/smpl/` folder. Rename the files:
- basicmodel_neutral_lbs_10_207_0_v1.1.0 -> SMPL_NEUTRAL.pkl
- basicmodel_m_lbs_10_207_0_v1.1.0.pkl -> SMPL_MALE.pkl
- basicmodel_f_lbs_10_207_0_v1.1.0.pkl -> SMPL_FEMALE.pkl

## Download pretrained models

Download the [pretrained model and config file](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/yangsong_zhang_mbzuai_ac_ae/EkEslm3PEvdFrKl_23hNYj0BLOg24tjmBPWC6Hey-YstCA?e=PLQYMO) and put them under `results/masked_mimic_merged/`.

## Data Preprocessing

Download the [processed data](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/yangsong_zhang_mbzuai_ac_ae/EtJ6LsQZs0pCpVY-8IBy5hkBoZgLE7Qx1CVXJxTwl9tVJw?e=pxayix):

Note that, the InterPose data is only used for training, while for evaluation, we use the AMASS, OMOMO and BEHAVE dataset.

```bash
data/Dataset/amass_data/                         
data/Dataset/behave_data/   
data/Dataset/interpose_data/ 
data/Dataset/omomo_data/            
```


Or follow the detail instructions below:

<details>
<summary>Data Preprocessing</summary>

### Data Download
1. Download the [AMASS](https://amass.is.tue.mpg.de/) dataset and put it in the `data/Dataset/amass_data/` folder.
2. Download the [InterPose](https://mael-zys.github.io/InterPose/) dataset and put it in the `data/Dataset/interpose_data/` folder.

### Data Conversion

Run the following scripts to convert the AMASS and InterPose data to the MotionLib format and package them for faster loading.
```bash
bash scripts/convert_amass.sh
bash scripts/convert_InterPose.sh
```

More details can be found in [ProtoMotions](https://github.com/NVlabs/ProtoMotions/tree/main?tab=readme-ov-file#data).
</details>


Motions can be visualized via kinematic replay by running `PYTHON_PATH protomotions/scripts/play_motion.py <motion file> <simulator isaacgym/isaaclab/genesis> <robot type>`.

# Training Your Agent

We provide an example training script to train MaskedMimic with our dataset:
```bash
bash scripts/train_maskedmimic_smpl_merge.sh
```



# Evaluation

We provide some evaluation scripts to calculate the spatial controllability on AMASS, OMOMO and BEHAVE test set:

Pelvis position:
```bash
sh scripts/evaluate_final.sh results/masked_mimic_merged/last.ckpt merged pelvis_position 0.5
```

Two hands:
```bash
sh scripts/evaluate_final.sh results/masked_mimic_merged/last.ckpt merged hands 0.5
```

Here is an example command to visualize the agent's performance for hands control:
```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0

python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym +opt=masked_mimic/constraints/hands \
+motion_file=data/Dataset/omomo_data/omomo_test_smpl.pt \
+checkpoint=./results/masked_mimic_merged/last.ckpt \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +terrain=flat 
```


We provide a set of pre-defined keyboard controls.

| Key | Description                                                                |
|-----|----------------------------------------------------------------------------|
| `J` | Apply physical force to all robots (tests robustness)                      |
| `R` | Reset the task                                                             |
| `O` | Toggle camera. Will cycle through all entities in the scene.               |
| `L` | Toggle recording video from viewer. Second click will save frames to video |
| `;` | Cancel recording                                                           |
| `U` | Update inference parameters (e.g., in MaskedMimic user control task)       |
| `Q` | Quit       |


# References
This project repository builds upon [ProtoMotions](https://github.com/NVlabs/ProtoMotions/tree/main).
