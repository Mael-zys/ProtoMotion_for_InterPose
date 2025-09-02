export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

# Convert the motions to MotionLib format
python data/scripts/convert_amass_to_isaac.py data/Dataset/amass_data

# Package the data for faster loading
python data/scripts/package_motion_lib.py data/yaml_files/amass_train.yaml \
data/Dataset/amass_data data/Dataset/amass_data/amass_train_smpl.pt --create-text-embeddings

python data/scripts/package_motion_lib.py data/yaml_files/amass_test.yaml \
data/Dataset/amass_data data/Dataset/amass_data/amass_test_smpl.pt --create-text-embeddings