export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

python data/scripts/convert_amass_to_isaac.py data/Dataset/interpose_data --humanoid-type=smpl

## remove some impossible sample
python data/scripts/detect_occlusion_contact.py --data_path data/Dataset/interpose_data --save_path data/Dataset/interpose_data/interpose_data_copycat_occlusion_v3_filter_abnormal.pkl

## generate yaml files
python data/scripts/process_my_data.py --body_format smpl \
    --root_path data/Dataset/interpose_data \
    --output_path data/Dataset/interpose_data/interpose_smpl_train_final.yaml \
    --include_txt_file data/yaml_files/interpose_train.txt \
    --occlusion_data_path data/Dataset/interpose_data/interpose_data_copycat_occlusion_v3_filter_abnormal.pkl

python data/scripts/process_my_data.py --body_format smpl \
    --root_path data/Dataset/interpose_data \
    --output_path data/Dataset/interpose_data/interpose_smpl_test_final.yaml \
    --include_txt_file data/yaml_files/interpose_test.txt \
    --occlusion_data_path data/Dataset/interpose_data/interpose_data_copycat_occlusion_v3_filter_abnormal.pkl

## merge AMASS data
python data/scripts/merge_additional_our_data.py data/yaml_files/amass_train.yaml data/Dataset/interpose_data/interpose_smpl_train_final.yaml \
data/Dataset/interpose_data/merged_interpose_amass_train.yaml

python data/scripts/merge_additional_our_data.py data/yaml_files/amass_test.yaml data/Dataset/interpose_data/interpose_smpl_test_final.yaml \
data/Dataset/interpose_data/merged_interpose_amass_test.yaml

## package merged data
python data/scripts/package_motion_lib.py data/Dataset/interpose_data/merged_interpose_amass_train.yaml \
amass_data data/Dataset/interpose_data/merged_interpose_amass_train.pt --humanoid-type=smpl --create-text-embeddings

python data/scripts/package_motion_lib.py data/Dataset/interpose_data/merged_interpose_amass_test.yaml \
amass_data data/Dataset/interpose_data/merged_interpose_amass_test.pt --humanoid-type=smpl --create-text-embeddings
