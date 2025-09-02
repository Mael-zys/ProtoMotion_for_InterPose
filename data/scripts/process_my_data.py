import yaml
import os
import numpy as np
import joblib
import tqdm
import argparse
import codecs as cs

def main(args):
    body_format = args.body_format
    root_path = args.root_path
    output_path = args.output_path
    occlusion_data_path = args.occlusion_data_path
    fail_motion_path = args.fail_motion_path

    data_name_list = os.listdir(root_path)
    motion_list = []
    motion_idx = 0

    # Check if occlusion data exists
    if os.path.exists(occlusion_data_path):
        occlusion_data = joblib.load(occlusion_data_path)
        print(f"Loaded occlusion data from {occlusion_data_path}")
    else:
        occlusion_data = {}
        print(f"Occlusion data file not found at {occlusion_data_path}, skipping occlusion checks.")

    exclude_file_list = []
    if args.exclude_yaml_file and os.path.exists(args.exclude_yaml_file):
        with open(args.exclude_yaml_file, "r") as f:
            exclude_file = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        for exclude_file_dict in exclude_file:
            exclude_file_list.append(exclude_file_dict['file'])
    exclude_file_list = set(exclude_file_list)

    id_list = []
    if args.include_txt_file and os.path.exists(args.include_txt_file):
        with cs.open(args.include_txt_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
    id_list = set(id_list)

    exclude_id_list = []

    if args.exclude_txt_file and os.path.exists(args.exclude_txt_file):
        with cs.open(args.exclude_txt_file, 'r') as f:
            for line in f.readlines():
                exclude_id_list.append(line.strip())
    exclude_id_list = set(exclude_id_list)
    
    # Check if fail motion file exists
    if os.path.exists(fail_motion_path):
        with open(fail_motion_path, 'r') as f:
            failed_motions_idx_list = set(line.strip() for line in f if line.strip())

        # Read existing YAML if available
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing_motion_file = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
            failed_motion_name_list = []
            for failed_motion_idx in failed_motions_idx_list:
                try:
                    failed_motion_name = existing_motion_file[int(failed_motion_idx)]['file']
                    releative_path = os.path.relpath(failed_motion_name, root_path)
                    failed_motion_name = releative_path.replace(os.sep, "_").replace(".npy", "").replace(f"-{body_format}", "")
                    failed_motion_name_list.append(failed_motion_name)
                except (IndexError, KeyError):
                    print(f"Warning: Failed motion index {failed_motion_idx} not found in existing YAML")
            failed_motions = set(failed_motion_name_list)
            # Rename output to avoid overwriting
            if args.only_failed_motion:
                output_path = output_path.replace(".yaml", "_only_failed_motions.yaml")
            else:
                output_path = output_path.replace(".yaml", "_with_failed_motions.yaml")
            print(f"Loaded {len(failed_motions)} failed motions from {fail_motion_path}")
        else:
            print(f"Warning: {output_path} does not exist, cannot map failed motions by index, skipping fail list.")
    else:
        print(f"Fail motion file not found at {fail_motion_path}, skipping failed motion checks.")

    total_duration = 0

    for i, path in enumerate(tqdm.tqdm(data_name_list)):
        if '-smplx' in path or '-smpl' in path:
            continue
        full_path = os.path.join(root_path, path)
        if not os.path.isdir(full_path):
            print(f"Warning: {full_path} is not a directory, skipping...")
            continue

        print(f"Processing {path} ({i + 1}/{len(data_name_list)})")
        motion_name_list = os.listdir(full_path)
        for j, name in enumerate(motion_name_list):
            if '.npz' not in name:
                continue

            if args.include_txt_file and os.path.exists(args.include_txt_file):
                if name.replace(".npz", "") not in id_list:
                    continue

            if args.exclude_txt_file and os.path.exists(args.exclude_txt_file):
                if name.replace(".npz", "") in exclude_id_list:
                    continue

            key_name = os.path.join(path, name).replace(os.sep, "_").replace(".npz", "")

            if args.only_failed_motion:
                if key_name not in failed_motions:
                    print(f"Warning: {key_name} is not in failed motions list, skipping...")
                    continue
            elif os.path.exists(fail_motion_path) and key_name in failed_motions:
                print(f"Warning: {key_name} is in failed motions list, skipping...")
                continue

            end_time = float("inf")

            if os.path.exists(occlusion_data_path) and key_name in occlusion_data:
                this_motion_occlusion = occlusion_data[key_name]
                if this_motion_occlusion['idxes'][0] < 10:
                    print(f"Warning: {key_name} has occlusion at frame {this_motion_occlusion['idxes'][0]}, skipping...")
                    continue
                else:
                    end_time = this_motion_occlusion['idxes'][0]

            motion_file_path = os.path.join(root_path, path + f'-{body_format}', name.replace(".npz", ".npy")
                .replace("-", "_")
                .replace(" ", "_")
                .replace("(", "_")
                .replace(")", "_"))

            if not os.path.exists(motion_file_path):
                print(f"Warning: Missing {motion_file_path}, skipping...")
                continue
            
            if motion_file_path in exclude_file_list:
                print(f"Warning: {motion_file_path} exists in exclude file {args.exclude_yaml_file}, skipping...")
                continue

            motion_data = np.load(motion_file_path, allow_pickle=True).item()
            ori_motion_data = np.load(os.path.join(root_path, path, name))

            end_time = min(end_time, ori_motion_data['trans'].shape[0])

            motion_list.append({
                "file": motion_file_path,
                "fps": int(motion_data['fps']),
                "idx": motion_idx,
                "sub_motions": [
                    {
                        "hml3d_id": motion_idx,
                        "idx": motion_idx,
                        "labels": [
                            str(ori_motion_data['text']) if 'text' in ori_motion_data else ''
                        ],
                        "timings": {
                            "start": 0.0,
                            "end": end_time / int(motion_data['fps'])
                        },
                        "weight": 1.0
                    }
                ]
            })
            motion_idx += 1
            total_duration += end_time / int(motion_data['fps'])

    data = {
        'motions': motion_list
    }

    with open(output_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    print(f"YAML file generated: {output_path}")
    print(f"Total motion sequences: {motion_idx}. Total duration: {total_duration / 3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--body_format", type=str, default="smpl")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--occlusion_data_path", type=str, default="")
    parser.add_argument("--fail_motion_path", type=str, default="")
    parser.add_argument("--only_failed_motion", action="store_true")
    parser.add_argument("--exclude_yaml_file", type=str, default=None)
    parser.add_argument("--include_txt_file", type=str, default=None)
    parser.add_argument("--exclude_txt_file", type=str, default=None)
    args = parser.parse_args()

    main(args)
