import os
import glob
import argparse
import numpy as np
import joblib
from tqdm import tqdm
import smplx
import torch
from data.scripts.detect_abnormalty import detect_issues

# Define joint indices according to your skeleton format
LEFT_FOOT_IDX = 10    
RIGHT_FOOT_IDX = 11   

# Keypoint indices for detecting sitting/squatting
PELVIS_IDX = 0
HEAD_IDX = 15
LEFT_KNEE_IDX = 4
RIGHT_KNEE_IDX = 5


def main(args):
    npz_files = glob.glob(os.path.join(args.data_path, "**/*.npz"), recursive=True)
    print(f"Found {len(npz_files)} npz files")

    occlusion_dict = {}
    issue_counter = {"airborne": 0, "sitting": 0, "no_floor_contact": 0, 
                     "orientation_jump": 0, "orientation_acc_jump": 0, 
                     "translation_jump": 0, "translation_acc_jump": 0, 
                     "total": 0}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = smplx.create('data/smpl/SMPLX_NEUTRAL.npz', model_type='smplx', gender='neutral', use_face_contour=True,
                         num_expression_coeffs=10, device=device, use_pca=False, flat_hand_mean=True)
    model.to(device)
    for npz_file in tqdm(npz_files, desc="Checking motions"):
        try:
            data = dict(np.load(npz_file, allow_pickle=True))
        except Exception as e:
            print(f"Failed to load {npz_file}: {e}")
            continue
        
        object_support_action = [' sits ',' sitting ', 'ride', ' ski', 'upstairs', 'downstairs', ' lying ', ' lies ', ' skating ']
        continue_flag = False
        for action in object_support_action:
            if action in str(data['text']).lower():
                rel_key = os.path.relpath(npz_file, args.data_path)
                rel_key = rel_key.replace(os.sep, "_").replace(".npz", "")
                issue = 'sitting'
                first_idx = 0
                occlusion_dict[rel_key] = {"issue": issue, "idxes": [first_idx]}
                issue_counter[issue] += 1
                print(f"Detected {issue} in {rel_key} at frame {first_idx}")
                issue_counter["total"] += 1
                continue_flag = True
                break
        if continue_flag:
            continue        
        
        if 'trans' not in data:
            continue
        if 'joints3d' not in data:
            
            batch_size = data['poses'].shape[0]
            params = {}
            params['body_pose'] = data['poses'][:, 3:66].reshape(batch_size, 21, 3)
            params['global_orient'] = data['poses'][:, :3].reshape(batch_size, 3)
            params['betas'] = np.tile(data['betas'][None, :], (batch_size, 1)).reshape(batch_size, 10)
            params['transl'] = data['trans'].reshape(batch_size, 3)
            params['left_hand_pose'] = data['poses'][:, 75:120].reshape(batch_size, 15, 3)
            params['right_hand_pose'] = data['poses'][:, 120:165].reshape(batch_size, 15, 3)

            params['expression'] = np.zeros((batch_size, 10))
            params['jaw_pose'] = np.zeros((batch_size, 3))
            params['leye_pose'] = np.zeros((batch_size, 3))
            params['reye_pose'] = np.zeros((batch_size, 3))

            
            # Prepare batch parameters for SMPLX
            batch_params = {}

            # Process other parameters
            for key in params:
                if key in ['fps', 'frame_numbers', 'gender']:
                    continue
                    
                if isinstance(params[key], np.ndarray) and params[key].size > 0:
                    # Extract data for all frames in batch
                    batch_data = params[key]
                    
                    # Convert to tensor
                    tensor = torch.tensor(batch_data, device=device).float()
                    
                    # Adjust shape based on parameter type
                    if key in ['global_orient', 'jaw_pose', 'leye_pose', 'reye_pose']:
                        tensor = tensor.reshape(batch_size, 3)
                    elif key in ['body_pose']:
                        tensor = tensor.reshape(batch_size, 21, 3)
                    elif key in ['left_hand_pose', 'right_hand_pose']:
                        tensor = tensor.reshape(batch_size, 15, 3)
                    elif key in ['expression']:
                        tensor = tensor.reshape(batch_size, 10)
                    elif key in ['transl']:
                        tensor = tensor.reshape(batch_size, 3)

                    
                    batch_params[key] = tensor
            
            # Forward pass to get SMPLX output for all frames at once
            try:
                output = model(**batch_params)
            except RuntimeError as e:
                print(f"Error in batch processing:")
                for k, v in batch_params.items():
                    print(f"  {k}: {v.shape}")
                raise e
            
            # Get joints for all frames
            joints3d = output.joints.detach().cpu().numpy()
            joints3d = joints3d.reshape(batch_size, -1, 3)  # (N, J, 3)
            joints3d -= joints3d[0:1, 0:1, :].reshape(1, 1, 3)  # Center the joints to the first frame
            joints3d[:, :, 2] -= joints3d[:, :, 2].min(keepdims=True)  # Adjust Z axis to start from 0 based on all frames
            # joints3d = None
            
        else:
            joints3d = data['joints3d']  # (N,J,3)

        trans = data['trans']  # (N,3)
        trans[:, 2] = joints3d[:, 0, 2]  # Adjust Z axis to start from 0 based on all frames
        issue, first_idx = detect_issues(
            trans,
            joints3d=joints3d,
            global_orient=data['poses'][:, :3].reshape(-1, 3),
            ignore_sitting=False
        )

        if issue is not None and len(issue) > 0:
            rel_key = os.path.relpath(npz_file, args.data_path)
            rel_key = rel_key.replace(os.sep, "_").replace(".npz", "")
            occlusion_dict[rel_key] = {"issue": issue, "idxes": [int(first_idx)]}
            print(f"Detected {issue} in {rel_key} at frame {first_idx}")

        issue_counter["total"] += 1

    print(f"\n=== Detection Summary ===")
    print(f"Total motions checked: {issue_counter['total']}")


    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    joblib.dump(occlusion_dict, args.save_path, compress=True)
    print(f"\nSaved occlusion dict to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to folder containing .npz motion files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output occlusion pkl")
    parser.add_argument("--airborne_thresh", type=float, default=1.4, help="Z height threshold for airborne detection")
    parser.add_argument("--sitting_thresh", type=float, default=0.4, help="Z height threshold for sitting detection")
    parser.add_argument("--airborne_ratio", type=float, default=0.3, help="Min frame ratio for airborne")
    parser.add_argument("--sitting_ratio", type=float, default=0.3, help="Min frame ratio for sitting")
    parser.add_argument("--contact_thresh", type=float, default=0.2, help="Foot contact Z threshold (meters)")
    parser.add_argument("--contact_miss_ratio", type=float, default=0.5, help="If more than X'%' frames missing foot contact -> error")
    args = parser.parse_args()
    main(args)
