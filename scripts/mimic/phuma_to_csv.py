"""Convert PHUMA retargeted .npy motion files to CSV format for csv_to_npz.py

PHUMA's G1 retargeted data has 23 DOF (G1-29dof minus 6 wrist joints).
This script zero-pads the missing wrist joints to produce a 29-DOF CSV
that can be fed into csv_to_npz.py.

PHUMA format (per .npy dict):
    root_trans: (N, 3)   - root position, Z-up, meters
    root_ori:   (N, 4)   - root quaternion, xyzw (scipy convention)
    dof_pos:    (N, 23)  - joint positions, radians
    fps:        int       - frame rate (typically 30)

Output CSV columns (per row):
    base_pos(3), base_quat_xyzw(4), joint_pos(29)  = 36 columns

Usage:
    # Single file
    python phuma_to_csv.py -f /path/to/motion.npy

    # Batch: all .npy files in a directory
    python phuma_to_csv.py -d /path/to/phuma/g1/

    # Then generate NPZ for training:
    # isaaclab -p scripts/mimic/csv_to_npz.py -f output.csv --input_fps 30
"""

import argparse
import os
import glob

import numpy as np

# ── Joint ordering ──────────────────────────────────────────────────────────
# PHUMA G1 23-DOF order (indices 0-22):
PHUMA_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "right_shoulder_pitch_joint", # 19
    "right_shoulder_roll_joint",  # 20
    "right_shoulder_yaw_joint",   # 21
    "right_elbow_joint",          # 22
]

# unitree_rl_lab G1 29-DOF SDK order:
UNITREE_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19  ← MISSING in PHUMA
    "left_wrist_pitch_joint",     # 20  ← MISSING in PHUMA
    "left_wrist_yaw_joint",       # 21  ← MISSING in PHUMA
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26  ← MISSING in PHUMA
    "right_wrist_pitch_joint",    # 27  ← MISSING in PHUMA
    "right_wrist_yaw_joint",      # 28  ← MISSING in PHUMA
]

# Build mapping: for each of the 29 target joints, which PHUMA index (-1 if missing)
PHUMA_TO_29DOF_MAP = []
for joint_name in UNITREE_29DOF_JOINT_NAMES:
    if joint_name in PHUMA_JOINT_NAMES:
        PHUMA_TO_29DOF_MAP.append(PHUMA_JOINT_NAMES.index(joint_name))
    else:
        PHUMA_TO_29DOF_MAP.append(-1)  # missing → zero-pad


def convert_phuma_to_csv(npy_path: str, output_path: str | None = None) -> str:
    """Convert a single PHUMA .npy file to CSV.

    Args:
        npy_path: Path to the PHUMA .npy file.
        output_path: Path for the output CSV. If None, replaces .npy with .csv.

    Returns:
        Path to the output CSV file.
    """
    if output_path is None:
        output_path = npy_path.replace(".npy", ".csv")

    data = np.load(npy_path, allow_pickle=True).item()

    root_trans = data["root_trans"]  # (N, 3)
    root_ori = data["root_ori"]     # (N, 4) xyzw
    dof_pos = data["dof_pos"]       # (N, 23)
    fps = data["fps"]
    num_frames = root_trans.shape[0]

    print(f"Loading: {npy_path}")
    print(f"  Frames: {num_frames}, FPS: {fps}, Duration: {num_frames / fps:.2f}s")

    # Map 23-DOF → 29-DOF (zero-pad wrist joints)
    dof_pos_29 = np.zeros((num_frames, 29), dtype=np.float64)
    for target_idx, source_idx in enumerate(PHUMA_TO_29DOF_MAP):
        if source_idx >= 0:
            dof_pos_29[:, target_idx] = dof_pos[:, source_idx]
        # else: stays 0.0 (wrist joints)

    # CSV format: [base_pos(3), base_quat_xyzw(4), joint_pos(29)] = 36 columns
    # csv_to_npz.py expects xyzw quaternion and converts to wxyz internally
    csv_data = np.hstack([root_trans, root_ori, dof_pos_29])
    assert csv_data.shape == (num_frames, 36), f"Expected (N, 36), got {csv_data.shape}"

    np.savetxt(output_path, csv_data, delimiter=",", fmt="%.10f")
    print(f"  Saved: {output_path} ({num_frames} frames, {csv_data.shape[1]} columns)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert PHUMA .npy to CSV for csv_to_npz.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a single PHUMA .npy file")
    group.add_argument("-d", "--dir", type=str, help="Directory containing PHUMA .npy files (batch)")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory for CSV files (default: same as input)")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.npy")))
        print(f"Found {len(files)} .npy files in {args.dir}")

    for npy_path in files:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            basename = os.path.basename(npy_path).replace(".npy", ".csv")
            output_path = os.path.join(args.output_dir, basename)
        else:
            output_path = None

        try:
            convert_phuma_to_csv(npy_path, output_path)
        except Exception as e:
            print(f"  ERROR processing {npy_path}: {e}")

    print(f"\nDone! Converted {len(files)} files.")
    print(f"\nNext step: generate NPZ for training with Isaac Sim:")
    print(f"  isaaclab -p scripts/mimic/csv_to_npz.py -f <output.csv> --input_fps 30")


if __name__ == "__main__":
    main()
