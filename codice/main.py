from bvh_reader.bvh_loader import read_bvh
from bvh_reader.Animation_deep import positions_global
from resampling import resample_animation
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

all_mpjpe = []
all_drift = []

# Function to align the animations by removing a number of extra frames determined manually in Unreal engine
def align_animations(label, opt_anim, moc_anim):
    frame_shifts = {
        "jumpingjacks": (0, 2),  # shift mocopi by 2 frames
        "running": (0, 0),       # no shift needed
        "squats": (0, 0),        # no shift needed
        "walking": (0, 0),       # no shift needed
        "take_1": (40, 0),       # shift optitrack by 40 frames
    }
    opt_shift, moc_shift = frame_shifts.get(label, (0, 0))

    #Shift animations by slicing
    opt_anim_aligned = opt_anim[opt_shift:] if opt_shift > 0 else opt_anim
    moc_anim_aligned = moc_anim[moc_shift:] if moc_shift > 0 else moc_anim

    # Align length
    n_frames = min(opt_anim_aligned.shape[0], moc_anim_aligned.shape[0])
    opt_anim_aligned = opt_anim_aligned[:n_frames]
    moc_anim_aligned = moc_anim_aligned[:n_frames]

    return opt_anim_aligned, moc_anim_aligned

# Function to rotate the skeleton - The angle is manually determined via visual inspection
def rotate_skeleton(skel_data, angle_deg, axis='y'):
    rot = R.from_euler(axis, angle_deg, degrees=True)
    rotated = rot.apply(skel_data.reshape(-1, 3)).reshape(skel_data.shape)
    return rotated

def compare_bvh_pairs(opt_path, moc_path, label="motion", output_dir="output", 
                      opt_fps=360, moc_fps=50, target_fps=50):
    # Load motions
    opt_anim, opt_joint_names, _ = read_bvh(opt_path)
    moc_anim, moc_joint_names, _ = read_bvh(moc_path)

    # Get world positions
    opt_anim_aligned, moc_anim_aligned = align_animations(label, opt_anim, moc_anim)
    opt_pos = positions_global(opt_anim_aligned)
    moc_pos = positions_global(moc_anim_aligned)

    # Resample to common FPS - NOT WORKING, scaled the animation to same FPS in Blender instead
    #opt_pos_resampled = resample_animation(opt_pos, opt_fps, target_fps) 
    #moc_pos_resampled = resample_animation(moc_pos, moc_fps, target_fps)

    opt_pos_resampled = opt_pos
    moc_pos_resampled = moc_pos

    #Manually rotate skeletons to match facing direction
    moc_pos_resampled = rotate_skeleton(moc_pos_resampled, angle_deg=90, axis='y')
    
    # Define the joint names (without prefix)
    joint_names_opt = ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    joint_names_moc = ["head", "l_hand", "r_hand", "l_foot", "r_foot"]

    # Build mapping: for each joint, find its index in both skeletons
    opt_indices = [opt_joint_names.index(name) for name in joint_names_opt]
    moc_indices = [moc_joint_names.index(name) for name in joint_names_moc]

    # DEBUGGING - Difference between hands
    opt_left = opt_pos_resampled[0, opt_joint_names.index("LeftHand")]
    opt_right = opt_pos_resampled[0, opt_joint_names.index("RightHand")]
    moc_left = moc_pos_resampled[0, moc_joint_names.index("l_hand")]
    moc_right = moc_pos_resampled[0, moc_joint_names.index("r_hand")]

    print("LeftHand diff:", np.linalg.norm(opt_left - moc_left))
    print("RightHand diff:", np.linalg.norm(opt_right - moc_right))

    #DEBUGGING - Frame 0 positions
    ##print("OptiTrack joint names:", opt_joint_names)
    #print("OptiTrack indices:", opt_indices)

    #print("OptiTrack root position (frame 0):", opt_pos_resampled[0, 0])
    #print("OptiTrack head position (frame 0):", opt_pos_resampled[0, opt_indices[0]])
    #print("Mocopi root position (frame 0):", moc_pos_resampled[0, 0])
    #print("Mocopi head position (frame 0):", moc_pos_resampled[0, moc_indices[0]])
    
    #print("OptiTrack frames:", opt_pos_resampled.shape[0])
    #print("Mocopi frames:", moc_pos_resampled.shape[0])

    # Align both skeletons so their lowest foot is at Y=0 in the first frame
    opt_floor = min(opt_pos_resampled[0, opt_indices[3], 1], opt_pos_resampled[0, opt_indices[4], 1])
    moc_floor = min(moc_pos_resampled[0, moc_indices[3], 1], moc_pos_resampled[0, moc_indices[4], 1])
    opt_pos_resampled[:, :, 1] -= opt_floor
    moc_pos_resampled[:, :, 1] -= moc_floor

    # Align initial root positions
    frame_align = 5  # Starting frame for alignment, takes in consideration the mismatch between resting pose and actual motion
    opt_root_offset = opt_pos_resampled[frame_align, 0, :]
    moc_root_offset = moc_pos_resampled[frame_align, 0, :]
    opt_pos_resampled -= opt_root_offset
    moc_pos_resampled -= moc_root_offset

    # DEBUG - Check root positions (should be [0,0,0] after alignment)
    print("OptiTrack root position (frame 0):", opt_pos_resampled[0, 0])
    print("Mocopi root position (frame 0):", moc_pos_resampled[0, 0])

    # DEBUG - Check facing direction (e.g., root to spine/chest)
    opt_dir = opt_pos_resampled[0, 1] - opt_pos_resampled[0, 0]
    moc_dir = moc_pos_resampled[0, 1] - moc_pos_resampled[0, 0]
    print("OptiTrack forward vector:", opt_dir)
    print("Mocopi forward vector:", moc_dir)
    print("Dot product:", np.dot(opt_dir, moc_dir))

    opt_norm = np.linalg.norm(opt_dir)
    moc_norm = np.linalg.norm(moc_dir)

    if opt_norm == 0 or moc_norm == 0:
        print("Warning: One of the direction vectors is zero, cannot compute angle.")
        angle = float('nan')
    else:
        cos_angle = np.clip(np.dot(opt_dir, moc_dir) / (opt_norm * moc_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        print("Angle between OptiTrack and Mocopi facing directions (deg):", angle)
        if angle > 10:  # threshold in degrees
            print("Warning: Skeletons are not well aligned in orientation (angle > 10 degrees).")

    
    # Compute root drift
    drift = np.linalg.norm(opt_pos_resampled[:, 0, :] - moc_pos_resampled[:, 0, :], axis=1)

    # DEBUG - frame 0 positions
    #print("OptiTrack root position (frame 0):", opt_pos_resampled[0, 0])
    #print("OptiTrack head position (frame 0):", opt_pos_resampled[0, opt_indices[0]])
    #print("Mocopi root position (frame 0):", moc_pos_resampled[0, 0])
    #print("Mocopi head position (frame 0):", moc_pos_resampled[0, moc_indices[0]])

    # DEBUG - Check height (root to head)
    #print("OptiTrack height (root to head):", np.linalg.norm(opt_pos_resampled[0, opt_indices[0]] - opt_pos_resampled[0, 0]))
    #print("Mocopi height (root to head):", np.linalg.norm(moc_pos_resampled[0, moc_indices[0]] - moc_pos_resampled[0, 0]))
    #print("OptiTrack head offset:", opt_anim.offsets[4])
    #print("OptiTrack head local position, frame 0:", opt_anim.positions[0, 4])
    #print("OptiTrack head parent index:", opt_anim.parents[4])
    #print("OptiTrack head parent name:", opt_joint_names[opt_anim.parents[4]])
    #print("OptiTrack neck global position, frame 0:", opt_pos_resampled[0, 3])

    # Compute scale factor to match OptiTrack to Mocopi (height is already similar now)
    #opt_height = np.linalg.norm(opt_pos_resampled[0, opt_indices[0]] - opt_pos_resampled[0, 0])
    #moc_height = np.linalg.norm(moc_pos_resampled[0, moc_indices[0]] - moc_pos_resampled[0, 0])
    #scale = moc_height / opt_height
    #opt_pos_resampled *= scale
    #print(f"Applied scale factor to OptiTrack: {scale}")

    # DEBUG - Print first 5 frames of selected joints
    #for i in range(5):
    #    print(f"Frame {i}:")
    #    for name, opt_idx, moc_idx in zip(joint_names_opt, opt_indices, moc_indices):
    #        print(f"  {name}: OptiTrack {opt_pos_resampled[i, opt_idx]}, Mocopi {moc_pos_resampled[i, moc_idx]}")

    # Select only those joints for MPJPE
    opt_selected = opt_pos_resampled[:, opt_indices, :]
    moc_selected = moc_pos_resampled[:, moc_indices, :]

    # Get root joint positions (aligned to frame_align)
    opt_root = opt_pos_resampled[:, 0:1, :]
    moc_root = moc_pos_resampled[:, 0:1, :]

    # Subtract root from the selected joints (align all selected joints to the root)
    opt_aligned = opt_selected - opt_root
    moc_aligned = moc_selected - moc_root
    mpjpe = np.linalg.norm(opt_aligned - moc_aligned, axis=2).mean(axis=1)

    # Ignore initial frames before alignment at frame 5
    drift = drift[frame_align:]
    mpjpe = mpjpe[frame_align:]

    mpjpe_mean = np.mean(mpjpe)
    mpjpe_std = np.std(mpjpe)
    mpjpe_max = np.max(mpjpe)
    mpjpe_min = np.min(mpjpe)

    drift_mean = np.mean(drift)
    drift_std = np.std(drift)
    drift_max = np.max(drift)
    drift_min = np.min(drift)

    all_mpjpe.append(mpjpe)
    all_drift.append(drift)

    print(f"{label}: MPJPE mean={mpjpe_mean:.2f}, std={mpjpe_std:.2f}, max={mpjpe_max:.2f}, min={mpjpe_min:.2f}")
    print(f"{label}: Drift mean={drift_mean:.2f}, std={drift_std:.2f}, max={drift_max:.2f}, min={drift_min:.2f}")


    # DEBUG - Check shape of opt_aligned 
    print("OptiTrack aligned shape:", opt_aligned.shape)

    #DEBUG - Check first 5 frames of head and root positions
    #print("First 5 frames, Head positions (OptiTrack):")
    #print(opt_pos_resampled[:5, opt_indices[0], :])
    #print("First 5 frames, Head positions (Mocopi):")
    #print(moc_pos_resampled[:5, moc_indices[0], :])
    #print("First 5 frames, Root positions (OptiTrack):")
    #print(opt_pos_resampled[:5, 0, :])
    #print("First 5 frames, Root positions (Mocopi):")
    #print(moc_pos_resampled[:5, 0, :])

    print("Joint mapping:")
    for opt_idx, moc_idx, name in zip(opt_indices, moc_indices, joint_names_opt):
        dist = np.linalg.norm(opt_pos_resampled[:, opt_idx, :] - moc_pos_resampled[:, moc_idx, :], axis=1).mean()
        print(f"{name}: mean error = {dist:.2f} cm")

    # Create output folder if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    np.savetxt(f"{output_dir}/{label}/{label}_drift.csv", drift, delimiter=",")
    np.savetxt(f"{output_dir}/{label}/{label}_mpjpe.csv", mpjpe, delimiter=",")

    # Plot
    plt.figure()
    plt.plot(drift, label="Root Drift (cm)")
    plt.plot(mpjpe, label="MPJPE (cm)")
    plt.legend()
    plt.title(f"{label}: Mocopi vs OptiTrack Error")
    plt.xlabel("Frame")
    plt.ylabel("Distance (cm)")
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.savefig(f"{output_dir}/{label}/{label}_plot.png")
    plt.close()

    # Print summary
    print(f"{label}: final drift = {drift[-1]:.2f} cm, mean MPJPE = {np.mean(mpjpe):.2f} cm")

# MAIN SECTION
if __name__ == "__main__":
    compare_bvh_pairs(
        "YOUR OPTITRACK PATH",
        "YOUR MOCOPI PATH",
        label="jumpingjacks"
    )
    compare_bvh_pairs(
        "YOUR OPTITRACK PATH",
        "YOUR MOCOPI PATH",
        label="running"
    )
    compare_bvh_pairs(
        "YOUR OPTITRACK PATH",
        "YOUR MOCOPI PATH",
        label="squats"
    )
    compare_bvh_pairs(
        "YOUR OPTITRACK PATH",
        "YOUR MOCOPI PATH",
        label="walking"
    )
    compare_bvh_pairs(
        "YOUR OPTITRACK PATH",
        "YOUR MOCOPI PATH",
        label="take_1"
    )

    # After all animations:
    all_mpjpe_flat = np.concatenate(all_mpjpe)
    all_drift_flat = np.concatenate(all_drift)

    print("\n=== AGGREGATE METRICS ACROSS ALL ANIMATIONS ===")
    print(f"MPJPE: mean={np.mean(all_mpjpe_flat):.2f}, std={np.std(all_mpjpe_flat):.2f}, max={np.max(all_mpjpe_flat):.2f}, min={np.min(all_mpjpe_flat):.2f}")
    print(f"Drift: mean={np.mean(all_drift_flat):.2f}, std={np.std(all_drift_flat):.2f}, max={np.max(all_drift_flat):.2f}, min={np.min(all_drift_flat):.2f}")

    
