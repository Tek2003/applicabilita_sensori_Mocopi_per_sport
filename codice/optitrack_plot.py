import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from bvh_reader.bvh_loader import read_bvh
from bvh_reader.Animation_deep import positions_global

def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_skeleton_with_indices(global_pos, joint_names, parents, frame_idx=0, color='blue', label='OptiTrack'):
    # List of substrings that identify finger joints
    finger_keywords = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{label} Skeleton - Frame {frame_idx}")

    joints = global_pos[frame_idx]

    # Plot bones
    for i, p in enumerate(parents):
        if p == -1:
            continue
        ax.plot([joints[i, 0], joints[p, 0]],
                [joints[i, 1], joints[p, 1]],
                [joints[i, 2], joints[p, 2]],
                c=color, lw=2)

    # Plot all joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=40, label=label)

    # Label joints with index, except fingers (for clarity)
    offset = np.array([2.0, 2.0, 0.0])  # Change values to adjust distance and direction
    for i, name in enumerate(joint_names):
        if not any(finger in name for finger in finger_keywords):
            ax.text(*(joints[i] + offset), str(i), fontsize=10, color='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    set_axes_equal(ax)
    plt.show()


opt_anim, opt_names, _ = read_bvh('YOUR OPTITRACK PATH')
opt_pos = positions_global(opt_anim)

plot_skeleton_with_indices(opt_pos, opt_names, opt_anim.parents, frame_idx=0)
