import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bvh_reader.bvh_loader import read_bvh
from bvh_reader.Animation_deep import positions_global

def plot_two_skeletons(global_pos1, names1, parents1, 
                       global_pos2, names2, parents2, 
                       frame_idx=10, color1='blue', color2='red', 
                       label1='OptiTrack', label2='Mocopi'):
    highlight = {"Head", "Hips", "Hip", "LeftHand", "RightHand", "LeftFoot", "RightFoot",
                 "head", "root", "l_hand", "r_hand", "l_foot", "r_foot"}

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Frame {frame_idx} - {label1} (blue) vs {label2} (red)")

    joints1 = global_pos1[frame_idx]
    joints2 = global_pos2[frame_idx]

    # Plot bones for OptiTrack
    for i, p in enumerate(parents1):
        if p == -1:
            continue
        ax.plot([joints1[i, 0], joints1[p, 0]],
                [joints1[i, 1], joints1[p, 1]],
                [joints1[i, 2], joints1[p, 2]],
                c=color1, lw=2)
    # Plot all joints for OptiTrack
    ax.scatter(joints1[:, 0], joints1[:, 1], joints1[:, 2], c=color1, s=40, label=label1)
    # Optionally label highlighted joints
    for i, name in enumerate(names1):
        if name in highlight:
            ax.text(joints1[i, 0], joints1[i, 1], joints1[i, 2], name, fontsize=9, color=color1)

    # Plot bones for Mocopi
    for i, p in enumerate(parents2):
        if p == -1:
            continue
        ax.plot([joints2[i, 0], joints2[p, 0]],
                [joints2[i, 1], joints2[p, 1]],
                [joints2[i, 2], joints2[p, 2]],
                c=color2, lw=2)
    # Plot all joints for Mocopi
    ax.scatter(joints2[:, 0], joints2[:, 1], joints2[:, 2], c=color2, s=40, label=label2)
    # Optionally label highlighted joints
    for i, name in enumerate(names2):
        if name in highlight:
            ax.text(joints2[i, 0], joints2[i, 1], joints2[i, 2], name, fontsize=9, color=color2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Example usage:
opt_anim, opt_names, _ = read_bvh('YOUR OPTITRACK PATH')
moc_anim, moc_names, _ = read_bvh('YOUR MOCOPI PATH')

opt_pos = positions_global(opt_anim)
moc_pos = positions_global(moc_anim)

print("OptiTrack parents:", opt_anim.parents)
print("OptiTrack joint names:", opt_names)
print("Mocopi parents:", moc_anim.parents)
print("Mocopi joint names:", moc_names)

plot_two_skeletons(opt_pos, opt_names, opt_anim.parents, moc_pos, moc_names, moc_anim.parents, frame_idx=0)