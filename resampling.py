import numpy as np
from scipy.interpolate import interp1d

def resample_animation(pos_array, original_fps, target_fps):
    n_frames, n_joints, dims = pos_array.shape
    duration = n_frames / original_fps
    original_times = np.linspace(0, duration, n_frames)
    target_n_frames = int(duration * target_fps)
    target_times = np.linspace(0, duration, target_n_frames)

    pos_resampled = np.zeros((target_n_frames, n_joints, dims))
    for j in range(n_joints):
        for d in range(dims):
            f = interp1d(original_times, pos_array[:, j, d], kind='linear')
            pos_resampled[:, j, d] = f(target_times)
    return pos_resampled
