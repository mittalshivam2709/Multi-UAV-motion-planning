import numpy as np

def compute_velocity_obstacle(uav_pos, uav_vel, obstacles, max_speed=5.0):
    safe_velocity = np.array([0.0, 0.0])
    for obs in obstacles:
        rel_pos = obs['pos'] - uav_pos
        rel_vel = obs['vel'] - uav_vel
        if np.linalg.norm(rel_pos) < 5.0:
            safe_velocity -= rel_pos / (np.linalg.norm(rel_pos) + 1e-5)
    return safe_velocity / np.linalg.norm(safe_velocity + 1e-5) * max_speed
