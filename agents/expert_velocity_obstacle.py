import numpy as np

def compute_velocity_obstacle(uav_pos, uav_vel, obstacles, max_speed=5.0, safety_radius=2.0):
    """
    Compute a safe velocity using the Velocity Obstacle method.

    Parameters:
        uav_pos (np.ndarray): Position of the UAV (2D).
        uav_vel (np.ndarray): Current velocity of the UAV (2D).
        obstacles (list of dict): Each dict has 'pos' and 'vel' keys with 2D np.ndarrays.
        max_speed (float): Maximum allowed speed for the UAV.
        safety_radius (float): Minimum distance to maintain from obstacles.

    Returns:
        np.ndarray: New velocity vector (2D) that avoids collisions.
    """
    avoidance_vector = np.zeros(2)

    for obs in obstacles:
        rel_pos = obs['pos'] - uav_pos
        rel_vel = uav_vel - obs['vel']
        distance = np.linalg.norm(rel_pos)
        closing_rate = np.dot(rel_pos, rel_vel) / (distance + 1e-6)

        if distance < safety_radius and closing_rate > 0:
            # Move away from the obstacle
            avoidance_vector -= rel_pos / (distance + 1e-6)

    if np.linalg.norm(avoidance_vector) == 0:
        # No immediate collision risk, move forward
        return uav_vel if np.linalg.norm(uav_vel) <= max_speed else \
               (uav_vel / np.linalg.norm(uav_vel)) * max_speed
    else:
        # Normalize and scale
        safe_direction = avoidance_vector / (np.linalg.norm(avoidance_vector) + 1e-6)
        return safe_direction * max_speed
