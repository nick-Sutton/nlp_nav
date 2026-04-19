#!/usr/bin/env python3

import rclpy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import distance_transform_edt
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import matplotlib.pyplot as plt


def compute_likelihood_field(occupancy_grid):
    """
    Compute the likelihood field (distance transform) from an occupancy grid.

    The likelihood field stores, for each cell, the distance to the nearest
    obstacle. This is used by the sensor model to quickly evaluate how close
    a LiDAR endpoint is to the nearest wall.

    Args:
        occupancy_grid: object with .info (width, height, resolution) and .data

    Returns:
        2D numpy array of distances in meters (shape: height x width)
    """
    width = occupancy_grid.info.width
    height = occupancy_grid.info.height
    data = np.array(occupancy_grid.data).reshape(height, width)

    # ============================================================
    # TODO [D1 - Sensor Model]: Compute the distance transform
    #
    # Step 1: Create a boolean obstacle mask
    #   - In the PGM image: 0 = obstacle (black), 254 = free (white)
    #   - Mark cells as obstacles where pixel value < 50
    obs_mask = np.zeros((height, width))
    obs_mask = data < 50 
    #
    # Step 2: Use scipy.ndimage.distance_transform_edt()
    #   - Input: a boolean array where True = FREE cells
    #     (pass ~obstacle_mask, i.e., the inverse of obstacles)
    #   - This computes the distance (in pixels) from each free cell
    #     to the nearest obstacle cell
    dist_field = distance_transform_edt(~obs_mask)
    #
    # Step 3: Convert pixel distance to meters
    #   - Multiply by occupancy_grid.info.resolution (0.05 m/pixel)
    #
    dist_field = dist_field * occupancy_grid.info.resolution
    # ============================================================

    return dist_field


def sensor_model(pose, scan, likelihood_field, map_info,
                 z_hit=0.85, z_rand=0.15, sigma_hit=0.30, z_max=3.5, step=8):
    """
    Compute the log-likelihood of a LiDAR scan given a particle pose.

    Uses the likelihood field model from Probabilistic Robotics Ch. 5.
    For each LiDAR ray, we compute where the ray endpoint would land in the
    map, look up the distance to the nearest obstacle, and compute the
    probability using a Gaussian + uniform mixture.

    Args:
        pose: particle pose (x, y, theta) in map coordinates
        scan: LaserScan message
        likelihood_field: 2D array from compute_likelihood_field()
        map_info: map metadata (resolution, origin, etc.)
        z_hit: probability weight for Gaussian hit model
        z_rand: probability weight for uniform random model
        sigma_hit: std dev of Gaussian hit model (meters)
        z_max: maximum range of LiDAR
        step: only use every N-th ray (for speed)

    Returns:
        log-likelihood (float). Higher = better match.
    """
    x, y, theta = pose
    res = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    H, W = likelihood_field.shape

    ll = 0.0
    used = 0

    for i in range(0, len(scan.ranges), step):
        r = scan.ranges[i]

        if not np.isfinite(r) or r >= scan.range_max:
            continue

        # ============================================================
        # TODO [D1 - Sensor Model]: Compute ray endpoint and look up likelihood
        #
        # Step 1: Compute the angle of this ray in world frame
        angle = scan.angle_min + i * scan.angle_increment
        
        # Step 2: Compute where this ray hits in world coordinates
        x_end = x + r * math.cos(theta + angle)
        y_end = y + r * math.sin(theta + angle)
        
        # Step 3: Convert world coordinates to map grid coordinates
        mx = int((x_end - origin_x) / res)
        my = int((y_end - origin_y) / res)
        
        # Step 4: If (mx, my) is inside the map bounds:
        #   - Look up dist = likelihood_field[my, mx]
        #   - Compute p_hit = exp(-(dist^2) / (2 * sigma_hit^2))
        #   - Compute total probability: p = z_hit * p_hit + z_rand / max(1e-6, (scan.range_max - scan.range_min))
        #   - Add log(p) to the running total: ll += log(max(p, 1e-12))
        #   - Increment 'used' counter
        #   If outside bounds: ll += log(1e-4), increment 'used'

        if 0 <= mx < W and 0 <= my < H:
            dist = likelihood_field[my, mx]
            p_hit = math.exp(-(dist**2) / (2 * sigma_hit**2))
            p = z_hit * p_hit + z_rand / max(1e-6, (scan.range_max - scan.range_min))
            ll += math.log(max(p, 1e-12))
            used += 1
        else:
            ll += math.log(1e-4)
            used += 1
        # ============================================================

    return ll if used > 0 else -1e9


class SensorModelNode(Node):
    """Standalone node for testing the sensor model (D1).
    Subscribes to /map topic, computes and saves the likelihood field."""
    def __init__(self):
        super().__init__('sensor_model_node')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.subscription = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos)
        self.map_received = False

    def map_callback(self, msg):
        self.get_logger().info("Computing likelihood field...")
        self.map_info = msg.info
        dist_field = compute_likelihood_field(msg)
        import os
        from ament_index_python.packages import get_package_share_directory
        field_dir = os.path.join(get_package_share_directory('particle-filter-ros2'), 'field')
        os.makedirs(field_dir, exist_ok=True)
        np.save(os.path.join(field_dir, 'likelihood_field.npy'), dist_field)
        self.plot_and_save_field(dist_field, field_dir)
        self.get_logger().info("Likelihood field computed and saved.")
        self.map_received = True

    def plot_and_save_field(self, dist_field, field_dir):
        import os
        sigma = 0.2
        likelihood = np.exp(-0.5 * (dist_field / sigma)**2)
        likelihood = likelihood / np.max(likelihood)
        plt.imshow(likelihood, cmap='gray', origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(field_dir, 'likelihood_field.png'), dpi=300)
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = SensorModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
