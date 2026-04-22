#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray, TransformStamped
import tf2_ros
import numpy as np
import math
import random
import yaml
import os
from types import SimpleNamespace
from motionModel import sample_motion_odometry, compute_odometry_deltas, quaternion_to_yaw
from sensorModel import sensor_model, compute_likelihood_field

class ParticleFilterNode(Node):
    def __init__(self, num_particles=1000):
        super().__init__('particle_filter_node')
        self.num_particles = num_particles
        self.particles = None
        self.weights = None
        self.last_odom = None
        self.map_info = None
        self.alphas = (0.05, 0.01, 0.05, 0.01)
        np.random.seed(42)
        random.seed(42)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/amcl_pose', 10)
        self.particle_pub = self.create_publisher(PoseArray, '/pf_particle_cloud', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info("Particle Filter node initialized")

        self.map_init()

        self.prev_odom_pose = None
        self.current_odom_pose = None
        self.u = None
        self.have_u = False

    def map_init(self):
        """Load map from yaml/pgm files and compute the likelihood field."""
        if self.map_info is not None:
            return

        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('nlp_nav')
        yaml_path = os.path.join(pkg_share, 'maps', 'map.yaml')

        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)

        self.map_info = SimpleNamespace()
        self.map_info.resolution = map_metadata['resolution']
        self.map_info.origin = SimpleNamespace()
        self.map_info.origin.position = SimpleNamespace(
            x=map_metadata['origin'][0],
            y=map_metadata['origin'][1]
        )
        from PIL import Image
        img_path = os.path.join(pkg_share, 'maps', map_metadata['image'])
        img = Image.open(img_path)
        self.map_info.width, self.map_info.height = img.size

        # Flip vertically: PGM origin is top-left, ROS map origin is bottom-left
        data = np.flipud(np.array(img))

        occupancy_grid = SimpleNamespace()
        occupancy_grid.info = self.map_info
        occupancy_grid.data = data.flatten().tolist()
        self.occupancy_grid = occupancy_grid

        self.likelihood_field = compute_likelihood_field(occupancy_grid)
        self.get_logger().info("Likelihood field computed from map")

        self.initialize_particles(occupancy_grid)
        self.get_logger().info("Particles initialized from saved map file")

    def initialize_particles(self, occupancy_grid):
        """
        Initialize particles randomly in free space on the map.

        Particles should be placed at random positions in cells that are
        free (pixel value > 250), with random orientations [-pi, pi].
        """
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        res = occupancy_grid.info.resolution
        origin_x = occupancy_grid.info.origin.position.x
        origin_y = occupancy_grid.info.origin.position.y
        data = np.array(occupancy_grid.data).reshape(height, width)

        # ============================================================
        # TODO [D2]: Initialize self.particles (shape: num_particles x 3)
        self.particles = np.zeros((self.num_particles, 3))
        
        # Step 1: Find free cells where data > 250
        #   free_indices = np.argwhere(data > 250)
        #   This gives an array of (row, col) = (y, x) indices
        free_indices = np.argwhere(data > 250)
        
        # Step 2: Randomly choose num_particles cells from free_indices
        chosen = free_indices[np.random.choice(len(free_indices), size=self.num_particles, replace=True)]

        # Step 3: Convert pixel coordinates to world coordinates:
        pixel_y = chosen[:, 0]
        pixel_x = chosen[:, 1]
        world_x = origin_x + pixel_x * res
        world_y = origin_y + pixel_y * res
        theta   = np.random.uniform(-np.pi, np.pi, self.num_particles)

        self.particles[:, 0] = world_x
        self.particles[:, 1] = world_y
        self.particles[:, 2] = theta
        
        # Step 4: Set self.weights to uniform (1/N for each particle)
        self.weights = np.ones(self.num_particles) / self.num_particles


        # ============================================================

        self.publish_particle_cloud()

    def odom_callback(self, msg):
        """Receive odometry and compute motion delta since last scan."""
        if self.particles is None or self.map_info is None:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = quaternion_to_yaw(msg.pose.pose.orientation)
        current_pose = (x, y, theta)

        if self.prev_odom_pose is None:
            self.prev_odom_pose = current_pose
            self.last_odom = msg
            return

        # Compute TOTAL delta since last scan (don't update prev_odom_pose here)
        self.u = self.compute_odometry_deltas_from_poses(self.prev_odom_pose, current_pose)
        self.current_odom_pose = current_pose
        self.have_u = True
        self.last_odom = msg

    def compute_odometry_deltas_from_poses(self, prev_pose, curr_pose):
        """Compute rot1-trans-rot2 from two (x, y, theta) poses."""
        x0, y0, t0 = prev_pose
        x1, y1, t1 = curr_pose

        dx = x1 - x0
        dy = y1 - y0
        trans = math.sqrt(dx**2 + dy**2)

        if trans < 1e-6:
            rot1 = 0.0
        else:
            rot1 = math.atan2(dy, dx) - t0
            rot1 = (rot1 + math.pi) % (2 * math.pi) - math.pi

        rot2 = t1 - t0 - rot1
        rot2 = (rot2 + math.pi) % (2 * math.pi) - math.pi

        return (rot1, trans, rot2)

    def scan_callback(self, msg):
        """
        Main particle filter loop. Called every time a new LiDAR scan arrives.
        Implements: Prediction -> Correction -> Resampling
        """
        if self.likelihood_field is None or self.particles is None:
            return

        # === Prediction step (motion model) ===
        if self.u is not None and self.have_u:
            rot1, trans, rot2 = self.u
            new_particles = []
            for p in self.particles:
                new_p = self.apply_motion_delta(p, self.u, self.alphas)
                new_particles.append(new_p)
            self.particles = np.array(new_particles)
            self.prev_odom_pose = self.current_odom_pose
            self.u = None
            self.have_u = False

        # ============================================================
        # TODO [D2]: Correction step - compute particle weights
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # For each particle, call sensor_model() to get a log-likelihood:
        log_weights = []
        for particle in self.particles:
            ll = sensor_model(particle, msg, self.likelihood_field, self.map_info)
            log_weights.append(ll)
        
        # Then convert log-likelihoods to weights:
        #   1. Find max log-likelihood: max_ll = np.max(log_weights)
        #   2. Subtract max for numerical stability: weights = exp(log_weights - max_ll)
        #   3. Add tiny constant to avoid zeros: weights += 1e-300
        #   4. Normalize so they sum to 1: weights /= sum(weights)
        log_weights = np.array(log_weights)
        max_ll = np.max(log_weights)
        weights = np.exp(log_weights - max_ll)
        weights += 1e-300
        weights /= sum(weights)
        
        # Store result in self.weights
        self.weights = weights
        # ============================================================

        # ============================================================
        # TODO [D2]: Resampling step
        #
        # Compute the effective particle count:
        N_eff = 1.0 / sum(weights**2)
        if N_eff < self.num_particles * 0.5:
            self.resample_particles()
        # ============================================================

        self.publish_particle_cloud()
        self.publish_estimated_pose()

    def apply_motion_delta(self, pose, u, alphas):
        """
        Apply a noisy motion delta to a single particle.
        Same algorithm as sample_motion_odometry, but takes (rot1, trans, rot2)
        directly instead of raw odometry messages.
        """
        x, y, theta = pose
        rot1, trans, rot2 = u
        a1, a2, a3, a4 = alphas

        # ============================================================
        # TODO [D2]: Apply noisy motion to this particle
        
        # Same math as sample_motion_odometry in motionModel.py:
        # 1. Compute noise std devs from alphas and deltas
        rot1_noise_std = math.sqrt(a1 * rot1**2 + a2 * trans**2)
        trans_noise_std = math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
        rot2_noise_std = math.sqrt(a1 * rot2**2 + a2 * trans**2)

        # 2. Add Gaussian noise to rot1, trans, rot2
        rot1_hat  = rot1  - np.random.normal(0, rot1_noise_std)
        trans_hat = trans - np.random.normal(0, trans_noise_std)
        rot2_hat  = rot2  - np.random.normal(0, rot2_noise_std)

        # 3. Compute new (x, y, theta) from noisy deltas
        
        x_new     = x + trans_hat * math.cos(theta + rot1_hat)
        y_new     = y + trans_hat * math.sin(theta + rot1_hat)
        theta_new = theta + rot1_hat + rot2_hat
        theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi
        # ============================================================

        return (x_new, y_new, theta_new)

    def resample_particles(self):
        """
        Resample particles using multinomial resampling (importance sampling).
        Particles with higher weights are more likely to be selected.
        """
        # ============================================================
        # TODO [D2]: Implement multinomial resampling
        #
        # Use np.random.choice to draw num_particles indices WITH replacement,
        # where the probability of each index equals its weight:
        #
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        
        # ============================================================

    def publish_estimated_pose(self):
        """Publish the weighted mean of all particles as the estimated robot pose."""
        x_mean = np.average(self.particles[:, 0], weights=self.weights)
        y_mean = np.average(self.particles[:, 1], weights=self.weights)
        # Use circular mean for angle (handles wraparound correctly)
        sin_sum = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_sum = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        theta_mean = math.atan2(sin_sum, cos_sum)

        x_var = float(np.average((self.particles[:, 0] - x_mean) ** 2, weights=self.weights))
        y_var = float(np.average((self.particles[:, 1] - y_mean) ** 2, weights=self.weights))
        theta_var = float(np.average(
            np.arctan2(
                np.sin(self.particles[:, 2] - theta_mean),
                np.cos(self.particles[:, 2] - theta_mean)
            ) ** 2,
            weights=self.weights
        ))

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(x_mean)
        msg.pose.pose.position.y = float(y_mean)
        msg.pose.pose.orientation.z = math.sin(theta_mean / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta_mean / 2.0)
        msg.pose.covariance[0]  = x_var
        msg.pose.covariance[7]  = y_var
        msg.pose.covariance[35] = theta_var
        self.pose_pub.publish(msg)
        self._broadcast_map_to_odom(x_mean, y_mean, theta_mean, msg.header.stamp)

    def _broadcast_map_to_odom(self, map_x, map_y, map_yaw, stamp):
        """Broadcast the dynamic map->odom TF derived from the particle filter estimate."""
        if self.current_odom_pose is None:
            return

        ox, oy, oth = self.current_odom_pose

        # T_map_odom = T_map_base * inv(T_odom_base)
        dth = map_yaw - oth
        tx = map_x - (ox * math.cos(dth) - oy * math.sin(dth))
        ty = map_y - (ox * math.sin(dth) + oy * math.cos(dth))

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(dth / 2.0)
        t.transform.rotation.w = math.cos(dth / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def publish_particle_cloud(self):
        """Publish all particles as a PoseArray for visualization in RViz."""
        if self.particles is None:
            return

        particle_array = PoseArray()
        particle_array.header.stamp = self.get_clock().now().to_msg()
        particle_array.header.frame_id = "map"

        for x, y, theta in self.particles:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            p.orientation.z = math.sin(theta / 2.0)
            p.orientation.w = math.cos(theta / 2.0)
            particle_array.poses.append(p)

        self.particle_pub.publish(particle_array)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
