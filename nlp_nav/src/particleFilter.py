#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, PoseWithCovarianceStamped, TransformStamped
import tf2_ros
import numpy as np
import math
import random
import yaml
import os
from types import SimpleNamespace
from motionModel import sample_motion_odometry, compute_odometry_deltas, quaternion_to_yaw
from sensorModel import compute_likelihood_field

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
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 10)

        self.pose_pub = self.create_publisher(PoseStamped, '/estimated_pose', 10)
        self.particle_pub = self.create_publisher(PoseArray, '/pf_particle_cloud', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Latest localization estimate (map frame) — updated by scan callback
        self.estimated_x = 0.0
        self.estimated_y = 0.0
        self.estimated_theta = 0.0

        # Monotonicity guard: last TF stamp in nanoseconds
        self._last_tf_stamp_ns = 0

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
        free_indices = np.argwhere(data > 250)

        # Step 2: Randomly choose num_particles cells from free_indices
        chosen = free_indices[np.random.choice(len(free_indices), size=self.num_particles, replace=True)]

        # Step 3: Convert pixel coordinates to world coordinates
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

    def _broadcast_map_to_odom(self, map_x, map_y, map_yaw, stamp):
        """Compute and broadcast the map→odom transform."""
        ox, oy, oth = self.current_odom_pose
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

    def initialpose_callback(self, msg):
        """Reinitialize particles around the pose set via RViz '2D Pose Estimate'."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        std_xy = 0.5
        std_th = 0.2

        self.particles[:, 0] = np.random.normal(x, std_xy, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, std_xy, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta, std_th, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.get_logger().info(
            f"Particles reset to ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}deg)")
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

        # === Prediction step (motion model) — vectorised ===
        if self.u is not None and self.have_u:
            self._batch_motion_update(self.u, self.alphas)
            self.prev_odom_pose = self.current_odom_pose
            self.u = None
            self.have_u = False

        # ============================================================
        # TODO [D2]: Correction step - compute particle weights

        # Vectorised sensor model — evaluates all particles at once.
        log_weights = self._batch_sensor_model(msg)
        max_ll = np.max(log_weights)
        weights = np.exp(log_weights - max_ll)
        weights += 1e-300
        w_sum = weights.sum()
        if w_sum > 0 and np.isfinite(w_sum):
            weights /= w_sum
        else:
            weights = np.ones(self.num_particles) / self.num_particles

        self.weights = weights
        # ============================================================

        # ============================================================
        # TODO [D2]: Resampling step
        N_eff = 1.0 / np.sum(weights**2)
        if N_eff < self.num_particles * 0.5:
            self.resample_particles()
        # ============================================================

        self.publish_particle_cloud()
        self.publish_estimated_pose()

        # Broadcast map→odom using the scan's own timestamp (monotonically
        # increasing from Gazebo, so TF_OLD_DATA cannot occur).
        if self.current_odom_pose is not None:
            scan_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if scan_ns > self._last_tf_stamp_ns:
                self._last_tf_stamp_ns = scan_ns
                self._broadcast_map_to_odom(
                    self.estimated_x, self.estimated_y, self.estimated_theta,
                    msg.header.stamp)

    def apply_motion_delta(self, pose, u, alphas):
        x, y, theta = pose
        rot1, trans, rot2 = u
        a1, a2, a3, a4 = alphas

        rot1_noise_std = math.sqrt(a1 * rot1**2 + a2 * trans**2)
        trans_noise_std = math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
        rot2_noise_std = math.sqrt(a1 * rot2**2 + a2 * trans**2)

        rot1_hat  = rot1  - np.random.normal(0, rot1_noise_std)
        trans_hat = trans - np.random.normal(0, trans_noise_std)
        rot2_hat  = rot2  - np.random.normal(0, rot2_noise_std)

        x_new     = x + trans_hat * math.cos(theta + rot1_hat)
        y_new     = y + trans_hat * math.sin(theta + rot1_hat)
        theta_new = theta + rot1_hat + rot2_hat
        theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi

        return (x_new, y_new, theta_new)

    def _batch_motion_update(self, u, alphas):
        rot1, trans, rot2 = u
        a1, a2, a3, a4 = alphas
        N = self.num_particles
        std_rot1  = math.sqrt(a1 * rot1**2 + a2 * trans**2)
        std_trans = math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
        std_rot2  = math.sqrt(a1 * rot2**2 + a2 * trans**2)
        rot1_hat  = rot1  - np.random.normal(0, std_rot1,  N)
        trans_hat = trans - np.random.normal(0, std_trans, N)
        rot2_hat  = rot2  - np.random.normal(0, std_rot2,  N)
        x  = self.particles[:, 0]
        y  = self.particles[:, 1]
        th = self.particles[:, 2]
        self.particles[:, 0] = x  + trans_hat * np.cos(th + rot1_hat)
        self.particles[:, 1] = y  + trans_hat * np.sin(th + rot1_hat)
        th_new = th + rot1_hat + rot2_hat
        self.particles[:, 2] = np.arctan2(np.sin(th_new), np.cos(th_new))

    def _batch_sensor_model(self, scan, z_hit=0.85, z_rand=0.15, sigma_hit=0.30, step=8):
        res = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        H, W = self.likelihood_field.shape
        range_width = max(1e-6, scan.range_max - scan.range_min)
        ranges = np.array(scan.ranges)
        all_idx = np.arange(len(ranges))[::step]
        valid = np.isfinite(ranges[all_idx]) & (ranges[all_idx] < scan.range_max)
        idx = all_idx[valid]
        if len(idx) == 0:
            return np.zeros(self.num_particles)
        r = ranges[idx]
        ang = scan.angle_min + idx * scan.angle_increment
        x  = self.particles[:, 0, None]
        y  = self.particles[:, 1, None]
        th = self.particles[:, 2, None]
        xe = x + r * np.cos(th + ang)
        ye = y + r * np.sin(th + ang)
        mx = ((xe - origin_x) / res).astype(int)
        my = ((ye - origin_y) / res).astype(int)
        ok = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
        mxs = np.clip(mx, 0, W - 1)
        mys = np.clip(my, 0, H - 1)
        dist = self.likelihood_field[mys, mxs]
        p_hit = np.exp(-(dist**2) / (2 * sigma_hit**2))
        p = np.where(ok, z_hit * p_hit + z_rand / range_width, 1e-4)
        return np.log(np.maximum(p, 1e-12)).sum(axis=1)

    def resample_particles(self):
        """Multinomial resampling."""
        # ============================================================
        # TODO [D2]: Implement multinomial resampling
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        # ============================================================

    def publish_estimated_pose(self):
        """Publish the weighted mean of all particles as the estimated robot pose."""
        x_mean = np.average(self.particles[:, 0], weights=self.weights)
        y_mean = np.average(self.particles[:, 1], weights=self.weights)
        sin_sum = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_sum = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        theta_mean = math.atan2(sin_sum, cos_sum)

        # Store for TF timer
        self.estimated_x = float(x_mean)
        self.estimated_y = float(y_mean)
        self.estimated_theta = theta_mean

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = self.estimated_x
        pose_msg.pose.position.y = self.estimated_y
        pose_msg.pose.orientation.z = math.sin(theta_mean / 2.0)
        pose_msg.pose.orientation.w = math.cos(theta_mean / 2.0)
        self.pose_pub.publish(pose_msg)

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
