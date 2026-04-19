#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
import math
import numpy as np

def inactive_check(odom_prev, odom_curr, trans_thresh=1e-4, rot_thresh=1e-3):
    x_prev = odom_prev.pose.pose.position.x
    y_prev = odom_prev.pose.pose.position.y
    x_curr = odom_curr.pose.pose.position.x
    y_curr = odom_curr.pose.pose.position.y

    yaw_prev = quaternion_to_yaw(odom_prev.pose.pose.orientation)
    yaw_curr = quaternion_to_yaw(odom_curr.pose.pose.orientation)

    dx = x_curr - x_prev
    dy = y_curr - y_prev
    dyaw = yaw_diff(yaw_prev, yaw_curr)

    if abs(dx) < trans_thresh and abs(dy) < trans_thresh and abs(dyaw) < rot_thresh:
        return True
    return False

def quaternion_to_yaw(q: Quaternion):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_diff(yaw1, yaw2):
    dyaw = yaw2 - yaw1
    while dyaw > math.pi:
        dyaw -= 2 * math.pi
    while dyaw < -math.pi:
        dyaw += 2 * math.pi
    return dyaw

def compute_odometry_deltas(odom_prev, odom_curr):
    x1 = odom_prev.pose.pose.position.x
    y1 = odom_prev.pose.pose.position.y
    x2 = odom_curr.pose.pose.position.x
    y2 = odom_curr.pose.pose.position.y

    yaw1 = quaternion_to_yaw(odom_prev.pose.pose.orientation)
    yaw2 = quaternion_to_yaw(odom_curr.pose.pose.orientation)

    # ============================================================
    # TODO [D1 - Motion Model]: Compute odometry deltas (rot1, trans, rot2)
    #
    # This implements the odometry motion model from Probabilistic Robotics.
    # Given two consecutive odometry readings, decompose the motion into:
    #   delta_rot1:  initial rotation to face the direction of translation
    #   delta_trans: straight-line distance traveled
    #   delta_rot2:  final rotation to match the new heading
    #
    # Step 1: Compute delta_x, delta_y, and delta_trans (Euclidean distance)
    delta_x = x2 - x1
    delta_y = y2 - y1
    delta_trans = np.sqrt(delta_x**2 + delta_y**2)
    
    # Step 2: Handle the special case when delta_trans < 1e-6 (pure rotation):
    if delta_trans < 1e-6:
        delta_rot1 = 0.0
        delta_rot2 = yaw_diff(yaw1, yaw2)
    else:
        # Step 3: Otherwise (translation + rotation):
        delta_rot1 = math.atan2(delta_y, delta_x) - yaw1
        
        # Normalize delta_rot1 to [-pi, pi]: (angle + pi) % (2*pi) - pi
        delta_rot1 = (delta_rot1 + np.pi) % (2*np.pi) - np.pi
        delta_rot2 = yaw2 - yaw1 - delta_rot1
    
        # Step 4: Normalize delta_rot2 to [-pi, pi]
        delta_rot2 = (delta_rot2 + np.pi) % (2*np.pi) - np.pi
    # ============================================================

    return delta_rot1, delta_trans, delta_rot2

def sample_motion_odometry(pose_prev, odom_prev, odom_curr, alphas):
    delta_rot1, delta_trans, delta_rot2 = compute_odometry_deltas(odom_prev, odom_curr)
    a1, a2, a3, a4 = alphas

    # ============================================================
    # TODO [D1 - Motion Model]: Sample a new pose with noise
    #
    # This adds Gaussian noise to the odometry deltas to model
    # motion uncertainty (Probabilistic Robotics Table 5.6).
    #
    
    x_prev, y_prev, theta_prev = pose_prev
    x_new = x_prev
    y_new = y_prev
    theta_new = theta_prev

    # Step 1: Compute standard deviations for each delta:
    std_rot1  = np.sqrt(a1 * delta_rot1**2 + a2 * delta_trans**2)
    std_trans = np.sqrt(a3 * delta_trans**2 + a4 * (delta_rot1**2 + delta_rot2**2))
    std_rot2  = np.sqrt(a1 * delta_rot2**2 + a2 * delta_trans**2)

    # Step 2: Subtract Gaussian noise from each delta:
    delta_rot1_hat  = delta_rot1  - np.random.normal(0.0, std_rot1)
    delta_trans_hat = delta_trans - np.random.normal(0.0, std_trans)
    delta_rot2_hat  = delta_rot2  - np.random.normal(0.0, std_rot2)

    # Step 3: Compute new pose from noisy deltas:
    x_new     = x_prev + delta_trans_hat * np.cos(theta_prev + delta_rot1_hat)
    y_new     = y_prev + delta_trans_hat * np.sin(theta_prev + delta_rot1_hat)
    theta_new = theta_prev + delta_rot1_hat + delta_rot2_hat

    # Normalize theta_new to [-pi, pi]
    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
    # ============================================================

    return (x_new, y_new, theta_new)

class MotionModelNode(Node):
    def __init__(self, testing=False):
        super().__init__('motion_model_node')
        self.testing = testing
        self.subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        if self.testing:
            self.pred_pub = self.create_publisher(PoseStamped, '/predicted_pose', 10)
        self.odom_prev = None
        self.odom_curr = None
        self.particle_pose = (0.0, 0.0, 0.0)
        self.alphas = (0.1, 0.01, 0.1, 0.01)
        self.theta_cumulative = 0.0
        self.last_yaw = None

    def odom_callback(self, msg):
        if self.odom_curr is None:
            self.odom_curr = msg
            self.last_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
            self.theta_cumulative = self.last_yaw
            return

        self.odom_prev = self.odom_curr
        self.odom_curr = msg

        yaw_curr = quaternion_to_yaw(self.odom_curr.pose.pose.orientation)
        dyaw = yaw_diff(self.last_yaw, yaw_curr)
        self.theta_cumulative += dyaw
        self.last_yaw = yaw_curr

        if inactive_check(self.odom_prev, self.odom_curr):
            print("Odom change too small; skipping motion update.")
            return

        dr1, dt, dr2 = compute_odometry_deltas(self.odom_prev, self.odom_curr)

        if dt < 1e-6:
            dr1 = 0.0
            dr2 = dyaw

        MAX_TRANS = 0.1
        MAX_ROT   = math.pi / 4
        if dt > MAX_TRANS or abs(dr1) > MAX_ROT or abs(dr2) > MAX_ROT:
            return

        self.get_logger().info(f"d_rot1={dr1:.3f}, d_trans={dt:.3f}, d_rot2={dr2:.3f}")

        # Sample new pose (prediction for one example particle)
        new_pose = sample_motion_odometry(
            self.particle_pose, self.odom_prev, self.odom_curr, self.alphas
        )
        self.particle_pose = new_pose
        x, y, theta = new_pose
        self.get_logger().info(f"Predicted pose: x={x:.3f}, y={y:.3f}, θ={theta:.3f}")
        
        # visualize in rviz2
        # run with 'ros2 run rviz2 rviz2' and add the topics /odom and /predicted_pose
        if self.testing:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.orientation.z = math.sin(theta/2.0)
            pose_msg.pose.orientation.w = math.cos(theta/2.0)
            self.pred_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionModelNode(testing=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
