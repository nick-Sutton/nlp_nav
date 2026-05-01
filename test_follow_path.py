#!/usr/bin/env python3
"""
Send a straight-line path to the DWA controller's /follow_path action server.
Usage:  python3 test_follow_path.py [distance_m]   (default: 2.0 m forward)
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def build_path(frame_id: str, distance: float, step: float = 0.4) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    n = max(1, int(distance / step))
    for i in range(1, n + 1):
        p = PoseStamped()
        p.header.frame_id = frame_id
        p.pose.position.x = i * step
        p.pose.position.y = 0.0
        p.pose.orientation.w = 1.0  # facing forward (yaw = 0)
        path.poses.append(p)
    return path


def main():
    rclpy.init()
    node = Node(
        'follow_path_test',
        parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
    )

    distance = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    node.get_logger().info(f'Target distance: {distance:.1f} m forward in odom frame')

    client = ActionClient(node, FollowPath, '/follow_path')
    node.get_logger().info('Waiting for /follow_path action server...')
    client.wait_for_server()
    node.get_logger().info('Server ready.')

    path = build_path('odom', distance)
    path.header.stamp = node.get_clock().now().to_msg()

    goal = FollowPath.Goal()
    goal.path = path
    goal.controller_id = 'FollowPath'
    goal.goal_checker_id = 'general_goal_checker'

    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        node.get_logger().error('Goal rejected by controller_server.')
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info('Goal accepted — robot should be moving.')
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)

    status = result_future.result().status
    node.get_logger().info(f'Finished. Action status code: {status}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
