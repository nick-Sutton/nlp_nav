#!/usr/bin/env python3
"""
Wrap the Nav2 controller's Twist into TwistStamped so it reaches
ros_gz_bridge, which subscribes TwistStamped on /cmd_vel.

Subscribes : /cmd_vel_raw  (geometry_msgs/Twist   — from controller_server)
Publishes  : /cmd_vel      (geometry_msgs/TwistStamped — for ros_gz_bridge)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self._pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Twist, '/cmd_vel_raw', self._cb, 10)
        self.get_logger().info('cmd_vel_bridge ready (Twist → TwistStamped)')

    def _cb(self, msg: Twist):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_footprint'
        out.twist = msg
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
