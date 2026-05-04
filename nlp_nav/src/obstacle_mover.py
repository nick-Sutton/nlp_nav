#!/usr/bin/env python3

import math
import threading
import subprocess
import rclpy
from rclpy.node import Node

# Try gz.transport Python bindings for smooth, zero-overhead pose updates.
# Gz Harmonic ships these as gz-transport13-python (python3-gz-transport13).
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_pb2 import Pose as GzPose
    from gz.msgs10.boolean_pb2 import Boolean as GzBool
    _GZ_TRANSPORT = True
except ImportError:
    _GZ_TRANSPORT = False

# (model_name, x1, y1, x2, y2, period_seconds, phase_offset_seconds)
OBSTACLES = [
    ('obs_0',  0.0,  1.0,  0.0, -0.5, 20.0, 0.0),                      # red
    ('obs_1',  -1.757,    4.36032,  -1.757,     2.86032,  16.0, 16.0),  # green
    ('obs_2', -2.02724, 3.91556,  -4.02724, 3.91556,  16.0, 16.0),  # blue
    # ('obs_2', -4.02724, 3.91556,  -4.02724,     0.91556,   8.0, 8.0),   # blue
]
Z = 0.6   # centre of 1.2 m capsule (radius 0.3 + length 0.6) sitting on ground


class ObstacleMover(Node):
    def __init__(self):
        super().__init__('obstacle_mover')
        self._t0 = self.get_clock().now().nanoseconds / 1e9

        if _GZ_TRANSPORT:
            self._gz = GzNode()
            self.get_logger().info('ObstacleMover: using gz.transport (smooth)')
            hz = 20.0
        else:
            self.get_logger().info('ObstacleMover: gz.transport unavailable, using threaded subprocess')
            hz = 10.0

        self.create_timer(1.0 / hz, self._tick)

    def _tick(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self._t0
        for name, x1, y1, x2, y2, period, offset in OBSTACLES:
            alpha = (math.sin(2 * math.pi * (t + offset) / period) + 1.0) / 2.0
            x = x1 + (x2 - x1) * alpha
            y = y1 + (y2 - y1) * alpha
            if _GZ_TRANSPORT:
                self._set_pose_transport(name, x, y)
            else:
                threading.Thread(
                    target=self._set_pose_subprocess,
                    args=(name, x, y),
                    daemon=True,
                ).start()

    def _set_pose_transport(self, name, x, y):
        pose = GzPose()
        pose.name = name
        pose.position.x = x
        pose.position.y = y
        pose.position.z = Z
        pose.orientation.w = 1.0
        self._gz.request(
            '/world/default/set_pose', pose, GzPose, GzBool, 100
        )

    def _set_pose_subprocess(self, name, x, y):
        req = (
            f'name: "{name}" '
            f'position {{ x: {x:.3f} y: {y:.3f} z: {Z} }} '
            f'orientation {{ x: 0 y: 0 z: 0 w: 1 }}'
        )
        subprocess.run(
            [
                'gz', 'service',
                '-s', '/world/default/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '500',
                '--req', req,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
