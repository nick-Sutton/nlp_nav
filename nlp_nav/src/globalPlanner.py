#!/usr/bin/env python3
import heapq
import math
from types import SimpleNamespace
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from builtin_interfaces.msg import Time
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator

# Must match nav2_params.yaml values
ROBOT_RADIUS     = 0.22
INFLATION_RADIUS = 0.30
COST_SCALING     = 3.0
FREE_THRESH      = 100   # A* treats cells with cost >= this as blocked


class MapCostmap:
    """Inflated costmap built from a static OccupancyGrid, no lifecycle nodes required."""

    def __init__(self, grid: OccupancyGrid) -> None:
        info = grid.info
        res  = info.resolution
        w, h = info.width, info.height
        inf_cells = int(math.ceil(INFLATION_RADIUS / res)) + 1

        cost = bytearray(w * h)

        obstacle_positions: List[Tuple[int, int]] = []
        for idx, v in enumerate(grid.data):
            if v >= 65:
                cost[idx] = 254
                obstacle_positions.append((idx % w, idx // w))

        for ox, oy in obstacle_positions:
            for dy in range(-inf_cells, inf_cells + 1):
                ny = oy + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in range(-inf_cells, inf_cells + 1):
                    nx = ox + dx
                    if nx < 0 or nx >= w:
                        continue
                    dist_m = math.sqrt(dx * dx + dy * dy) * res
                    if dist_m > INFLATION_RADIUS:
                        continue
                    idx2 = ny * w + nx
                    if dist_m < ROBOT_RADIUS:
                        c = 253
                    else:
                        c = max(0, min(252, int(
                            253 * math.exp(-COST_SCALING * (dist_m - ROBOT_RADIUS)))))
                    if c > cost[idx2]:
                        cost[idx2] = c

        self.data = cost
        self.metadata = SimpleNamespace(
            size_x=w, size_y=h, resolution=res, origin=info.origin)


class LatestPoseSubscriber(Node):
    def __init__(self):
        super().__init__('latest_pose_subscriber')
        self.start_pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, '/estimated_pose', self._save, 1)

    def _save(self, pose: PoseStamped):
        self.start_pose = pose


class GlobalPlanner(Node):
    def __init__(self, pose_node: LatestPoseSubscriber):
        super().__init__('global_planner')
        self.navigator = BasicNavigator()
        self._pose_node = pose_node
        self._map: Optional[OccupancyGrid] = None
        self._costmap: Optional[MapCostmap] = None  # cached — rebuilt only on map change

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._costmap_pub = self.create_publisher(OccupancyGrid, '/global_costmap/costmap', map_qos)
        self.create_subscription(OccupancyGrid, '/map', self._save_map, map_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.get_logger().info('GlobalPlanner ready — waiting for goal on /goal_pose')
        self.pub = self.create_publisher(Path, "/plan", 10)

    def _save_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        self._costmap = None  # invalidate cache
        self.get_logger().info(
            f'Map received: {msg.info.width}×{msg.info.height} @ {msg.info.resolution:.3f} m/cell')

    def _get_costmap(self) -> Optional[MapCostmap]:
        if self._map is None:
            return None
        if self._costmap is None:
            self.get_logger().info('Building inflated costmap from map…')
            self._costmap = MapCostmap(self._map)
            self.get_logger().info('Costmap ready')
            self._publish_costmap(self._costmap)
        return self._costmap

    def _publish_costmap(self, cm: MapCostmap) -> None:
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = cm.metadata.resolution
        msg.info.width = cm.metadata.size_x
        msg.info.height = cm.metadata.size_y
        msg.info.origin = cm.metadata.origin
        # Convert 0-254 cost to 0-100 occupancy for RViz display
        msg.data = [min(100, int(c * 100 / 254)) for c in cm.data]
        self._costmap_pub.publish(msg)

    # ── coordinate conversion ───────────────────────────────────────────────

    def _pose_to_coord(self, pose: PoseStamped, cm: MapCostmap) -> Tuple[int, int]:
        res = cm.metadata.resolution
        ox  = cm.metadata.origin.position.x
        oy  = cm.metadata.origin.position.y
        mx = int((pose.pose.position.x - ox) / res)
        my = int((pose.pose.position.y - oy) / res)
        if 0 <= mx < cm.metadata.size_x and 0 <= my < cm.metadata.size_y:
            return mx, my
        self.get_logger().error(
            f'Pose ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}) outside costmap')
        return -1, -1

    def _coord_to_pose(self, coord: Tuple[int, int], cm: MapCostmap) -> PoseStamped:
        res = cm.metadata.resolution
        ox  = cm.metadata.origin.position.x
        oy  = cm.metadata.origin.position.y
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.header.stamp = Time()
        p.pose.position.x = coord[0] * res + ox + res / 2.0
        p.pose.position.y = coord[1] * res + oy + res / 2.0
        p.pose.orientation.w = 1.0
        return p

    # ── A* ──────────────────────────────────────────────────────────────────

    def _neighbors(self, coord: Tuple[int, int], cm: MapCostmap) -> List[Tuple[int, int]]:
        mx, my = coord
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = mx + dx, my + dy
                if 0 <= nx < cm.metadata.size_x and 0 <= ny < cm.metadata.size_y:
                    if cm.data[ny * cm.metadata.size_x + nx] < FREE_THRESH:
                        result.append((nx, ny))
        return result

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _a_star(self, start: PoseStamped, goal: PoseStamped, cm: MapCostmap) -> Path:
        start_c = self._pose_to_coord(start, cm)
        goal_c  = self._pose_to_coord(goal,  cm)

        if start_c == (-1, -1) or goal_c == (-1, -1):
            return Path()

        counter = 0
        heap = [(0.0, counter, start_c)]
        back_track: dict = {start_c: None}
        cost_so_far: dict = {start_c: 0.0}

        while heap:
            _, _, current = heapq.heappop(heap)
            if current == goal_c:
                break
            for nb in self._neighbors(current, cm):
                step = 1.414 if (nb[0] != current[0] and nb[1] != current[1]) else 1.0
                new_cost = cost_so_far[current] + step
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    counter += 1
                    heapq.heappush(
                        heap, (new_cost + self._heuristic(goal_c, nb), counter, nb))
                    back_track[nb] = current

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = Time()

        if goal_c not in back_track:
            self.get_logger().warn('A*: no path found to goal')
            return path

        poses = []
        node = goal_c
        while node is not None:
            poses.append(self._coord_to_pose(node, cm))
            node = back_track[node]
        poses.reverse()

        if poses:
            start.header.frame_id = 'map'
            start.header.stamp = Time()
            goal.header.frame_id = 'map'
            goal.header.stamp = Time()
            poses[0]  = start
            poses[-1] = goal

        for i in range(len(poses) - 1):
            dx = poses[i + 1].pose.position.x - poses[i].pose.position.x
            dy = poses[i + 1].pose.position.y - poses[i].pose.position.y
            yaw = math.atan2(dy, dx)
            poses[i].pose.orientation.x = 0.0
            poses[i].pose.orientation.y = 0.0
            poses[i].pose.orientation.z = math.sin(yaw / 2.0)
            poses[i].pose.orientation.w = math.cos(yaw / 2.0)

        path.poses = poses
        return path

    # ── goal callback ────────────────────────────────────────────────────────

    def _on_goal(self, goal: PoseStamped) -> None:
        start = self._pose_node.start_pose
        if start is None:
            self.get_logger().warn('No start pose yet — waiting for /estimated_pose')
            return

        cm = self._get_costmap()
        if cm is None:
            self.get_logger().warn('No map yet — waiting for /map')
            return

        self.get_logger().info(
            f'Planning: ({start.pose.position.x:.2f}, {start.pose.position.y:.2f}) -> '
            f'({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})')

        path = self._a_star(start, goal, cm)

        if not path.poses:
            self.get_logger().error('Empty path — not sending to controller')
            return

        self.get_logger().info(f'Path found: {len(path.poses)} waypoints — executing')
        self.pub.publish(path)
        self.navigator.followPath(path)


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    pose_node = LatestPoseSubscriber()
    planner   = GlobalPlanner(pose_node)
    executor.add_node(pose_node)
    executor.add_node(planner)
    try:
        executor.spin()
    except KeyboardInterrupt:
        print('Shutting down global planner')
    finally:
        planner.destroy_node()
        pose_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
