#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from nav2_msgs.msg import Costmap
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray, TransformStamped
import nav2_simple_commander
from nav2_simple_commander.robot_navigator import BasicNavigator
import heapq
from typing import List, Tuple



class LatestPoseSubscriber(Node):
    def __init__(self):
        """Gets the starting position of the robot
        """
        super().__init__('latest_pose_subscriber')
        self.start_pose = None
        self.subscription = self.create_subscription(
            Pose,                # Message type
            "/pose",             # Topic name
            self.save_start_pose,
            1                    # QoS queue size (1 = only latest kept)
        )
    
    def save_start_pose(self, pose:Pose):
        """Saves it

        Args:
            pose (Pose): the position to save
        """
        self.start_pose = pose
        

class GlobalPlanner(Node):
    
    def __init__(self, start_pose: Pose):
        """Runs A* global planner. 
           Gets the goal position from the `/goal_pose` topic

        Args:
            start_pose (Pose): the starting position of the robot
        """
        super().__init__('global_planner')
        self.start_pose = start_pose
        self.navigator = BasicNavigator()
        self.create_subscription(Pose, "/goal_pose", self.callback, 10)
        
    def __pose_to_coord(self, pose:Pose, cost_map:Costmap) -> Tuple[int, int]:
        """Convert world Pose to map coordinates

        Args:
            pose (Pose): Position of robot
            cost_map (Costmap): The costmap whose coordinates to convert to

        Returns:
            Tuple[int, int]: The map coordinates corresponding to the given world pose
        """
        resolution = cost_map.metadata.resolution
        origin_x = cost_map.metadata.origin.position.x
        origin_y = cost_map.metadata.origin.position.y
        
        mx = int((pose.position.x - origin_x) // resolution) # Integer division because map coordinates are full integers
        my = int((pose.position.y - origin_y) // resolution)
        
        if 0 <= mx < cost_map.metadata.size_x and 0 <= my < cost_map.metadata.size_y:
            return mx, my
        else:
            self.get_logger().error(f"Pose ({pose.position.x}, {pose.position.y}) is outside costmap bounds")
            return (-1, -1) # Invalid
        
    def __coord_to_pose(self, coord:Tuple[int, int], cost_map:Costmap) -> Pose:
        """Convert map coordinates to world Pose

        Args:
            coord (Tuple[int, int]): Map coordinates (mx, my)
            cost_map (Costmap): The costmap whose coordinates to convert from
            cost_map (Costmap): The costmap whose coordinates to convert from

        Returns:
            Pose: The world pose corresponding to the given map coordinates
        """
        resolution = cost_map.metadata.resolution
        origin_x = cost_map.metadata.origin.position.x
        origin_y = cost_map.metadata.origin.position.y
        
        mx, my = coord
        
        pose = Pose()
        pose.position.x = mx * resolution + origin_x + resolution / 2.0 # scale to resolution, offset by origin, center inside cell
        pose.position.y = my * resolution + origin_y + resolution / 2.0 # " "
        
        return pose
    
    def __get_neighbors(self, pose:Pose, cost_map:Costmap) -> List[Tuple[int, int]]:
        """Get valid neighboring map coordinates (8-connected)

        Args:
            mx (int): Map x coordinate
            my (int): Map y coordinate
            cost_map (Costmap): The costmap to check for valid neighbors

        Returns:
            List[Tuple[int, int]]: List of valid neighboring map coordinates
        """
        mx, my = self.__pose_to_coord(pose, cost_map)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue # Skip current cell
                nx, ny = mx + dx, my + dy
                if 0 <= nx < cost_map.metadata.size_x and 0 <= ny < cost_map.metadata.size_y:
                    index = ny * cost_map.metadata.size_x + nx
                    if cost_map.data[index] < 230: # Not an obstacle -- 253 = Inscribed (robot center would cause collision), 254 = Lethal, 255 = No info
                        neighbors.append((nx, ny))
        return neighbors
    
    def __heuristic(self, goal:Tuple[int, int], next:Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)

        Args:
            goal (Tuple[int, int]): Goal map coordinate
            next (Tuple[int, int]): Next map coordinate
        """
        x1, y1 = goal
        x2, y2 = next
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def __build_path(self, start:Pose, goal:Pose, back_track:dict) -> Path:
        """Builds a path from the back_track dictionary

        Args:
            back_track (dict): Dictionary mapping each map coordinate to its previous coordinate

        Returns:
            Path: The path constructed from the back_track dictionary
        """
        path = Path()
        path.poses.append(goal)
        
    
    def __a_star(self, start:Pose, goal:Pose, path:Path, cost_map:Costmap):
        """Performs A* planning

        Args:
            start (Pose): _description_
            goal (Pose): _description_
            current (Pose): _description_
            path (Path): _description_
            cost_map (Costmap): _description_
        """
        # Initialize our priority queue of all possible positions & cost values
        entry_num: int = 0 # Order of entry into prio queue
        prio_queue: List[Tuple[float, int, Pose]] = [] # (priority (cost), entry number, Position)
        heapq.heappush(prio_queue, (0.0, entry_num, start))
        
        back_track = dict() # Key is a map coordinate, value is the coordinate used to get there
        cost_so_far = dict()
        
        back_track[start] = None
        cost_so_far[start] = 0
        
        while not len(prio_queue) == 0:
            # Set the current position to the last element
            _, _, current = prio_queue[0]
            # If we're there, stop
            if current == goal:
                break
            
            # Build the back tracking path
            for next in self.__get_neighbors(current, cost_map):
                new_cost = cost_so_far[current] + 1
                if self.__coord_to_pose(next, cost_map) == goal: # If a neighbor is the goal, stop & move on
                    back_track[goal] = current
                    break
                    
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    prio = new_cost + self.__heuristic(self.__pose_to_coord(goal, cost_map), next)
                    entry_num += 1
                    heapq.heappush(prio_queue, (prio, entry_num, self.__coord_to_pose(next, cost_map)))
                    back_track[next] = self.__coord_to_pose(next, cost_map)
            
            
            
            
            
        
        
    def callback(self, goal:Pose) -> None:
        """Starts A* planning

        Args:
            goal (Pose): _description_
        """
        cost_map = self.navigator.getGlobalCostmap()
        goal_x, goal_y, = goal.position.x, goal.position.y

        current_x = self.start_pose.position.x
        current_y = self.start_pose.position.y
        self.get_logger().info(f"Start pose: {current_x}, {current_y}")
        self.get_logger().info(f"Goal pose: {goal_x}, {goal_y}")
        
        path = Path()

        self.__a_star(self.start_pose, goal,  path, cost_map)
        
        self.navigator.followPath(path)
        
        
        
        
    
    

def main():
    rclpy.init()
    latest_pose = LatestPoseSubscriber()
    planner = None
    try:
        while latest_pose.start_pose is None:
            rclpy.spin_once(latest_pose, timeout_sec=0.1)

        planner = GlobalPlanner(latest_pose.start_pose)
        latest_pose.destroy_node()
        rclpy.spin(planner)
    except KeyboardInterrupt:
        # ctrl+c 
        print("Killing global planner")
    finally:
        if planner is not None:
            planner.destroy_node()
        if latest_pose is not None:
            latest_pose.destroy_node()
        rclpy.shutdown()
    
if __name__ == "__main__":
    main()