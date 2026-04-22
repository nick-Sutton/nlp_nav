#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from nav2_msgs.msg import Costmap
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray, TransformStamped
import nav2_simple_commander
from nav2_simple_commander.robot_navigator import BasicNavigator


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
    
    def __a_star(self, start:Pose, goal:Pose, current:Pose, path:Path, cost_map:Costmap):
        """Performs A* planning

        Args:
            start (Pose): _description_
            goal (Pose): _description_
            current (Pose): _description_
            path (Path): _description_
            cost_map (Costmap): _description_
        """
        pass
        
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

        self.__a_star(self.start_pose, goal, self.start_pose, path, cost_map)
        
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