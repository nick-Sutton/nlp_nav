#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray, TransformStamped
from nav2_simple_commander import BasicNavigator



class GlobalPlanner(Node):
    
    def __init__(self):
        self.create_subscription(OccupancyGrid, "/map", self.callback, 10)
        
    def callback(self):
        pass
    
    

def main():
    rclpy.init()
    planner = GlobalPlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        # ctrl+c 
        print("Killing global planner")
    
if __name__ == "__main__":
    main()