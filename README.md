# NLP_Nav

## Installation

#### Install ROS2 Jazzy
Follow the instructions to install ROS2 Jazzy for Ubuntu 24.04 at: https://docs.ros.org/en/jazzy/Installation.html#. Please install ros-jazzy-desktop version.

#### Install the Turtlebot3 Simulation Package (for the Gazebo Simulation Environment)
Carefully follow instructions at: https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/ and install turtlebot3 simulation

#### Install NLP-Nav
In your ROS2 Workspace clone the NLP-Nav Repo:
```
git clone https://github.com/nick-Sutton/nlp_nav.git
```

## Building and Running the Navigation Stack
```
source /opt/ros/jazzy/setup.bash
colcon build --packages-select nlp_nav && source install/setup.bash
export TURTLEBOT3_MODEL=burger
```

#### Static Enviroment
The Static enviroment is a default house and includes no dynamic obstacles.
```
ros2 launch nlp_nav nlp_nav_launch.py moving_obstacles:=false

```

#### Dynamic Enviroment
The Dynamic enviroment adds moving obstacles to certain parts of the enviroment. These obstacles
are meant to represent people that might be navigating the robots enviroment.
```
ros2 launch nlp_nav nlp_nav_launch.py moving_obstacles:=true
```

#### In RViz:
Select the "2D Pose Estimation" Button in RViz and then select the
starting location of the robot. This will fix the Position misalignment.

Click the "+" sign in the RViz toolbar. Then under "rviz_default_plugins" select "SetGoal." This will add the
"2D Goal Pose" option to the toolbar.

## Navigating In RViz
To perform planning select the "2D Goal Pose" in the toolbar. Then select a position on the map to navigate to. The robot will navigate to the selected position using the local and global planners.

## Navigating with Natural Language
After running the Launch command run the following command in a seperate terminal:

```
ros2 run nlp_nav nlp.py
```

Input Commands into the terminal and watch the robot navigate to rooms based on the command you gave.









