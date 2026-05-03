#pragma once

#include <string>
#include <memory>
#include <vector>
#include <limits>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "tf2/utils.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace nlp_nav {

  struct TrajectoryPoint {
    double x, y, yaw;
  };

  struct Trajectory {
    double vx{0.0}, vth{0.0};
    std::vector<TrajectoryPoint> points;
  };

  class DWAController : public nav2_core::Controller {
    public:
      DWAController() = default;
      ~DWAController() override = default;

      void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

      void cleanup() override;
      void activate() override;
      void deactivate() override;

      geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * goal_checker) override;

      void setPlan(const nav_msgs::msg::Path & path) override;
      void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

    private:
      struct GoalPoint { double x{0.0}; double y{0.0}; };

      Trajectory simulateTrajectory(
        const geometry_msgs::msg::PoseStamped & start,
        double vx, double vth);

      bool isCollisionFree(const Trajectory & traj);

      Trajectory selectBest(
        const std::vector<Trajectory> & candidates,
        const GoalPoint & goal);

      double minObstacleDist(const Trajectory & traj);

      rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
      std::shared_ptr<tf2_ros::Buffer> tf_;
      std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
      nav2_costmap_2d::Costmap2D * costmap_{nullptr};
      std::string plugin_name_;

      // Velocity / acceleration limits
      double min_vel_x_{0.0};
      double max_vel_x_{0.5};
      double acc_lim_x_{2.5};
      double max_vel_theta_{1.0};
      double acc_lim_theta_{3.2};

      // Trajectory sampling resolution
      double vx_step_{0.05};
      double vth_step_{0.1};

      // Simulation parameters
      double sim_time_{1.5};
      double sim_granularity_{0.025};
      double dt_{0.1};

      // DWA scoring weights  (alpha*heading + beta*clearance + gamma*velocity)
      double alpha_{1.0};
      double beta_{0.2};
      double gamma_{0.2};

      GoalPoint getLookahead(
        const geometry_msgs::msg::PoseStamped & pose,
        const GoalPoint & fallback) const;

      // Goal position extracted from the latest global plan
      GoalPoint goal_;
      std::string goal_frame_id_{""};
      nav_msgs::msg::Path plan_;
      bool has_plan_{false};

      double lookahead_dist_{0.5};

      // Effective speed limit factor [0, 1]
      double speed_limit_{1.0};
    };

}  // namespace nlp_nav
