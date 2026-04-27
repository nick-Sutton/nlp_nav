#include "nlp_nav/dwa_controller.hpp"

namespace nlp_nav {

    void DWAController::configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
    {
        node_ = parent;
        plugin_name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros_->getCostmap();

        auto node = node_.lock();

        // Declare params
        auto declare = [&](const std::string & key, double default_val) {
            if (!node->has_parameter(plugin_name_ + "." + key)) {
            node->declare_parameter(plugin_name_ + "." + key, default_val);}
        };

        declare("min_vel_x",      min_vel_x_);
        declare("max_vel_x",      max_vel_x_);
        declare("acc_lim_x",      acc_lim_x_);
        declare("max_vel_theta",  max_vel_theta_);
        declare("acc_lim_theta",  acc_lim_theta_);
        declare("vx_step",        vx_step_);
        declare("vth_step",       vth_step_);
        declare("sim_time",       sim_time_);
        declare("sim_granularity", sim_granularity_);
        declare("dt",             dt_);
        declare("alpha",          alpha_);
        declare("beta",           beta_);
        declare("gamma",          gamma_);

        node->get_parameter(plugin_name_ + ".min_vel_x",       min_vel_x_);
        node->get_parameter(plugin_name_ + ".max_vel_x",       max_vel_x_);
        node->get_parameter(plugin_name_ + ".acc_lim_x",       acc_lim_x_);
        node->get_parameter(plugin_name_ + ".max_vel_theta",   max_vel_theta_);
        node->get_parameter(plugin_name_ + ".acc_lim_theta",   acc_lim_theta_);
        node->get_parameter(plugin_name_ + ".vx_step",         vx_step_);
        node->get_parameter(plugin_name_ + ".vth_step",        vth_step_);
        node->get_parameter(plugin_name_ + ".sim_time",        sim_time_);
        node->get_parameter(plugin_name_ + ".sim_granularity", sim_granularity_);
        node->get_parameter(plugin_name_ + ".dt",              dt_);
        node->get_parameter(plugin_name_ + ".alpha",           alpha_);
        node->get_parameter(plugin_name_ + ".beta",            beta_);
        node->get_parameter(plugin_name_ + ".gamma",           gamma_);

        RCLCPP_INFO(node->get_logger(), "DWAController configured as '%s'", plugin_name_.c_str());
    }

    void DWAController::cleanup()  {}
    void DWAController::activate() {}
    void DWAController::deactivate() {}

    void DWAController::setPlan(const nav_msgs::msg::Path & path) {
        if (!path.poses.empty()) {
            const auto & last = path.poses.back();
            goal_.x = last.pose.position.x;
            goal_.y = last.pose.position.y;
            goal_frame_id_ = path.header.frame_id;
            has_plan_ = true;
        }
    }

    void DWAController::setSpeedLimit(const double & speed_limit, const bool & percentage) {
        speed_limit_ = percentage ? (speed_limit / 100.0) : (speed_limit / max_vel_x_);
        speed_limit_ = std::max(0.0, std::min(1.0, speed_limit_));
    }

    geometry_msgs::msg::TwistStamped DWAController::computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * /*goal_checker*/) 
    {
        geometry_msgs::msg::TwistStamped cmd;
        cmd.header = pose.header;

        if (!has_plan_) {
            return cmd;  // zero velocity until a plan is received
        }

        const double eff_max_vx = max_vel_x_ * speed_limit_;

        // Build the dynamic window from current velocity and kinematic limits
        const double min_vx  = std::max(min_vel_x_,      velocity.linear.x  - acc_lim_x_     * dt_);
        const double max_vx  = std::min(eff_max_vx,       velocity.linear.x  + acc_lim_x_     * dt_);
        const double min_vth = std::max(-max_vel_theta_,  velocity.angular.z - acc_lim_theta_ * dt_);
        const double max_vth = std::min( max_vel_theta_,  velocity.angular.z + acc_lim_theta_ * dt_);

        // Sample (vx, vth) pairs and keep collision-free trajectories
        std::vector<Trajectory> candidates;
        for (double vx = min_vx; vx <= max_vx + 1e-9; vx += vx_step_) {
            for (double vth = min_vth; vth <= max_vth + 1e-9; vth += vth_step_) {

                Trajectory traj = simulateTrajectory(pose, vx, vth);
                if (isCollisionFree(traj)) {
                    candidates.push_back(traj);
                }
            }
        }

        if (candidates.empty()) {
            // No safe trajectory found — stop and rotate in place
            cmd.twist.angular.z = (max_vth > 0.0) ? max_vth : 0.0;
            return cmd;
        }

        // Transform the stored goal from its original frame (map) into the
        // frame the pose is expressed in (odom) so heading scoring is consistent.
        GoalPoint goal_local = goal_;
        if (!goal_frame_id_.empty() && goal_frame_id_ != pose.header.frame_id) {
            try {
                auto tf_stamped = tf_->lookupTransform(
                    pose.header.frame_id, goal_frame_id_, tf2::TimePointZero);
                geometry_msgs::msg::PoseStamped goal_msg, goal_transformed;
                goal_msg.header.frame_id = goal_frame_id_;
                goal_msg.pose.position.x = goal_.x;
                goal_msg.pose.position.y = goal_.y;
                goal_msg.pose.orientation.w = 1.0;
                tf2::doTransform(goal_msg, goal_transformed, tf_stamped);
                goal_local.x = goal_transformed.pose.position.x;
                goal_local.y = goal_transformed.pose.position.y;
            } catch (const tf2::TransformException & ex) {
                auto node = node_.lock();
                RCLCPP_WARN(node->get_logger(), "Goal transform failed: %s", ex.what());
                return cmd;
            }
        }

        Trajectory best = selectBest(candidates, goal_local);
        cmd.twist.linear.x  = best.vx;
        cmd.twist.angular.z = best.vth;

        auto node = node_.lock();
        RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
            "DWA: %zu candidates, sending vx=%.3f vth=%.3f",
            candidates.size(), best.vx, best.vth);

        return cmd;
    }

    Trajectory DWAController::simulateTrajectory(
        const geometry_msgs::msg::PoseStamped & start,
        double vx, double vth)
    {
        Trajectory traj;
        traj.vx  = vx;
        traj.vth = vth;

        double x   = start.pose.position.x;
        double y   = start.pose.position.y;
        double yaw = tf2::getYaw(start.pose.orientation);

        for (double t = 0.0; t <= sim_time_ + 1e-9; t += sim_granularity_) {
            x   += vx  * std::cos(yaw) * sim_granularity_;
            y   += vx  * std::sin(yaw) * sim_granularity_;
            yaw += vth * sim_granularity_;
            traj.points.push_back({x, y, yaw});
        }

        return traj;
    }

    bool DWAController::isCollisionFree(const Trajectory & traj)
    {
        for (const auto & pt : traj.points) {
            unsigned int mx, my;
            if (!costmap_->worldToMap(pt.x, pt.y, mx, my)) {
                return false;  // outside map bounds — treat as obstacle
            }
            uint8_t cost = costmap_->getCost(mx, my);
            // NO_INFORMATION (255) is treated as free — only reject known lethal cells
            if (cost != nav2_costmap_2d::NO_INFORMATION &&
                cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
                return false;
            }
        }
        return true;
    }

    double DWAController::minObstacleDist(const Trajectory & traj)
    {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto & pt : traj.points) {
            unsigned int mx, my;
            if (!costmap_->worldToMap(pt.x, pt.y, mx, my)) { continue; }

            // Normalise cost [0, LETHAL) → distance [1, 0)
            double dist = 1.0 - static_cast<double>(costmap_->getCost(mx, my)) /
                                nav2_costmap_2d::LETHAL_OBSTACLE;
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        return (min_dist == std::numeric_limits<double>::max()) ? 0.0 : min_dist;
    }

    Trajectory DWAController::selectBest(
        const std::vector<Trajectory> & candidates,
        const GoalPoint & goal)
    {
        Trajectory best;
        double best_score = -std::numeric_limits<double>::infinity();

        for (const auto & traj : candidates) {
            if (traj.points.empty()) { continue; }
            const auto & end = traj.points.back();

            // Heading: angle difference between trajectory end and goal direction.
            // Normalize to [-π, π] to keep the score in [0, π].
            const double goal_angle = std::atan2(goal.y - end.y, goal.x - end.x);
            double heading_diff = end.yaw - goal_angle;
            heading_diff = std::atan2(std::sin(heading_diff), std::cos(heading_diff));
            const double heading = M_PI - std::abs(heading_diff);

            // Clearance: normalised min obstacle distance along trajectory
            const double clearance = minObstacleDist(traj);

            // DWA objective: G(v,ω) = α·heading + β·clearance + γ·velocity
            const double score = alpha_ * heading + beta_ * clearance + gamma_ * traj.vx;
            if (score > best_score) {
                best_score = score;
                best = traj;
            }
        }
        return best;
    }

}  // namespace nlp_nav

PLUGINLIB_EXPORT_CLASS(nlp_nav::DWAController, nav2_core::Controller)
