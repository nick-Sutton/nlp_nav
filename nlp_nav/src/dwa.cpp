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
                node->declare_parameter(plugin_name_ + "." + key, default_val);
            }
        };

        declare("min_vel_x",       min_vel_x_);
        declare("max_vel_x",       max_vel_x_);
        declare("acc_lim_x",       acc_lim_x_);
        declare("max_vel_theta",   max_vel_theta_);
        declare("acc_lim_theta",   acc_lim_theta_);
        declare("vx_step",         vx_step_);
        declare("vth_step",        vth_step_);
        declare("sim_time",        sim_time_);
        declare("sim_granularity", sim_granularity_);
        declare("dt",              dt_);
        declare("alpha",           alpha_);
        declare("beta",            beta_);
        declare("gamma",           gamma_);
        declare("lookahead_dist",  lookahead_dist_);

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
        node->get_parameter(plugin_name_ + ".lookahead_dist",  lookahead_dist_);

        RCLCPP_INFO(node->get_logger(), "DWAController configured as '%s'", plugin_name_.c_str());
    }

    void DWAController::cleanup()   {}
    void DWAController::activate()  {}
    void DWAController::deactivate() {}

    void DWAController::setPlan(const nav_msgs::msg::Path & path) {
        if (!path.poses.empty()) {
            plan_ = path;
            const auto & last = path.poses.back();
            goal_.x = last.pose.position.x;
            goal_.y = last.pose.position.y;
            goal_frame_id_ = path.header.frame_id;
            has_plan_ = true;
            // Reset stall counter on new plan so recovery doesn't trigger immediately
            consecutive_stall_ = 0;
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

        GoalPoint target = getLookahead(pose, goal_local);

        // Pre-compute forward candidates before stall logic.
        std::vector<Trajectory> fwd;
        for (const auto & t : candidates) {
            if (t.vx > 0.0) fwd.push_back(t);
        }

        // Only count a stall when no forward path exists at all.
        if (fwd.empty()) {
            consecutive_stall_++;
        } else {
            consecutive_stall_ = 0;
        }

        if (consecutive_stall_ > 10) {
            // back up straight to create room
            if (consecutive_stall_ <= 20) {
                cmd.twist.linear.x  = -max_vel_x_ * 0.4;
                cmd.twist.angular.z = 0.0;
                consecutive_stall_++;
                auto node = node_.lock();
                RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 500,
                    "DWA stall recovery PHASE 1 (backup), count=%d", consecutive_stall_);
            }
            // rotate toward the next lookahead waypoint
            else if (consecutive_stall_ <= 30) {
                const double robot_yaw = tf2::getYaw(pose.pose.orientation);
                const double angle_to_target = std::atan2(
                    target.y - pose.pose.position.y,
                    target.x - pose.pose.position.x);
                const double err = std::atan2(
                    std::sin(angle_to_target - robot_yaw),
                    std::cos(angle_to_target - robot_yaw));
                cmd.twist.angular.z = (err >= 0.0) ? max_vel_theta_ : -max_vel_theta_;
                consecutive_stall_++;
                auto node = node_.lock();
                RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 500,
                    "DWA stall recovery PHASE 2 (rotate), count=%d", consecutive_stall_);
            }
            // recovery cycle complete, reset and try DWA normally again
            else {
                consecutive_stall_ = 0;
            }
            return cmd;
        }

        if (candidates.empty()) {
            // Rotate toward the next waypoint to try to open up a path
            const double robot_yaw = tf2::getYaw(pose.pose.orientation);
            const double angle_to_target = std::atan2(
                target.y - pose.pose.position.y,
                target.x - pose.pose.position.x);
            const double err = std::atan2(
                std::sin(angle_to_target - robot_yaw),
                std::cos(angle_to_target - robot_yaw));
            cmd.twist.angular.z = (err >= 0.0) ? max_vel_theta_ : -max_vel_theta_;
            return cmd;
        }

        // Normal DWA: pick the best scored trajectory
        GoalPoint robot_pos{pose.pose.position.x, pose.pose.position.y};
        Trajectory best = selectBest(candidates, target, robot_pos);
        cmd.twist.linear.x  = best.vx;
        cmd.twist.angular.z = best.vth;

        auto node = node_.lock();
        RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
            "DWA: %zu candidates, sending vx=%.3f vth=%.3f stall=%d",
            candidates.size(), cmd.twist.linear.x, cmd.twist.angular.z, consecutive_stall_);

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
                return false;  // outside map bounds
            }
            uint8_t cost = costmap_->getCost(mx, my);
            // Reject INSCRIBED_INFLATED (253) and above
            if (cost != nav2_costmap_2d::NO_INFORMATION &&
                cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
                return false;
            }
        }
        return true;
    }

    double DWAController::minObstacleDist(const Trajectory & traj)
    {
        double sum   = 0.0;
        int    count = 0;
        for (const auto & pt : traj.points) {
            unsigned int mx, my;
            if (!costmap_->worldToMap(pt.x, pt.y, mx, my)) { continue; }
            sum += 1.0 - static_cast<double>(costmap_->getCost(mx, my)) /
                         nav2_costmap_2d::LETHAL_OBSTACLE;
            ++count;
        }
        return (count == 0) ? 0.0 : sum / count;
    }

    DWAController::GoalPoint DWAController::getLookahead(
        const geometry_msgs::msg::PoseStamped & pose,
        const GoalPoint & fallback) const
    {
        if (plan_.poses.empty()) return fallback;

        const double rx = pose.pose.position.x;
        const double ry = pose.pose.position.y;
        const std::string & plan_frame  = plan_.header.frame_id;
        const std::string & robot_frame = pose.header.frame_id;

        double tx = 0.0, ty = 0.0, cos_t = 1.0, sin_t = 0.0;
        if (plan_frame != robot_frame) {
            try {
                auto tf_stamped = tf_->lookupTransform(
                    robot_frame, plan_frame, tf2::TimePointZero);
                tx    = tf_stamped.transform.translation.x;
                ty    = tf_stamped.transform.translation.y;
                double yaw = tf2::getYaw(tf_stamped.transform.rotation);
                cos_t = std::cos(yaw);
                sin_t = std::sin(yaw);
            } catch (const tf2::TransformException &) {
                return fallback;
            }
        }

        // Transform all path points into robot frame
        std::vector<std::pair<double, double>> pts;
        pts.reserve(plan_.poses.size());
        for (const auto & pp : plan_.poses) {
            double px = cos_t * pp.pose.position.x - sin_t * pp.pose.position.y + tx;
            double py = sin_t * pp.pose.position.x + cos_t * pp.pose.position.y + ty;
            pts.push_back({px, py});
        }

        // Find the path point closest to the robot 
        size_t closest_idx = 0;
        double min_d = std::numeric_limits<double>::max();
        for (size_t i = 0; i < pts.size(); ++i) {
            double d = std::hypot(pts[i].first - rx, pts[i].second - ry);
            if (d < min_d) { min_d = d; closest_idx = i; }
        }

        // From the closest point onward, return the first point that is:
        //   (a) at least lookahead_dist_ from the robot, AND
        //   (b) not inside an obstacle (cost < INSCRIBED_INFLATED_OBSTACLE)
        // This prevents the robot from being pulled toward a wall by a path
        // that passes through inflated obstacle space.
        for (size_t i = closest_idx; i < pts.size(); ++i) {
            double d = std::hypot(pts[i].first - rx, pts[i].second - ry);
            if (d >= lookahead_dist_) {
                // Verify the candidate target is collision-free on the costmap
                unsigned int mx, my;
                if (costmap_->worldToMap(pts[i].first, pts[i].second, mx, my)) {
                    uint8_t cost = costmap_->getCost(mx, my);
                    if (cost == nav2_costmap_2d::NO_INFORMATION ||
                        cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
                        return GoalPoint{pts[i].first, pts[i].second};
                    }
                    // Point is in obstacle space — keep searching further along path
                }
            }
        }

        // All remaining points are within lookahead or in obstacles — steer to final goal
        return fallback;
    }

    Trajectory DWAController::selectBest(
        const std::vector<Trajectory> & candidates,
        const GoalPoint & goal,
        const GoalPoint & robot_pos)
    {
        Trajectory best;
        double best_score = -std::numeric_limits<double>::infinity();

        // Compute the goal direction once from the robot's current position.
        const double goal_angle = std::atan2(
            goal.y - robot_pos.y,
            goal.x - robot_pos.x);

        for (const auto & traj : candidates) {
            if (traj.points.empty()) { continue; }
            const auto & end = traj.points.back();

            // Heading: how well the trajectory endpoint's yaw aligns with the
            // direction from the robot's current position toward the target.
            // Normalised to [0, 1] — higher is better.
            double heading_diff = end.yaw - goal_angle;
            heading_diff = std::atan2(std::sin(heading_diff), std::cos(heading_diff));
            const double heading = (M_PI - std::abs(heading_diff)) / M_PI;

            // Clearance: already in [0,1] from minObstacleDist.
            // Zero-velocity trajectories stay in place — penalise them so the robot
            // doesn't score a free clearance bonus for standing still.
            const double clearance = (std::abs(traj.vx) < 1e-9 &&
                                      std::abs(traj.vth) < 1e-9) ? 0.0 : minObstacleDist(traj);

            // Velocity: normalised forward speed. Reverse trajectories score 0 here
            // so they are only selected when no forward option exists.
            const double velocity = std::max(0.0, traj.vx / max_vel_x_);

            // DWA objective: G = α·heading + β·clearance + γ·velocity
            const double score = alpha_ * heading
                               + beta_  * clearance
                               + gamma_ * velocity;

            if (score > best_score) {
                best_score = score;
                best = traj;
            }
        }
        return best;
    }

}  // namespace nlp_nav

PLUGINLIB_EXPORT_CLASS(nlp_nav::DWAController, nav2_core::Controller)