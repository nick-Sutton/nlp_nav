#include "nav2_core/controller.hpp"

class DWAController : public nav2_core::Controller
{
    public:
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
        nav2_core::GoalChecker * goal_checker) override
    {
        // Build dynamic window
        double min_vx = std::max(params_.min_vel_x, velocity.linear.x - params_.acc_lim_x * dt_);
        double max_vx = std::min(params_.max_vel_x, velocity.linear.x + params_.acc_lim_x * dt_);
        double min_vth = std::max(-params_.max_vel_theta, velocity.angular.z - params_.acc_lim_theta * dt_);
        double max_vth = std::min(params_.max_vel_theta, velocity.angular.z + params_.acc_lim_theta * dt_);

        // Sample trajectories inside the window
        std::vector<Trajectory> candidates;

        for (double vx = min_vx; vx <= max_vx; vx += vx_step_) {
            for (double vth = min_vth; vth <= max_vth; vth += vth_step_) {

                // Roll out  each (vx, vth)
                Trajectory traj = simulateTrajectory(pose, vx, vth);

                // Discard unsafe trajectories
                if (isCollisionFree(traj)) {
                    candidates.push_back(traj);
                }
            }
        }

        // Score Trajectories:
        // G(v,w) = alpha*heading + beta*clearance + gamma*velocity
        Trajectory best = selectBest(candidates, pose);

        geometry_msgs::msg::TwistStamped cmd;
        cmd.twist.linear.x = best.vx;
        cmd.twist.angular.z = best.vth;
        return cmd;
    }

    Trajectory simulateTrajectory(
        const geometry_msgs::msg::PoseStamped &start,
        double vx, double vth)
    {
        Trajectory traj;
        double x   = start.pose.position.x;
        double y   = start.pose.position.y;
        double yaw = tf2::getYaw(start.pose.orientation);

        // Integrate forward using a circular arc motion model
        for (double t = 0; t <= sim_time_; t += sim_granularity_) {
            x   += vx * std::cos(yaw) * sim_granularity_;
            y   += vx * std::sin(yaw) * sim_granularity_;
            yaw += vth * sim_granularity_;

            traj.points.push_back({x, y, yaw});
        }

        traj.vx  = vx;
        traj.vth = vth;
        return traj;
    }

    /**
     * Check if a trajectory is in collision
     */
    bool isCollisionFree(const Trajectory &traj)
    {
        for (auto &pt : traj.points) {
            unsigned int mx, my;
            // Convert world coords to costmap cell
            if (!costmap_->worldToMap(pt.x, pt.y, mx, my)) {
                return false;
            }

            unsigned char cost = costmap_->getCost(mx, my);
            if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
                return false;
            }
        }
        return true;
    }

    /** Pick candidate trajectory based on DWA score */
    Trajectory selectBest(
        const std::vector<Trajectory> &candidates,
        const geometry_msgs::msg::PoseStamped &pose)
    {
        Trajectory best;
        double best_score = -std::numeric_limits<double>::infinity();

        for (auto &traj : candidates) {
            auto &end = traj.points.back();

            // Heading
            double goal_angle = std::atan2(
            goal_.y - end.y,
            goal_.x - end.x);
            double heading = M_PI - std::abs(end.yaw - goal_angle);

            // Clearance
            double clearance = minObstacleDist(traj);

            // Velocity
            double velocity = traj.vx;

            // DWA function
            double score = alpha_ * heading
                        + beta_  * clearance
                        + gamma_ * velocity;

            if (score > best_score) {
                best_score = score;
                best = traj;
            }
        }

        return best;
    }

  void setPlan(const nav_msgs::msg::Path & path) override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;
};