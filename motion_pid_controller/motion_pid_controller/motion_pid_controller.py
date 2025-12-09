#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
# from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, Buffer
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import Twist, Quaternion
from std_msgs.msg import Float32
# from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
# from zed_interfaces.msg import ObjectsStamped
from math import sqrt, cos, sin, pi, atan2, inf, degrees

import numpy as np
import sys

class AngularPID:
    def __init__(self, Kp, Kd, Ki, Kp_angle, integral_limit=None):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.Kp_angle = Kp_angle
        self.current_error = 0.0
        self.last_error = 0.0
        self.total_error = 0.0
        self.error_difference = 0.0

        # Simple integral windup limit (absolute value)
        self.integral_limit = integral_limit

    """ Reset the PID controller state. """
    def reset(self):
        self.current_error = 0.0
        self.last_error = 0.0
        self.total_error = 0.0
        self.error_difference = 0.0
    
    def update_control(self, current_error, angle_min, dt, logger):
        # HACK TODO: update_control's frequency is completely defined by the LaserScan callback frequency
        # which is not ideal. A better implementation would be to have a separate control loop running
        # at a fixed frequency independent of the LaserScan callback frequency.

        # todo Part B: use the current cross track error to update the control commands
        # to the husky such that it can better follow the wall
        # Hint:
        # You can update the controls using:
        # (1) only the distance to the closest wall
        # (2) the distance to the closest wall and the orientation with respect to the wall
        # If you select option 2, you might want to use cascading PID control.

        if dt <= 0.0:
            # Fallback to previous control if dt is invalid
            self.get_logger().warn('PID dt invalid!!', throttle_duration_sec=0.25)
            dt = 1e-6

        # Update error terms
        self.last_error = self.current_error
        self.current_error = current_error
        self.error_difference = (self.current_error - self.last_error) / dt

        # Integrate with respect to time
        self.total_error += self.current_error * dt

        # Integral windup protection
        if self.integral_limit is not None:
            if self.total_error > self.integral_limit:
                self.total_error = self.integral_limit
            elif self.total_error < -self.integral_limit:
                self.total_error = -self.integral_limit
        
        # Citation and credits to https://github.com/ssscassio/ros-wall-follower-2-wheeled-robot/blob/master/catkin_ws/src/two-wheeled-robot-motion-planning/scripts/follow_wall.py
        # for design of a combined distance and angle based controller

        # return max(
        #         min(direction * (self.Kp * self.current_error + self.Kd * self.error_difference) + angle_p * (angle_min - (pi / 2) * direction)
        #         , 2.5)
        #     , -2.5)

        # PID terms
        p_term = self.Kp * self.current_error
        i_term = self.Ki * self.total_error
        d_term = self.Kd * self.error_difference
        # Angle correction term (same as before, but you could also scale with dt if desired)
        angle_term = self.Kp_angle * angle_min

        # NOTE: Positive angular velocity => turning to the left
        control = p_term + i_term + d_term - angle_term

        logger.info('AP: %s' % str(p_term), throttle_duration_sec=0.25)
        logger.info('AI: %s' % str(i_term), throttle_duration_sec=0.25)
        logger.info('AD: %s' % str(d_term), throttle_duration_sec=0.25)
        logger.info('Angle term: %s' % str(angle_term), throttle_duration_sec=0.25)

        # Clamp output for safety, limited by robot capabilities
        control = max(min(control, 0.2), -0.2)

        # min() is used to clamp the output to 1.0 for turns during sensor blackout
        return control
    
class LinearPID:
    def __init__(self, Kp, Kd, Ki, max_linear_vel, integral_limit=None):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.max_linear_vel = max_linear_vel

        self.current_error = 0.0
        self.last_error = 0.0
        self.total_error = 0.0
        self.error_difference = 0.0

        # Simple integral windup limit (absolute value)
        self.integral_limit = integral_limit

    """ Reset the PID controller state. """
    def reset(self):
        self.current_error = 0.0
        self.last_error = 0.0
        self.total_error = 0.0
        self.error_difference = 0.0
    
    def update_control(self, current_error, dt, logger):
        if dt <= 0.0:
            # Fallback to previous control if dt is invalid
            self.get_logger().warn('PID dt invalid!!', throttle_duration_sec=0.25)
            dt = 1e-6

        # Update error terms
        self.last_error = self.current_error
        self.current_error = current_error
        self.error_difference = (self.current_error - self.last_error) / dt

        # Integrate with respect to time
        self.total_error += self.current_error * dt

        # Integral windup protection
        if self.integral_limit is not None:
            if self.total_error > self.integral_limit:
                self.total_error = self.integral_limit
            elif self.total_error < -self.integral_limit:
                self.total_error = -self.integral_limit
        
        # PID terms
        p_term = self.Kp * self.current_error
        i_term = self.Ki * self.total_error
        d_term = self.Kd * self.error_difference

        control = p_term + i_term + d_term

        logger.info('LP: %s' % str(p_term), throttle_duration_sec=0.25)
        logger.info('LI: %s' % str(i_term), throttle_duration_sec=0.25)
        logger.info('LD: %s' % str(d_term), throttle_duration_sec=0.25)

        # Clamp output for safety, limited by robot capabilities. No reverse movement allowed as well
        control = max(min(control, self.max_linear_vel), 0.0)

        # min() is used to clamp the output to 1.0 for turns during sensor blackout
        return control

class MotionPIDController(Node):
    def __init__(self):
        super().__init__('motion_pid_controller')

        self.forward_speed = self.declare_parameter("forward_speed", 0.5).value
        self.target_xpos = self.declare_parameter("target_xpos", 2.5).value
        self.AKp_ = self.declare_parameter("Kp", 0.2).value #Kp
        self.AKd_ = self.declare_parameter("Kd", 0.08).value #Kd
        self.AKi_ = self.declare_parameter("Ki", 0.00001).value #Ki
        self.AKp_angle_ = self.declare_parameter("Kp_angle", 1.5).value

        self.LKp_ = self.declare_parameter("LKp", 1.0).value
        self.LKd_ = self.declare_parameter("LKd", 0.0).value
        self.LKi_ = self.declare_parameter("LKi", 0.0).value

        # Control loop frequency (Hz) and timer
        self.control_frequency = self.declare_parameter("control_frequency", 50.0).value
        self.control_period = 1.0 / self.control_frequency

        # this is the publisher to publish the controls to
        # use this publisher to publish the controls determined by the PID controller to control the robot in gazebo
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10))

        self.odom_sub = self.create_subscription(
            Odometry, 
            '/zed/zed_node/odom',
            self.odometry_callback, 
            10)

        # todo Part A: initialize your cross track error publisher here
        self.cte_publisher = self.create_publisher(Float32, '/cte', QoSProfile(depth=10))

        # todo Part B: initialize your PID controller here
        self.angular_pid_controller = AngularPID(self.AKp_, self.AKd_, self.AKi_, self.AKp_angle_)
        self.linear_pid_controller = LinearPID(self.LKp_, self.LKd_, self.LKi_, self.forward_speed)

        # State for control loop
        self.latest_cte = 0.0
        self.latest_angle_min = 0.0
        self.last_control_time = self.get_clock().now()
        self.latest_xpos = 0.0

        # Fixed-rate control loop timer
        self.control_timer = self.create_timer(self.control_period, self.control_loop)

        # You can use this to keep track of the position of the vehicle if necessary
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    
    def quat_to_yaw(self, qx, qy, qz, qw):
        """Convert quaternion to yaw (heading) in radians."""
        # ROS standard ZYX convention
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return atan2(siny_cosp, cosy_cosp)

    def odometry_callback(self, msg: Odometry):
        # Purely use y position as CTE for straight line following
        final_cte = msg.pose.pose.position.y 

        cte_msg = Float32()
        cte_msg.data = final_cte
        self.cte_publisher.publish(cte_msg)

        yaw = self.quat_to_yaw(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        self.get_logger().info('Yaw (in degrees): %s' % str(degrees(yaw)), throttle_duration_sec=0.25)

        # Part B
        # 1) complete the PID class in the code above
        # 2) call the PID controller update method with the calculated cross track error to compute the control command to be given to the robot
        # 3) initialize a Twist message and populate it using the controls obtained
        # 4) publish the Twist message using the self.cmd_pub publisher
        # Example: cmd.angular.z = ??? (also remember to set your cmd.linear.x to get the robot to move forward)

        self.latest_cte = final_cte
        self.latest_angle_min = yaw # TODO: Verify that this is supposed to be the angle relative to the line/wall/lane
        self.latest_xpos = msg.pose.pose.position.x

    def control_loop(self):
        # Compute time delta
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = now

        # Get control from PID controller
        steering_angle = self.angular_pid_controller.update_control(self.latest_cte, self.latest_angle_min, dt, self.get_logger())
        linear_velocity = self.linear_pid_controller.update_control(self.target_xpos - self.latest_xpos, dt, self.get_logger())

        # Publish cmd_vel message
        actuator_cmd = Twist()
        actuator_cmd.linear.x = linear_velocity
        actuator_cmd.angular.z = steering_angle
        self.cmd_pub.publish(actuator_cmd)

        self.get_logger().info('================================================', throttle_duration_sec=0.25)
        self.get_logger().info('CTE: %s' % str(self.latest_cte), throttle_duration_sec=0.25)
        self.get_logger().info('Steering command (Angular velocity in radians): %s' % str(steering_angle), throttle_duration_sec=0.25)
        self.get_logger().info('Angle to lane: %s' % str(degrees(self.latest_angle_min)), throttle_duration_sec=0.25)
        self.get_logger().info('Linear error: %s' % str(self.target_xpos - self.latest_xpos), throttle_duration_sec=0.25)
        self.get_logger().info('Velocity command: %s ' % str(linear_velocity), throttle_duration_sec=0.25)
        self.get_logger().info('================================================\n', throttle_duration_sec=0.25)

    # def laser_scan_callback(self, msg: LaserScan):
    #     # Part A
    #     # Calculate the cross track error by using the information given by the laser scanner
    #     # Hint: refer to the information given in the assignment sheet to better understand the angles of the laser scanner

    #     # The laser scanner has the following specs:
    #     # Angle resolution: 0.00655 rad = 0.375 deg
    #     # Angle limits: -135deg to +135deg (-2.356 rad to +2.356rad), full sweep of 270 degrees
    #     # Range limits: 0.1m to 30m
    #     # Number of readings: 720

    #     # Only publish the minimum range by the left side of the laser, minus the desired distance to the wall
    #     final_cte = min(min(msg.ranges[0:360]), inf) - self.desired_distance_from_wall
    #     cte_msg = Float32()
    #     # Clamp the CTE value to max range as Float32 cannot accept inf
    #     cte_msg.data = msg.range_max if final_cte > msg.range_max else final_cte

    #     # Determine angle of adjacent wall
    #     # Citation and credits to https://github.com/ssscassio/ros-wall-follower-2-wheeled-robot/blob/master/catkin_ws/src/two-wheeled-robot-motion-planning/scripts/follow_wall.py
    #     # for design of a combined distance and angle based controller
    #     angular_direction = -1
    #     size = len(msg.ranges)
    #     min_index = int(size * (angular_direction + 1) / 4) # 0/4
    #     max_index = int(size * (angular_direction + 3) / 4) # 2/4
    #     for i in range(min_index, max_index):
    #         if msg.ranges[i] < msg.ranges[min_index] and msg.ranges[i] > 0.01:
    #             min_index = i
    #     angle_min = (min_index - size / 2) * msg.angle_increment

    #     self.cte_publisher.publish(cte_msg)

    #     # Part B
    #     # 1) complete the PID class in the code above
    #     # 2) call the PID controller update method with the calculated cross track error to compute the control command to be given to the robot
    #     # 3) initialize a Twist message and populate it using the controls obtained
    #     # 4) publish the Twist message using the self.cmd_pub publisher
    #     # Example: cmd.angular.z = ??? (also remember to set your cmd.linear.x to get the robot to move forward)
    #     steering_angle = self.pid_controller.update_control(cte_msg.data, angle_min)
    #     actuator_cmd = Twist()
    #     actuator_cmd.linear.x = self.forward_speed
    #     actuator_cmd.angular.z = steering_angle
    #     self.cmd_pub.publish(actuator_cmd)

    #     self.throttle_counter += 1
    #     if self.throttle_counter > 2:
    #         self.throttle_counter = 0
    #         self.get_logger().info('CTE: %s' % str(cte_msg.data))
    #         self.get_logger().info('Angular velocity: %s' % str(steering_angle))
    #         self.get_logger().info('Angle against wall: %s' % str(degrees(angle_min)+90))

    #     # Part C
    #     # Update the PID class with the parameters obtained from rqt
    #     # Set the initial parameters in the file wall_following_assignment/launch/wall_follower_python.launch.py
    #     self.forward_speed = self.get_parameter("forward_speed").value
    #     self.desired_distance_from_wall = self.get_parameter("desired_distance_from_wall").value
    #     self.Kp_ = self.get_parameter("Kp").value #Kp
    #     self.Kd_ = self.get_parameter("Kd").value #Kd
    #     self.Ki_ = self.get_parameter("Ki").value #Ki
    #     self.Kp_angle_ = self.get_parameter("Kp_angle").value

    #     self.pid_controller.Kp = self.Kp_
    #     self.pid_controller.Kd = self.Kd_
    #     self.pid_controller.Ki = self.Ki_
    #     self.pid_controller.Kp_angle = self.Kp_angle_

def main(args=None):
    rclpy.init(args=args)
    
    """
        Arguments available:
        parameters=[{"forward_speed": forward_speed_arg,
                    "Kp" : Kp_val,
                    "Kd" : Kd_val,
                    "Ki" : Ki_val,
                    "Kp_angle" : Kp_angle_val,
                    }]
    """

    wfh=MotionPIDController()
    rclpy.spin(wfh)
    wfh.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    
