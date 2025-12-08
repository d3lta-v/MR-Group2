#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
# from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, Buffer
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
# from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
# from zed_interfaces.msg import ObjectsStamped
from math import sqrt, cos, sin, pi, atan2, inf, degrees

import numpy as np
import sys

class PID:
    # NOTE: The laser scanner publishes messages at 50Hz.
    # We also assume 50Hz to be the control loop rate

    def __init__(self, Kp, Kd, Ki, Kp_angle):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        # feel free to add more parameters if needed
        self.Kp_angle = Kp_angle
        self.current_error = 0.0
        self.last_error = 0.0
        self.total_error = 0.0
        self.error_difference = self.current_error - self.last_error
    
    def update_control(self, current_error, angle_min):
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

        # Update error terms
        self.last_error = self.current_error
        self.current_error = current_error
        self.error_difference = self.current_error - self.last_error
        self.total_error = self.total_error + current_error
        
        # Citation and credits to https://github.com/ssscassio/ros-wall-follower-2-wheeled-robot/blob/master/catkin_ws/src/two-wheeled-robot-motion-planning/scripts/follow_wall.py
        # for design of a combined distance and angle based controller

        # return max(
        #         min(direction * (self.Kp * self.current_error + self.Kd * self.error_difference) + angle_p * (angle_min - (pi / 2) * direction)
        #         , 2.5)
        #     , -2.5)

        # NOTE: positive angular velocity => turning to the left
        # min() is used to clamp the output to 1.0 for turns during sensor blackout
        return min((self.Kp * self.current_error + self.Ki * self.total_error + self.Kd * self.error_difference) - self.Kp_angle * (angle_min + (pi / 2)), 1.0)


class MotionPIDController(Node):
    def __init__(self):
        super().__init__('motion_pid_controller')

        self.throttle_counter = 0  # throttling the logging messages so it doesn't spam the console

        self.forward_speed = self.declare_parameter("forward_speed").value
        self.Kp_ = self.declare_parameter("Kp").value #Kp
        self.Kd_ = self.declare_parameter("Kd").value #Kd
        self.Ki_ = self.declare_parameter("Ki").value #Ki
        self.Kp_angle_ = self.declare_parameter("Kp_angle").value

        # this is the publisher to publish the controls to
        # use this publisher to publish the controls determined by the PID controller to control the robot in gazebo
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10))

        self.odom_sub = self.create_subscription(
            Odometry, 
            '/zed/zed_node/odom',
            self.odometry_callback, 
            10)

        # this is the laser scan subscriber that executes the laser_scan_callback method each time a new laser
        # scan topic is received
        # if laser_sub is not receiving any messages, use QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10) instead
        # to match the QoSProfile of the laser scan publisher
        # self.laser_sub = self.create_subscription(
        #     LaserScan, 
        #     'scan', 
        #     self.laser_scan_callback, 
        #     QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10))

        # todo Part A: initialize your cross track error publisher here
        self.cte_publisher = self.create_publisher(Float32, '/cte', QoSProfile(depth=10))

        # todo Part B: initialize your PID controller here
        self.pid_controller = PID(self.Kp_, self.Kd_, self.Ki_, self.Kp_angle_)

        # You can use this to keep track of the position of the vehicle if necessary
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def odometry_callback(self, msg: Odometry):
        final_cte = msg.pose.pose.position.y # Purely use y position as CTE for straight line following
        cte_msg = Float32()
        # Clamp the CTE value to max range as Float32 cannot accept inf
        cte_msg.data = final_cte
        self.cte_publisher.publish(cte_msg)

        # Note that orientation is given as a quaternion
        # something = msg.pose.pose.orientation.x

        # Part B
        # 1) complete the PID class in the code above
        # 2) call the PID controller update method with the calculated cross track error to compute the control command to be given to the robot
        # 3) initialize a Twist message and populate it using the controls obtained
        # 4) publish the Twist message using the self.cmd_pub publisher
        # Example: cmd.angular.z = ??? (also remember to set your cmd.linear.x to get the robot to move forward)

        angle_min = 0.0 # No angle correction for odometry based CTE YET, maybe can use pose as an idea for robot orientation

        steering_angle = self.pid_controller.update_control(cte_msg.data, angle_min)
        actuator_cmd = Twist()
        actuator_cmd.linear.x = self.forward_speed
        actuator_cmd.angular.z = steering_angle
        self.cmd_pub.publish(actuator_cmd)

        self.throttle_counter += 1
        if self.throttle_counter > 2:
            self.throttle_counter = 0
            self.get_logger().info('CTE: %s' % str(cte_msg.data))
            self.get_logger().info('Angular velocity: %s' % str(steering_angle))
            self.get_logger().info('Angle against wall: %s' % str(degrees(angle_min)+90))

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

    wfh=MotionPIDController()
    rclpy.spin(wfh)
    wfh.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    
