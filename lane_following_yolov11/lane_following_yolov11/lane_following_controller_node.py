#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np
import time


class PID:
    def __init__(self, Kp, Kd, Ki, i_min=-float('inf'), i_max=float('inf')):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.i_min = i_min
        self.i_max = i_max
        self._prev_error = None
        self._integral = 0.0
        
        # Logging
        self._total_updates = 0
        self._last_p = 0.0
        self._last_i = 0.0
        self._last_d = 0.0

    def reset(self):
        self._prev_error = None
        self._integral = 0.0
        self._total_updates = 0
    
    def update_control(self, current_error, dt):
        """
        Calculate PID control output
        
        Args:
            current_error: Current error value
            dt: Time step since last update
            
        Returns:
            Control output (angular velocity)
        """
        self._total_updates += 1
        
        # Proportional term
        self._last_p = self.Kp * current_error
        p = self._last_p

        # Handle first iteration or invalid dt
        if dt is None or dt <= 0.0:
            self._prev_error = current_error
            return p

        # Integral term with anti-windup
        self._integral += current_error * dt
        if self._integral > self.i_max:
            self._integral = self.i_max
        elif self._integral < self.i_min:
            self._integral = self.i_min
        self._last_i = self.Ki * self._integral
        i = self._last_i

        # Derivative term
        if self._prev_error is None:
            self._last_d = 0.0
            d = 0.0
        else:
            derivative = (current_error - self._prev_error) / dt
            self._last_d = self.Kd * derivative
            d = self._last_d

        self._prev_error = current_error

        # Combined control output
        u = p + i + d
        return u
    
    def get_components(self):
        """Return last computed P, I, D components for logging"""
        return self._last_p, self._last_i, self._last_d


class LaneFollowingController(Node):
    def __init__(self):
        super().__init__('lane_following_controller')
        
        # Declare parameters
        self.declare_parameter('forward_speed', 1.0)  # m/s
        self.declare_parameter('Kp', 0.005)  # Proportional gain
        self.declare_parameter('Kd', 0.01)   # Derivative gain
        self.declare_parameter('Ki', 0.0001) # Integral gain
        self.declare_parameter('max_angular_velocity', 0.5)  # rad/s
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('enable_control', True)  # Enable/disable control output
        self.declare_parameter('lateral_error_threshold', 0.3)  # Fraction of image width
        self.declare_parameter('log_interval', 10)  # Log every N control updates
        
        # Get parameters
        self.forward_speed = self.get_parameter('forward_speed').value
        self.Kp_ = self.get_parameter('Kp').value
        self.Kd_ = self.get_parameter('Kd').value
        self.Ki_ = self.get_parameter('Ki').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.enable_control = self.get_parameter('enable_control').value
        self.lateral_error_threshold = self.get_parameter('lateral_error_threshold').value
        self.log_interval = self.get_parameter('log_interval').value
        
        # Initialize PID controller
        self.pid = PID(self.Kp_, self.Kd_, self.Ki_, i_min=-0.5, i_max=0.5)
        
        # Time tracking for dt calculation
        self._last_time_ns = None
        self._last_lane_time_ns = None
        self._controller_start_time = time.time()
        
        # State tracking
        self.last_lateral_error = 0.0
        self.lanes_detected = False
        self.image_center_x = None
        self.control_update_count = 0
        self.stop_count = 0
        
        # Statistics tracking
        self.total_error_sum = 0.0
        self.max_error = 0.0
        self.min_error = 0.0
        self.total_angular_velocity = 0.0
        self.max_angular_velocity_used = 0.0
        
        # Publishers
        self.cmd_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            QoSProfile(depth=10)
        )
        
        self.error_pub = self.create_publisher(
            Float32,
            '/lane_lateral_error',
            QoSProfile(depth=10)
        )
        
        # Subscribers
        self.lane_sub = self.create_subscription(
            Float32MultiArray,
            '/lane_boundaries',
            self.lane_callback,
            QoSProfile(depth=10)
        )
        
        # Watchdog timer - stop robot if no lane data received
        self.watchdog_timer = self.create_timer(0.5, self.watchdog_callback)
        
        # Statistics logging timer
        self.stats_timer = self.create_timer(5.0, self.log_statistics)
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Lane Following Controller Initialized\n'
            f'{"="*60}\n'
            f'  Forward speed: {self.forward_speed} m/s\n'
            f'  PID gains:\n'
            f'    Kp (Proportional): {self.Kp_}\n'
            f'    Ki (Integral):     {self.Ki_}\n'
            f'    Kd (Derivative):   {self.Kd_}\n'
            f'  Max angular velocity: {self.max_angular_velocity} rad/s\n'
            f'  Publishing to: {self.cmd_vel_topic}\n'
            f'  Control enabled: {self.enable_control}\n'
            f'  Error threshold: {self.lateral_error_threshold*100:.0f}% of image width\n'
            f'  Log interval: Every {self.log_interval} control updates\n'
            f'{"="*60}'
        )
    
    def lane_callback(self, msg: Float32MultiArray):
        """
        Process lane boundary data and calculate control commands
        
        Message format: [left_x, right_x, image_width]
        """
        callback_start = time.time()
        
        if len(msg.data) != 3:
            self.get_logger().error('Invalid lane boundary message format')
            return
        
        left_x = msg.data[0]
        right_x = msg.data[1]
        image_width = msg.data[2]
        
        # Store image center for reference
        self.image_center_x = image_width / 2.0
        
        # Calculate lane center
        lane_center_x = (left_x + right_x) / 2.0
        lane_width = right_x - left_x
        
        # Calculate lateral error (positive = robot is to the right of lane center)
        # Error in pixels - robot should steer left (negative angular vel) if positive error
        lateral_error = lane_center_x - self.image_center_x
        
        # Normalize error by image width for consistency
        normalized_error = lateral_error / image_width
        
        self.last_lateral_error = lateral_error
        self.lanes_detected = True
        self._last_lane_time_ns = self.get_clock().now().nanoseconds
        
        # Update statistics
        self.total_error_sum += abs(lateral_error)
        self.max_error = max(self.max_error, abs(lateral_error))
        if self.control_update_count == 0:
            self.min_error = abs(lateral_error)
        else:
            self.min_error = min(self.min_error, abs(lateral_error))
        
        # Publish lateral error
        error_msg = Float32()
        error_msg.data = float(lateral_error)
        self.error_pub.publish(error_msg)
        
        self.get_logger().debug(
            f'Lane data received: left={left_x:.1f}, right={right_x:.1f}, '
            f'width={lane_width:.1f}, center={lane_center_x:.1f}, error={lateral_error:.1f}'
        )
        
        # Check if error is within acceptable range
        if abs(normalized_error) > self.lateral_error_threshold:
            self.stop_count += 1
            self.get_logger().warn(
                f'Lateral error too large: {lateral_error:.1f}px '
                f'({normalized_error*100:.1f}% of image width). Stopping. '
                f'(Stop count: {self.stop_count})',
                throttle_duration_sec=1.0
            )
            self._stop_robot()
            return
        
        # Calculate dt
        now_ns = self.get_clock().now().nanoseconds
        if self._last_time_ns is None:
            dt = 0.05  # Default 50ms
        else:
            dt = (now_ns - self._last_time_ns) * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 0.05
        self._last_time_ns = now_ns
        
        # Update PID gains from parameters (allows runtime tuning)
        self.Kp_ = float(self.get_parameter('Kp').value)
        self.Ki_ = float(self.get_parameter('Ki').value)
        self.Kd_ = float(self.get_parameter('Kd').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.enable_control = bool(self.get_parameter('enable_control').value)
        
        self.pid.Kp = self.Kp_
        self.pid.Ki = self.Ki_
        self.pid.Kd = self.Kd_
        
        # Calculate control command
        # Negative error (left of center) -> positive angular vel (turn right)
        # Positive error (right of center) -> negative angular vel (turn left)
        angular_velocity = -self.pid.update_control(lateral_error, dt)
        
        # Get PID components for logging
        p_term, i_term, d_term = self.pid.get_components()
        
        # Clamp angular velocity
        angular_velocity_clamped = np.clip(
            angular_velocity,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )
        
        was_clamped = abs(angular_velocity) > self.max_angular_velocity
        
        # Update statistics
        self.total_angular_velocity += abs(angular_velocity_clamped)
        self.max_angular_velocity_used = max(self.max_angular_velocity_used, abs(angular_velocity_clamped))
        
        # Publish control command only if enabled and lanes detected
        if self.enable_control:
            cmd = Twist()
            cmd.linear.x = float(self.forward_speed)
            cmd.angular.z = float(angular_velocity_clamped)
            self.cmd_pub.publish(cmd)
            
            self.control_update_count += 1
            
            # Detailed logging every N updates
            if self.control_update_count % self.log_interval == 0:
                callback_time = (time.time() - callback_start) * 1000
                self.get_logger().info(
                    f'\n--- Control Update #{self.control_update_count} ---\n'
                    f'  Lane boundaries: L={left_x:.1f}px, R={right_x:.1f}px, Width={lane_width:.1f}px\n'
                    f'  Lane center: {lane_center_x:.1f}px (image center: {self.image_center_x:.1f}px)\n'
                    f'  Lateral error: {lateral_error:+.1f}px ({normalized_error*100:+.1f}%)\n'
                    f'  PID components:\n'
                    f'    P term: {-p_term:+.4f} (Kp={self.Kp_})\n'
                    f'    I term: {-i_term:+.4f} (Ki={self.Ki_})\n'
                    f'    D term: {-d_term:+.4f} (Kd={self.Kd_})\n'
                    f'  Angular velocity: {angular_velocity:+.4f} rad/s {" (CLAMPED)" if was_clamped else ""}\n'
                    f'  Commanded: linear={self.forward_speed:.2f} m/s, angular={angular_velocity_clamped:+.3f} rad/s\n'
                    f'  dt: {dt*1000:.1f}ms, Callback time: {callback_time:.1f}ms'
                )
            else:
                # Brief logging for other updates
                self.get_logger().info(
                    f'Update #{self.control_update_count}: Error={lateral_error:+6.1f}px | '
                    f'Angular={angular_velocity_clamped:+.3f} rad/s | '
                    f'P={-p_term:+.3f} I={-i_term:+.3f} D={-d_term:+.3f}',
                    throttle_duration_sec=0.5
                )
        else:
            self.get_logger().info(
                f'Control DISABLED. Error: {lateral_error:+6.1f}px | '
                f'Would command angular: {angular_velocity_clamped:+.3f} rad/s',
                throttle_duration_sec=1.0
            )
    
    def watchdog_callback(self):
        """
        Stop robot if no lane data received recently
        """
        if self._last_lane_time_ns is None:
            return
        
        time_since_last_lane = (self.get_clock().now().nanoseconds - self._last_lane_time_ns) * 1e-9
        
        # If no lane data for more than 1 second, stop the robot
        if time_since_last_lane > 1.0:
            if self.lanes_detected:
                self.get_logger().warn(
                    f'Watchdog triggered: No lane data for {time_since_last_lane:.2f}s. Stopping robot.'
                )
                self.lanes_detected = False
                self._stop_robot()
    
    def log_statistics(self):
        """Log performance statistics periodically"""
        if self.control_update_count == 0:
            return
        
        runtime = time.time() - self._controller_start_time
        avg_error = self.total_error_sum / self.control_update_count
        avg_angular_velocity = self.total_angular_velocity / self.control_update_count
        control_rate = self.control_update_count / runtime if runtime > 0 else 0
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Controller Statistics (Runtime: {runtime:.1f}s)\n'
            f'{"="*60}\n'
            f'  Control updates: {self.control_update_count}\n'
            f'  Control rate: {control_rate:.1f} Hz\n'
            f'  Lateral error:\n'
            f'    Average: {avg_error:.1f}px\n'
            f'    Min: {self.min_error:.1f}px\n'
            f'    Max: {self.max_error:.1f}px\n'
            f'  Angular velocity:\n'
            f'    Average: {avg_angular_velocity:.3f} rad/s\n'
            f'    Max used: {self.max_angular_velocity_used:.3f} rad/s\n'
            f'    Max allowed: {self.max_angular_velocity:.3f} rad/s\n'
            f'  Stop events: {self.stop_count}\n'
            f'  Current PID gains: Kp={self.Kp_}, Ki={self.Ki_}, Kd={self.Kd_}\n'
            f'{"="*60}'
        )
    
    def _stop_robot(self):
        """Send stop command to robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        # Reset PID to avoid integral windup
        self.pid.reset()
        self.get_logger().debug('Robot stopped, PID reset')


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        # Log final statistics
        runtime = time.time() - node._controller_start_time
        node.get_logger().info(
            f'\n{"="*60}\n'
            f'Final Controller Statistics\n'
            f'{"="*60}\n'
            f'  Total runtime: {runtime:.1f}s\n'
            f'  Total control updates: {node.control_update_count}\n'
            f'  Average control rate: {node.control_update_count/runtime:.1f} Hz\n'
            f'  Stop events: {node.stop_count}\n'
            f'{"="*60}'
        )
        # Stop robot on shutdown
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()