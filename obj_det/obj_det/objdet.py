import rclpy
from rclpy.node import Node
from zed_interfaces.msg import ObjectsStamped
from rclpy.serialization import deserialize_message
import math

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float32MultiArray

#
# This code is modified from obj_det_visualizer provided in the tutorial
# All credit goes to the original authors of the tutorial code.
#

class ObjectSubscriber(Node):

    def __init__(self):
        super().__init__('ObjectSubscriber')
        self.subscription = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        # Create a publisher for the Marker message
        self.marker_pub = self.create_publisher(Marker, '/bounding_box', 10)
        
        # Create a publisher for the Float32MultiArray message
        self.publisher = self.create_publisher(Float32MultiArray, '/bounding_box_corners', 10)

        ### Constants
        self.desired_distance_from_obj = 1.0

    def listener_callback(self, msg):

        # Create and initialize the Marker message
        self.marker = Marker()
        self.marker.header.frame_id = "map"  # Coordinate frame
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.ns = "bounding_box"
        self.marker.id = 0
        self.marker.type = Marker.LINE_LIST
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.1  # Line width
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0

        ## Custom code
        distances = [] # create list of distances, may be empty

        for obj in msg.objects:
            corners = self.find_corners(obj)

            # Add the points to form the bounding box
            self.add_bounding_box_edges(corners)
            self.publish_bounding_box(corners)

            ## Custom code
            # Get absolute distance between car and bag detected
            distances.append(math.sqrt(obj.position[0] ** 2 + obj.position[1] ** 2))


    def publish_bounding_box(self, corners):
        """ Publish the bounding box corners using a user-defined function. """
        corners_data = [d for c in corners for d in c]
        
        if len(corners_data) != 24:
            self.get_logger().warn("The provided data doesn't contain exactly 24 floats.")
            return
        
        # Create a Float32MultiArray message
        msg = Float32MultiArray()
        msg.data = corners_data
        
        # Publish the message to the '/bounding_box_corners' topic
        self.publisher.publish(msg)
        # self.get_logger().info(f"Publishing bounding box corners: {msg.data}")

    def add_bounding_box_edges(self, corners):
        # List of the 12 edges of the cuboid (connecting the corners)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Add the points to the Marker message
        for edge in edges:
            point_start = corners[edge[0]]
            point_end = corners[edge[1]]
            
            # Create Point messages for the start and end of the edge
            p_start = Point()
            p_start.x, p_start.y, p_start.z = point_start
            
            p_end = Point()
            p_end.x, p_end.y, p_end.z = point_end
            
            # Add points to the marker
            self.marker.points.append(p_start)
            self.marker.points.append(p_end)
        
        # Publish the marker
        self.publish_marker()

    def find_corners(self, msg):
        corners_obj = msg.bounding_box_3d.corners
        corners = [[c.kp[0].item(), c.kp[1].item(), c.kp[2].item()] for c in corners_obj]
        # print(corners[0][0])
        # print(type(corners[0][0]))

        return corners

    def publish_marker(self):
        # Publish the marker to RViz
        self.marker_pub.publish(self.marker)
        # self.get_logger().info("Bounding box marker published to RViz.")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSubscriber()

    # Set the publishing rate (10 Hz)
    rate = node.create_rate(10)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
