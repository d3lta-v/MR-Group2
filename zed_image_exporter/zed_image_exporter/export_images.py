import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import cv2

class ZedImageExporter(Node):
    def __init__(self):
        super().__init__('zed_image_exporter')

        # Declare parameters
        self.declare_parameter('bag_path', 'project_0.db3')
        self.declare_parameter('output_folder', 'output')
        self.declare_parameter('frame_skip', 10)
        self.declare_parameter('topic_name', '/zed/zed_node/rgb/image_rect_color')

        self.bag_path = self.get_parameter('bag_path').get_parameter_value().string_value
        self.output_folder = self.get_parameter('output_folder').get_parameter_value().string_value
        self.frame_skip = self.get_parameter('frame_skip').get_parameter_value().integer_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.image_count = 0
        self.export_count = 0

        self.export_images()

    def export_images(self):
        os.makedirs(self.output_folder, exist_ok=True)

        # Open the bag
        storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', 'cdr')

        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()
        topic_names = [t.name for t in topic_types]

        if self.topic_name not in topic_names:
            self.get_logger().error(f"Topic {self.topic_name} not found in bag!")
            return

        self.get_logger().info(f"Exporting images from topic {self.topic_name} every {self.frame_skip} frames...")

        # Loop through messages
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic != self.topic_name:
                continue

            self.image_count += 1
            if self.image_count % self.frame_skip != 0:
                continue

            # Deserialize and convert to OpenCV image
            msg = deserialize_message(data, Image)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Save image
            output_path = os.path.join(self.output_folder, f'image_{self.export_count:05d}.png')
            cv2.imwrite(output_path, cv_image)
            self.export_count += 1

        self.get_logger().info(f"Exported {self.export_count} images to {self.output_folder}")

def main(args=None):
    rclpy.init(args=args)
    node = ZedImageExporter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
