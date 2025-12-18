import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraViewNode(Node):
    def __init__(self):
        super().__init__('camera_view_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        self.get_logger().info("Camera view node started and listening to /camera/image_raw ")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Cannot convert image: {e}")
            return

        self.get_logger().info(f"Image size: {cv_image.shape}")
        cv2.imshow("Camera View", cv_image)
        cv2.waitKey(10)

def main(args=None):
    rclpy.init(args=args)
    node = CameraViewNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
