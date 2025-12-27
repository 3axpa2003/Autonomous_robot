import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArrowDetectorNode(Node):
    def __init__(self):
        super().__init__('arrow_detector_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/arrow_direction',
            10
        )

        self.bridge = CvBridge()
        self.kernel = np.ones((5, 5), np.uint8)
        self.last_direction = None

        cv2.namedWindow("Arrow Detection", cv2.WINDOW_NORMAL)

        self.get_logger().info("Arrow detector node started.")

    def detect_arrow(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blur, 30, 100)
        dilated = cv2.dilate(canny, self.kernel, iterations=2)
        eroded = cv2.erode(dilated, self.kernel, iterations=1)

        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                # Arrow shape
                if len(approx) == 7:
                    _, _, angle = cv2.fitEllipse(approx)
                    if 80 < angle < 100:
                        x_vals = approx[:, 0, 0]
                        center_x = (x_vals.max() + x_vals.min()) / 2
                        return "left" if np.median(x_vals) < center_x else "right"

        return None

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        direction = self.detect_arrow(img)

        if direction and direction != self.last_direction:
            msg_out = String()
            msg_out.data = direction
            self.publisher.publish(msg_out)
            self.get_logger().info(f"Arrow detected: {direction}")
            self.last_direction = direction

        cv2.imshow("Arrow Detection", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArrowDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
