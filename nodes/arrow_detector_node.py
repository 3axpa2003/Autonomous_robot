import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/robot_command',
            10
        )

        self.bridge = CvBridge()

        # Stop sign classifier
        self.stop_cascade = cv2.CascadeClassifier(
            'stop_sign_classifier_2.xml'
        )

        self.kernel = np.ones((5, 5), np.uint8)
        self.last_command = None

        cv2.namedWindow("Vision View", cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "Vision node started and listening to /camera/image_raw"
        )

    # -------- STOP DETECTION --------
    def detect_stop(self, gray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        stops = self.stop_cascade.detectMultiScale(
            blur,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return stops

    # -------- ARROW DETECTION --------
    def detect_arrow(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 30, 100)

        dilated = cv2.dilate(edges, self.kernel, iterations=2)
        eroded = cv2.erode(dilated, self.kernel, iterations=1)

        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4000:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 7:
                x_vals = approx[:, 0, 0]
                center_x = (x_vals.max() + x_vals.min()) / 2

                if np.mean(x_vals) < center_x:
                    return "left"
                else:
                    return "right"

        return None

    # -------- CALLBACK --------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f"Cannot convert image: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        command = None

        # 1) STOP has priority
        stops = self.detect_stop(gray)
        if len(stops) > 0:
            command = "stop"
            for (x, y, w, h) in stops:
                cv2.rectangle(
                    cv_image, (x, y), (x+w, y+h), (0, 0, 255), 2
                )
                cv2.putText(
                    cv_image, "STOP", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

        # 2) Arrow detection
        if command is None:
            direction = self.detect_arrow(cv_image)
            if direction:
                command = direction
                cv2.putText(
                    cv_image, direction.upper(), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3
                )

        # 3) Publish only if changed
        if command and command != self.last_command:
            msg_out = String()
            msg_out.data = command
            self.publisher.publish(msg_out)
            self.get_logger().info(f"Published command: {command}")
            self.last_command = command

        cv2.imshow("Vision View", cv_image)
        cv2.waitKey(10)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
