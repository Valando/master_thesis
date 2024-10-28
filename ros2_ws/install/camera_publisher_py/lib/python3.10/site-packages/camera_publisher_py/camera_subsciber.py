import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np


class CameraSubsciber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        
        self.camera_matrix = np.array([[1000, 0, 320],
                                       [0, 1000, 240],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([0.1, -0.25, 0.0, 0.0, 0.0], dtype=np.float32)


        self.subscription = self.create_subscription(
            Image,
            'camera_frame',  # Topic name
            self.listener_callback,
            10  # QoS profile (buffer size)
        )
        self.bridge = CvBridge()
        
    def listener_callback(self,msg):
        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Undistort the frame using the camera matrix and distortion coefficients
        undistorted_frame = self.undistort_image(frame)

        # Display the undistorted frame
        cv.imshow("Undistorted Frame", undistorted_frame)
        cv.waitKey(1)



    def undistort_image(self, frame):
        # Use the camera matrix and distortion coefficients to undistort the frame
        undistorted_frame = cv.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return undistorted_frame   


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubsciber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
