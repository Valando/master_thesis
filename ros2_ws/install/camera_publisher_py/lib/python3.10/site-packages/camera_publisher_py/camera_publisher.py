import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_frame', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # Publish at 10Hz
        self.bridge = CvBridge()
        self.ip_camera_url = 'http://172.24.120.73:8080/video'  # Replace with your IP camera URL
        stream ='/dev/video0'
       # stream = self.ip_camera_url
        self.cap = cv.VideoCapture(stream)
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the OpenCV image to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(ros_image)
            print("frames are here")
        else :
         print("no frames")

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
