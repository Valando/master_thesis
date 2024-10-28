import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np


class CameraSubsciber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        
        # kamera matrix
        self.camera_matrix = np.array([[1000, 0, 320],
                                       [0, 1000, 240],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([0.1, -0.25, 0.0, 0.0, 0.0], dtype=np.float32)

        # Parameters lk
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # ORB 
        self.orb = cv.ORB_create(nfeatures=1000)
        self.min_feature = 5
        self.max_feature = 20
        
        self.subscription = self.create_subscription(
            Image,
            'camera_frame',  # Topic
            self.listener_callback,
            10  # buffer size
        )
        self.bridge = CvBridge()

        
        self.old_gray = None
        self.p0 = None
        self.mask = None
        self.color = None
        
    def listener_callback(self, msg):
        
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        
        #processed_frame = self.optical_flow(frame)
        kp_list = self.uniform_features(frame)
        processed_frame = cv.drawKeypoints(frame, kp_list, None, color=(0, 255, 0))

        # processed frame
        if processed_frame is not None:
            cv.imshow("Optical Flow", processed_frame)
            cv.waitKey(1)

    def optical_flow(self, frame):
       
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      
        if self.old_gray is None:
            self.old_gray = frame_gray
            kp = self.orb.detect(self.old_gray)
            self.p0 = np.array([kp.pt for kp in kp], dtype=np.float32).reshape(-1, 1, 2)

            
            self.mask = np.zeros_like(frame)
            self.color = np.random.randint(0, 255, (len(self.p0), 3))
            return frame

        # OF
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        
        if p1 is not None and len(p1[st == 1]) > 0:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

            img = cv.add(frame, self.mask)

            # Update 
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

            return img

        return None  

    def uniform_features(self,frame): 
        KeyPoint_list = []
        h,w,c = frame.shape
        s_h = int(h * 1/4) #devide by four
        s_w = int(w * 1/4) #device by four
        for y in range (0,h,s_h):
         for x in range(0,w,s_w):
             y1 = min(y+s_h,h)
             x1 = min(x+s_w,w)
             grid_segment = frame[y:y1,x:x1]
             kp = self.orb.detect(grid_segment,None) 


                
             if len(kp) < self.min_feature:
               self.orb.setFastThreshold(3)  # Lower threshold 
               kp = self.orb.detect(grid_segment, None)
         
             if len(kp) > self.max_feature:
               kp = sorted(kp, key=lambda k: k.response, reverse=True)
               kp = kp[:self.max_feature]
                
                
             for k in kp:
               k.pt = (k.pt[0] + x, k.pt[1] + y)
             KeyPoint_list.extend(kp)
        return KeyPoint_list

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubsciber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
