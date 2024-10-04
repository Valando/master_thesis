import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from hsv import hsv_mask, desired_hsv

# Load the image
img = cv2.imread("DJI_0001.jpg")
img = img[1821:1920, 2635:2789]
cv2.imwrite('cropped_image.jpg', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# ORB for both image and video to match descriptor types
#orb = cv2.ORB_create()
orb = cv2.ORB_create(nfeatures=5000)  # Increase to detect more features

sift = cv2.SIFT_create()
#sift = cv2.SIFT_create(contrastThreshold=1)
f = 0
if f == 1:
    features = sift
else:
    features = orb

kp1, des1 = features.detectAndCompute(gray, None)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(gray, kp1, img)
cv2.imwrite('orb_keypoints.jpg', img_with_keypoints)

# Setup the video capture and FLANN matcher for ORB
video = cv2.VideoCapture('DJI_0004.MOV')
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
# Correct FLANN parameters for ORB

if f == 0:
 FLANN_INDEX_LSH = 6
 index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
 search_params = dict(checks=50)
else:
 FLANN_INDEX_KDTREE = 1
 index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
 search_params = dict(checks=50)


flann = cv2.FlannBasedMatcher(index_params, search_params)

frame_counter = 0

max_good_matches = 0

# Initialize best matches-related variables
best_kp2 = None
best_matches_list = None
best_matchesMask = None
best_gray2 = None
frame_mean = []
kp2_list = []
gray2_list = []
while video.isOpened():
    ret, frame = video.read()
   # if frame is not None:
     #frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    if not ret:
        print("End of video. Exiting ...")
        break
 
   

    # Match every 15th frame
    if frame_counter == 15:
        start_time = time.time()
        
        gray2 = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        kp2, des2 = features.detectAndCompute(gray2, None)

        # Match descriptors using FLANN
        matches = flann.knnMatch(des1, des2, k=2)
       
        # Apply ratio test
        matches_list = []
        matchesMask = []
        for i, match in enumerate(matches):
            if len(match) >= 2:  # Ensure there are at least two matches
                m, n = match
                if m.distance < 0.7 * n.distance:
                    matchesMask.append([1, 0])
                    matches_list.append([m])

        if len(matches_list) > max_good_matches:
            max_good_matches = len(matches_list)
            best_kp2 = kp2
            best_matches_list = matches_list.copy()
            best_matchesMask = matchesMask.copy() 
            best_gray2 = gray2.copy()
        kp2_list.append(kp2)
        gray2_list.append(gray2)
        # Reset frame counter
        frame_counter = 0
        end_time = time.time()
        frame_time = end_time - start_time
        print(frame_time)
        frame_mean.append(frame_time)
   # Increment the frame counter
    frame_counter += 1
        # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
     break
        
# Draw matches if best matches are found
if best_kp2 is not None:
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=best_matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(gray, kp1, best_gray2, best_kp2, best_matches_list, None, **draw_params)

draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
img4 = cv2.drawMatchesKnn(gray, kp1, gray2_list[0], kp2_list[0], matches_list, None, **draw_params)        
   


if best_kp2 is not None:
 plt.imshow(img3), plt.show()
plt.imshow(img4), plt.show()

print(np.mean(frame_mean)*1000,"ms")
# Release the video capture and close window
video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
