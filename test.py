import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("DJI_0001.jpg")
img = img[1821:1920, 2635:2789]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB for both image and video to match descriptor types
orb = cv2.ORB_create()
sift = cv2.SIFT_create()
brisk = cv2.BRISK_create()

kp1, des1 = orb.detectAndCompute(gray, None)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(gray, kp1, img)
cv2.imwrite('orb_keypoints.jpg', img_with_keypoints)

# Setup the video capture and FLANN matcher for ORB
video = cv2.VideoCapture('DJI_0004.MOV')

# Correct FLANN parameters for ORB
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

frame_counter = 0

max_good_matches = 0

# Initialize best matches-related variables
best_kp2 = None
best_matches_list = None
best_matchesMask = None
best_gray2 = None

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("End of video. Exiting ...")
        break

    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match every 5th frame
    if frame_counter == 5:
        kp2, des2 = orb.detectAndCompute(gray2, None)

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

        # Reset frame counter
        frame_counter = 0

        # Draw matches if best matches are found
        if best_kp2 is not None:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=best_matchesMask,
                               flags=cv2.DrawMatchesFlags_DEFAULT)
            img3 = cv2.drawMatchesKnn(gray, kp1, best_gray2, best_kp2, best_matches_list, None, **draw_params)

        
    # Increment the frame counter
    frame_counter += 1

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
plt.imshow(img3), plt.show()
# Release the video capture and close windows
video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
