import cv2

# Load the image
img = cv2.imread("DJI_0001b.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB for both image and video to match descriptor types
orb = cv2.ORB_create()
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
matches_list = []

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
        matchesMask = [[0, 0] for _ in range(len(matches))]
        for i, match in enumerate(matches):
            if len(match) == 2:  # Ensure there are at least two matches
                m, n = match
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]

        # Reset frame counter
        frame_counter = 0

        # Draw matches
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(gray, kp1, gray2, kp2, matches, None, **draw_params)

        # Save and display the matches
        matches_list.append(img3)
       # cv2.imshow("Matched Features", img3)

    # Increment the frame counter
    frame_counter += 1

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cv2.imshow("matches",matches_list[0])

# Release the video capture and close windows
video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
