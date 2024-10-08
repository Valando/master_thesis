import numpy as np
import cv2 as cv
import glob

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, e.g., (0,0,0), (25,0,0), (50,0,0), ..., for an 8x6 chessboard grid
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 25  # Scale each square to 25 mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane

# Load all images for calibration
images = glob.glob('checkers/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners for an 8x6 grid
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

    # If found, add object points and refine image points
    if ret:
        objpoints.append(objp)

        # Refine corner positions for greater accuracy
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (8, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

# Calibrate the camera with the collected points
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret)
print(mtx)
# Calculate the reprojection error to evaluate the accuracy
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total error: {}".format(mean_error / len(objpoints)))

cv.destroyAllWindows()
