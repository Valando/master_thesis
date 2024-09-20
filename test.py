import cv2

img = cv2.imread("DJI_0001b.jpg")#picture
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #greyscale
sift = cv2.SIFT_create() #do sift
kp1 , des1 = sift.detectAndCompute(gray,None) # create keypoints
img=cv2.drawKeypoints(gray,kp1,img) #add keypoints highlighted to image

cv2.imwrite('sift_keypoints.jpg',img) #write the image to folder
video = cv2.VideoCapture('DJI_0004.MOV')
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
 
# matches = flann.knnMatch(des1,des2,k=2)

while video.isOpened():
   

    ret, frame = video.read()
    # if frame is read correctly ret is True
    if not ret:
        print("receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
 
video.release()
cv2.destroyAllWindows()