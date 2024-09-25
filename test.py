import cv2

img = cv2.imread("DJI_0001b.jpg")#picture

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #greyscale
orb = cv2.ORB_create() #do sift
kp1 , des1 = orb.detectAndCompute(gray,None) # create keypoints
img=cv2.drawKeypoints(gray,kp1,img) #add keypoints highlighted to image

cv2.imwrite('sift_keypoints.jpg',img) #write the image to folder
video = cv2.VideoCapture('DJI_0004.MOV')
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
des_list = []
matches_list = [1]
# matches = flann.knnMatch(des1,des2,k=2)
frame_counter =0
matches = 0
while video.isOpened():
   
    ret, frame = video.read()
    # if frame is read correctly ret is True
    if not ret:
        print("receive frame (stream end?). Exiting ...")
        break
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_counter == 5:
     kp2, des2= orb.detectAndCompute(gray2,None)
     des_list.append(des1)
     matches = flann.knnMatch(des1,des2,k=2)
     matchesMask = [[0,0] for i in range(len(matches))]
     for i,(m,n) in enumerate(matches):
      if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
     frame_counter = 0
     draw_params = dict(matchColor = (0,255,0),
     singlePointColor = (255,0,0),
     matchesMask = matchesMask,
     flags = cv2.DrawMatchesFlags_DEFAULT)
 
     img3 = cv2.drawMatchesKnn(gray,kp1,gray2,kp2,matches,None,**draw_params)
     matches_list.append(img3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
    frame_counter =+ frame_counter
cv2.imshow("matches", matches_list[0])
video.release()
cv2.destroyAllWindows()