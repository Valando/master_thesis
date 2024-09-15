import cv2

img = cv2.imread("DJI_0001b.JPG")#picture
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #greyscale
sift = cv2.SIFT_create() #do sift
kp = sift.detect(gray,None) # create keypoints
img=cv2.drawKeypoints(gray,kp,img) #add keypoints highlighted to image

cv2.imwrite('sift_keypoints.jpg',img) #write the image to folder
video = cv2.VideoCapture('DJI_0004_MOV')
while video.isOpened():
    print("w")

    ret, frame = video.read()
    # if frame is read correctly ret is True
    if ret:
        print("receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
 
video.release()
cv2.destroyAllWindows()