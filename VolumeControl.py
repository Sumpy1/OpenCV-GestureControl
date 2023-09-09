#Volume Control by HandGesture using mediapipe,opencv, and osascript by Gopal Pokharel (Sumiran)
import math
import time
import mediapipe as mp
import cv2 as cv
import numpy as np
import HandtrackingModule as htm
#Use pyclaw if you are on Windows, Osascript only works for Mac
import osascript

wCam, hcam= 1280, 720

#Use 1 instead of 0 if you are using external webcam
cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hcam)
pre_time=0
detector=htm.HandDetector(detectionCon=0.7)
volumeBar=0


while True:
    success, img =cap.read()
    img= detector.findhands(img)
    lmList=detector.findposition(img,draw=False)
    if len(lmList) != 0:
        # print(lmList[4],lmList[8])

        x1, y1= lmList[4][1],lmList[4][2]
        x2, y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2, (y1+y2)//2

        cv.circle(img,(x1,y1),15,(0,0,255),cv.FILLED)
        cv.circle(img,(x2,y2),15,(0, 0, 255), cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),3)
        cv.circle(img, (cx, cy), 15, (0, 0, 255), cv.FILLED)
        length= math.hypot(x2-x1,y2-y1)
        # print(length)

        volume =np.interp(length,[50,300],[0,100])
        volumeBar = np.interp(length, [50, 300], [400, 150])
        # print(int(length),volume)


        # # Trying to mute if you made a fist closing all fingers
        # if length < 10:
        #     hands_closed = True
        # else:
        #     hands_closed = False
        #
        # # Mute or unmute the volume based on hand status
        # if hands_closed:
        #     osascript.osascript("set volume output muted true ")
        #     osascript.osascript("set volume output volume 0")
        # else:
        #     osascript.osascript(f"set volume output volume {int(volume)}")


        osascript.osascript(f"set volume output volume {int(volume)}")



    cv.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv.rectangle(img, (50, int(volumeBar)), (85, 400), (128, 0, 128), cv.FILLED)

    #To see frame rate
    curr_time = time.time()
    fps = 1 / (curr_time - pre_time)
    pre_time = curr_time
    #print(fps)

    cv.putText(img, f'Volume', (10, 100), cv.FONT_ITALIC, 3, (0, 255, 255), thickness=3)
    #Uncomment the below line if you want to see fps on screen
    # cv.putText(img, f'FPS:{int(fps)}', (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), thickness=3)
    cv.imshow('Gesture Control',img)
    cv.waitKey(1)
