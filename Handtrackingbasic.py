import cv2 as cv
import time
import mediapipe as mp


pre_time=0
curr_time=0
cap=cv.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

while True:
    success, img =cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c=img.shape
                cx, cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==4:
                    cv.circle(img,(cx,cy),25,(0,0,255),cv.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    curr_time=time.time()
    fps=1/(curr_time-pre_time)
    pre_time=curr_time

    cv.putText(img,str(f'FPS:{int(fps)}'),(10,70),cv.FONT_ITALIC,3,(0,255,255),thickness=3)
    cv.imshow('Sumiran',img)
    cv.waitKey(1)





