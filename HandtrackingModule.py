import time
import mediapipe as mp
import cv2 as cv

class HandDetector():
    def __init__(self,mode=False, maxhands=2,model_complexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxhands = maxhands
        self.model_complexity=model_complexity #new version of media pipe need this argument
        self.detectionCon=detectionCon
        self.trackCon=trackCon


        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxhands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findposition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):

                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                # if id == 4:
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx, cy), 25, (0, 0, 255))

        return lmList
def main():
    pre_time=0
    curr_time=0
    cap = cv.VideoCapture(0)
    detector=HandDetector()

    while True:
        success, img = cap.read()
        img=detector.findhands(img)
        lmList =detector.findposition(img)
        if len(lmList) !=0:
            print(lmList[4])
        curr_time = time.time()
        fps = 1 / (curr_time - pre_time)
        pre_time = curr_time

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), thickness=3)
        cv.imshow('Gesture Control', img)
        cv.waitKey(1)

if __name__ =="__main__":
    main()




