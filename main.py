import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import math
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

detector = HandDetector(detectionCon=0.8)   


def draw(img,buttonList):
    for button in buttonList:
        x,y=button.pos
        w,h=button.size
        cvzone.cornerRect(img,(button.pos[0],button.pos[1],button.size[0],button.size[1]),20,rt=0)
        cv2.rectangle(img,button.pos,(x+w,y+h),(255,0,255),cv2.FILLED)
        cv2.putText(img,button.text,(x+20,y+60),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),3)
    return img
#if you want to use transpancy in button then use this function
def draw2(img,buttonList):
    imgNew=np.zeros_like(img,np.uint8)    
    for button in buttonList:
        x,y=button.pos
        cvzone.cornerRect(imgNew,(button.pos[0],button.pos[1],button.size[0],button.size[1]),20,rt=0)
        cv2.rectangle(imgNew,button.pos,(x+button.size[0],y+button.size[1]),(255,0,255),cv2.FILLED)
        cv2.putText(imgNew,button.text,(x+20,y+60),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),3)
        out=img.copy()
        alpha=0.4
        mask=imgNew.astype(bool)
        
        out[mask]=cv2.addWeighted(img,alpha,imgNew,1-alpha,0)[mask]       
    return out



        
class Button():
    def __init__(self,pos,text,size=[85,85]):
        self.pos=pos
        self.size=size
        self.text=text      

   
keyboraed=Controller()      
finaltext=""
ButtonList=[]
keys=[["Q","W","E","R","T","Y","U","I","O","P","[","]"],["A","S","D","F","G","H","J","K","L",";","'"],["Z","X","C","V","B","N","M",",",".","/"]]
for i in range(len(keys)): 
        for j, key in enumerate(keys[i]):
            lp=(100*j)
            ButtonList.append(Button([lp+70,100*i+80],key))
            
            
lmlist=[] 
while True:
    sucess, img=cap.read()
    # img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                    
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    if id==8 or id==12:
                        if id==8:
                            x1=cx
                            y1=cy
                        if id==12:
                            x2=cx
                            y2=cy
                        for button in ButtonList:
                            lmlist.append([id,cx,cy])
                            x,y=button.pos
                            w,h=button.size
                            if x<cx<x+w and y<cy<y+h:
                                # print(id,cx,cy)
                               
                                
                                l=math.hypot(x2-x1,y2-y1)
                                # print(l)

                                if l<40:

                                    keyboraed.press(button.text)
                                    cv2.rectangle(img,button.pos,(x+w,y+h),(175,0,175),cv2.FILLED)
                                    cv2.putText(img,button.text,(x+20,y+60),cv2.FONT_HERSHEY_PLAIN,4,(0,255,255),3)
                                    sleep(0.2)
                                    finaltext+=button.text
                                   
               
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS) 
    cv2.rectangle(img,(50,400),(1200,500),(175,0,175),cv2.FILLED)
    cv2.putText(img,finaltext,(80,480),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),5)                               
    img=draw(img,ButtonList)
        
    cv2.imshow("image",img)
    cv2.waitKey(1)

