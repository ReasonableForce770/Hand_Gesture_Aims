import cv2
import mediapipe as mp
import time
import csv

cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()


mpDraw = mp.solutions.drawing_utils
ct=0
pt=0
file = open('gesture_data.csv', 'a', newline='')
writer = csv.writer(file)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id==4 or id==8 or id==12 or id==16 or id==20:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            landmarks = []

            base_x = handLms.landmark[0].x
            base_y = handLms.landmark[0].y
            base_z = handLms.landmark[0].z

            for lm in handLms.landmark:
                landmarks.append(lm.x - base_x)
                landmarks.append(lm.y - base_y)
                landmarks.append(lm.z - base_z)
            print(len(landmarks))
            key = cv2.waitKey(1)
# 0 = open fist , 1 = close fist , 2 = peace sign , 3 = thumbs up , 4 = ok sign
            if key == ord('0'):
                landmarks.append(0)
                writer.writerow(landmarks)
                print("Saved Gesture 0")

            elif key == ord('1'):
                landmarks.append(1)
                writer.writerow(landmarks)
                print("Saved Gesture 1")
            elif key == ord('2'):
                landmarks.append(2)
                writer.writerow(landmarks)
                print("Saved Gesture 2")
            elif key == ord('3'):
                landmarks.append(3)
                writer.writerow(landmarks)
                print("Saved Gesture 3")
            elif key == ord('4'):
                landmarks.append(4)
                writer.writerow(landmarks)
                print("Saved Gesture 4")
    ct=time.time()
    fps=1/(ct-pt)
    pt=ct
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break
cap.release()
cv2.destroyAllWindows()
#works with my_env