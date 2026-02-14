import cv2
import mediapipe as mp
import os


#gesture_name = "open_fist"
#gesture_name = "close_fist"
#gesture_name = "left"
gesture_name = "right"
save_path = f"img_gesture_dataset/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_list) * w) - 20
            x_max = int(max(x_list) * w) + 20
            y_min = int(min(y_list) * h) - 20
            y_max = int(max(y_list) * h) + 20

            x_min = max(0, x_min)
            y_min = max(0, y_min)

            cropped = frame[y_min:y_max, x_min:x_max]

            if cropped.size != 0:
                resized = cv2.resize(cropped, (64, 64))

                cv2.imshow("Cropped Hand", resized)

                key = cv2.waitKey(1)

                if key == ord('0'):
                    img_name = f"{save_path}/{count}.jpg"
                    cv2.imwrite(img_name, resized)
                    count += 1
                    print("Saved:", img_name)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#works in my_env