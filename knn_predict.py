import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
pred_queue = deque(maxlen=5)

model = joblib.load("knn_model.pkl")
CONF_THRESHOLD = 0.75

labels = {
    0: "Open Fist",
    1: "Closed Fist",
    2: "Peace Sign",
    3: "Thumbs Up",
    4: "Ok Sign"
}
# 0 = open fist , 1 = close fist , 2 = peace sign , 3 = thumbs up , 4 = ok sign
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            landmark_list = []

            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                landmark_list.extend([
                                    lm.x - wrist.x,
                                    lm.y - wrist.y,
                                    lm.z - wrist.z
                                    ])

            prediction = model.predict([landmark_list])
            pred = int(prediction[0])
            gesture_name = labels.get(pred, f"Unknown ({pred})")
            cv2.putText(
                        frame,
                         f"Raw pred: {prediction[0]}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                         2
                       )

            cv2.putText(frame, gesture_name,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
            pred_queue.append(pred)
            confidence = np.max(prediction)
            if confidence >= CONF_THRESHOLD:
                pred_queue.append(pred)
                final_pred = max(set(pred_queue), key=pred_queue.count)
                gesture_name = labels.get(final_pred, "Unknown")
            else:
                gesture_name = "Low Confidence"


            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("KNN Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#works with my_env