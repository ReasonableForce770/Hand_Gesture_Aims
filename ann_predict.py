import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque


model = load_model("gesture_model_ann2.h5")


CONF_THRESHOLD = 0.75


labels = {
    0: "Open Fist",
    1: "Closed Fist",
    2: "Peace Sign",
    3: "Thumbs Up",
    4: "Ok Sign"
}


pred_queue = deque(maxlen=5)

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

            
            input_data = np.array([landmark_list])

            
            prediction = model.predict(input_data, verbose=0)
            confidence = np.max(prediction)
            pred = np.argmax(prediction)

            if confidence >= CONF_THRESHOLD:
                pred_queue.append(pred)
                final_pred = max(set(pred_queue), key=pred_queue.count)
                gesture_name = labels.get(final_pred, "Unknown")
            else:
                gesture_name = "Low Confidence"

           
            cv2.putText(frame, gesture_name,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            cv2.putText(frame, f"Conf: {confidence:.2f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ANN Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#works in cnn_env