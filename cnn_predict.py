import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


model = tf.keras.models.load_model("cnn_model.h5")

class_names = ['close_fist', 'left', 'open_fist', 'right']
img_size = 64


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = "No hand"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * w)
            xmax = int(max(x_coords) * w)
            ymin = int(min(y_coords) * h)
            ymax = int(max(y_coords) * h)

            
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size != 0:
                roi = cv2.resize(hand_img, (img_size, img_size))
               
                roi = np.expand_dims(roi, axis=0)

                cv2.imshow("ROI", roi[0])

                prediction = model.predict(roi, verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                label = f"{class_names[class_index]} ({confidence:.2f})"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, label,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("CNN Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


img = cv2.imread(r"C:\Users\call2\OneDrive\Dtu\SOCITIES\AIMS\hand_gesture\img_gesture_dataset\right\0.jpg")

img = cv2.resize(img, (64,64))
img = img / 255.0
img = np.expand_dims(img, axis=0)
if img is None:
    print("IMAGE NOT LOADED")
print(model.predict(img))
cap.release()
cv2.destroyAllWindows()
