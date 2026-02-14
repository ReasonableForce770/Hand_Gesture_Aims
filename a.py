
'''

import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    cv2.imshow("Image", img)
    cv2.waitKey(1)



import numpy
print(numpy.__version__)

import tensorflow
print(tensorflow.__version__)

import mediapipe
print("All good")

import numpy
print(numpy.__version__)

import tensorflow
print(tensorflow.__version__)

from tensorflow.keras.models import load_model

model = load_model("")
model.summary()

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("gesture_model_ann.h5")

data = pd.read_csv("gesture_data.csv")
X = data.iloc[:, :-1].values   # features
y = data.iloc[:, -1].values    # labels

predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == y)
print("ANN Accuracy:", accuracy)
'''
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.h5")

img = cv2.imread("img_gesture_dataset/right/your_image.jpg")
img = cv2.resize(img, (64,64))
img = np.expand_dims(img, axis=0)

print(model.predict(img))
