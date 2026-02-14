import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


data = pd.read_csv("gesture_data.csv", header=None)

X = data.iloc[:, :-1].values   # 63 features
y = data.iloc[:, -1].values    # label


y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=16)


loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

model.save("gesture_model_ann2.h5")
#works with gesture_ml_env