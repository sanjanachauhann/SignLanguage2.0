import numpy as np
import os
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

# Load data
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello','listen'])
sequence_length = 20
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = keras.utils.to_categorical(labels).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Model architecture
log_dir = os.path.join('Logs')
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

model = keras.Sequential()
model.add(layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(actions.shape[0], activation='softmax'))

# Compile and train the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# Save the model
model.save('action.keras')

# Load the model
model.load_weights('action.keras')

# Evaluate the model
res = model.predict(X_test)
yhat = np.argmax(model.predict(X_test), axis=1)
ytrue = np.argmax(y_test, axis=1)
multilabel_confusion_matrix(ytrue, yhat)
print('Modelling Completed')