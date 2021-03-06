import keras.callbacks
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

class epochendCallBack(keras.callbacks.Callback):
    def __init__(self, p_dict):
        self.p_dict = p_dict
        self.train_on_hard_evaluate_on_soft = np.array([])
        self.train_on_soft_evaluate_on_soft = np.array([])
        self.train_on_hard_evaluate_on_hard = np.array([])
        self.train_on_soft_evaluate_on_hard = np.array([])
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        paradigm = self.p_dict["paradigm"]
        test_x = self.p_dict["X test"]
        test_y1 = self.p_dict["paradigm_test1"]
        test_y2 = self.p_dict["paradigm_test2"]
        loss1, acc1 = self.model.evaluate(test_x, test_y1, verbose=0)
        loss2, acc2 = self.model.evaluate(test_x, test_y2, verbose=0)

        if paradigm == "Hard":
            self.train_on_hard_evaluate_on_hard = np.append(self.train_on_hard_evaluate_on_hard, loss1)
            self.train_on_hard_evaluate_on_soft = np.append(self.train_on_hard_evaluate_on_soft, loss2)
        elif paradigm == "Soft":
            self.train_on_soft_evaluate_on_hard = np.append(self.train_on_soft_evaluate_on_hard, loss1)
            self.train_on_soft_evaluate_on_soft = np.append(self.train_on_soft_evaluate_on_soft, loss2)


df = pd.read_csv('combined_classes2.csv')
x = np.array(df[["p1", "p2"]])
x_ = np.array(df[["l1", "l2"]])
x_all = np.append(x, x_, axis=1)
y_soft = np.array(df[["soft_a", "soft_b", "soft_c", "soft_d"]])
# y_hard = np.array(df[["hard_a", "hard_b", "hard_c", "hard_d"]])
y_hard = np.array(df[["hard_classes"]])
Y_hard = np_utils.to_categorical(y_hard, 4)
y_all = np.append(y_soft, Y_hard, axis=1)

X_train_all, X_test_all, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
X_train = X_train_all[:,0:2]
X_test = X_test_all[:,0:2]
X_train_h1h2 = X_train_all[:,2:4]
X_test_h1h2 = X_test_all[:,2:4]
y_train_soft = y_train[:, 0:4]
y_train_hard = y_train[:, 4:8]
y_test_soft = y_test[:, 0:4]
y_test_hard = y_test[:, 4:8]

epochs = 100
batch_size = 160000

model = Sequential()
model.add(Dense(output_dim=20, input_dim=2, init='uniform'))
model_initial = model.get_weights()

model.add(Dense(output_dim=4, init='uniform'))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.5, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

"""
Hard Labels
"""
paradigm_dict = {}
paradigm_dict = {
    "paradigm": "Hard",
    "paradigm_train": y_train_hard,
    "X test": X_test,
    "paradigm_test1": y_test_hard,
    "paradigm_test2": y_test_soft
}
test_callback = epochendCallBack(paradigm_dict)
hard_fit = model.fit(X_train, paradigm_dict["paradigm_train"],
                  nb_epoch=epochs,
                  batch_size=batch_size,
                  verbose=1,
                  callbacks=[test_callback])

train_on_hard_loss = hard_fit.history['loss']
train_on_hard_batch_loss = test_callback.batch_losses
train_on_hard_evaluate_on_hard = test_callback.train_on_hard_evaluate_on_hard
train_on_hard_evaluate_on_soft = test_callback.train_on_hard_evaluate_on_soft

"""
Soft Labels
"""
paradigm_dict = {
    "paradigm": "Soft",
    "paradigm_train": y_train_soft,
    "X test": X_test,
    "paradigm_test1": y_test_hard,
    "paradigm_test2": y_test_soft
    }
test_callback = epochendCallBack(paradigm_dict)
soft_fit = model.fit(X_train, paradigm_dict["paradigm_train"],
                     nb_epoch=epochs,
                     batch_size=batch_size,
                     verbose=1,
                     callbacks=[test_callback])

train_on_soft_loss = soft_fit.history['loss']
train_on_soft_batch_loss = test_callback.batch_losses
train_on_soft_evaluate_on_hard = test_callback.train_on_soft_evaluate_on_hard
train_on_soft_evaluate_on_soft = test_callback.train_on_soft_evaluate_on_soft