# %matplotlib inline

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

df = pd.read_csv('combined_classes.csv')
x = np.array(df[["p1", "p2"]])
y_soft = np.array(df[["soft_a", "soft_b", "soft_c", "soft_d"]])
y_hard = np.array(df[["hard_a", "hard_b", "hard_c", "hard_d"]])
y_all = np.append(y_soft, y_hard, axis=1)
# y_hard = np.array(df[["hard_classes"]])
# Y_hard = np_utils.to_categorical(y_hard, 4)

X_train, X_test, y_train, y_test = train_test_split(x, y_all, test_size=0.2, random_state=42)
y_train_soft = y_train[:, 0:4]
y_train_hard = y_train[:, 4:8]
y_test_soft = y_test[:, 0:4]
y_test_hard = y_test[:, 4:8]

epochs = range(10, 150, 10)
batch_size = 200

train_on_hard_evaluate_on_soft = np.array([])
train_on_soft_evaluate_on_soft = np.array([])
train_on_hard_evaluate_on_hard = np.array([])
train_on_soft_evaluate_on_hard = np.array([])

model = Sequential()
model.add(Dense(output_dim=10, input_dim=2, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(output_dim=10, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(output_dim=10, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(output_dim=4, init='uniform'))
model.add(Activation('softmax'))

model.summary()

# sgd = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

with open('stats_epochs.csv', 'wb') as f:
    writer = csv.DictWriter(f, fieldnames=["Epochs", "Train on", "Evaluate on Hard", "Evaluate on Soft"], delimiter=',')
    writer.writeheader()
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    for e in epochs:
        for i in xrange(2):
            if i == 0:
                paradigm = "Hard"
                paradigm_train = y_train_hard
                paradigm_test = y_test_hard
            else:
                paradigm = "Soft"
                paradigm_train = y_train_soft
                paradigm_test = y_test_soft

            model.fit(X_train, paradigm_train,
                      nb_epoch=e,
                      batch_size=batch_size,
                      verbose=2,
                      validation_data=(X_test, paradigm_test))

            score_on_hard, acc_on_hard = model.evaluate(X_test, y_test_hard, batch_size=batch_size)
            score_on_soft, acc_on_soft = model.evaluate(X_test, y_test_soft, batch_size=batch_size)

            if i == 0:
                #Train on hard labels, evaluate on soft labels
                train_on_hard_evaluate_on_soft = np.append(train_on_hard_evaluate_on_soft , score_on_soft)
                #Train on hard labels, evaluate on hard labels
                train_on_hard_evaluate_on_hard = np.append(train_on_hard_evaluate_on_hard , score_on_hard)
            else:
                #Train on soft labels, evaluate on soft labels
                train_on_soft_evaluate_on_soft = np.append(train_on_soft_evaluate_on_soft, score_on_soft)
                #Train on soft labels, evaluate on hard labels
                train_on_soft_evaluate_on_hard = np.append(train_on_soft_evaluate_on_hard, score_on_hard)

            writer.writerow([e, paradigm, score_on_hard, score_on_soft])

