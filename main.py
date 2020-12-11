import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def run_it():

    # dataset
    l_train_labels = []
    l_train_samples = []

    for i in range(50):
        random_young = randint(13, 64)
        l_train_samples.append(random_young)
        l_train_labels.append(1)

        random_old = randint(65, 100)
        l_train_samples.append(random_old)
        l_train_labels.append(0)

    for i in range(950):
        random_young = randint(13, 64)
        l_train_samples.append(random_young)
        l_train_labels.append(0)

        random_old = randint(65, 100)
        l_train_samples.append(random_old)
        l_train_labels.append(1)

        train_labels = np.array(l_train_labels)
        train_samples = np.array(l_train_samples)

    train_labels, train_samples = shuffle(train_labels, train_samples)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1, 1))

    print(scaled_train_sample)

    # model

    model = Sequential([
        Dense(units=16, input_shape=(1,), activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=2, activation='softmax')
    ])

    model.summary()

    # train

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=scaled_train_sample, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

    # inferance
    l_test_labels = []
    l_test_samples = []

    for i in range(50):
        random_young = randint(13, 64)
        l_test_samples.append(random_young)
        l_test_labels.append(1)

        random_old = randint(65, 100)
        l_test_samples.append(random_old)
        l_test_labels.append(0)

    for i in range(950):
        random_young = randint(13, 64)
        l_test_samples.append(random_young)
        l_test_labels.append(0)

        random_old = randint(65, 100)
        l_test_samples.append(random_old)
        l_test_labels.append(1)

        test_labels = np.array(l_test_labels)
        test_samples = np.array(l_test_samples)

    test_labels, test_samples = shuffle(test_labels, test_samples)

    scaled_test_sample = scaler.fit_transform(test_samples.reshape(-1, 1))

    predictions = model.predict(x=scaled_test_sample, batch_size=10, verbose=0)

    j = 0
    for i in predictions:
        print(i)
        print(test_samples[j])
        j=j+1


if __name__ == '__main__':
    run_it()

