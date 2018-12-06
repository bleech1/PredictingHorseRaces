# Final project

# Colleen Caveney and Brendan Leech

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from itertools import product
from sklearn.metrics import classification_report
import keras.backend as K
import pandas as pd
import numpy as np
import csv
import math
import tensorflow as tf

DATAFILE = "combined_data.csv"

def CreateDataSets(combinedFilename):
    features = []
    labels = []
    headers = None
    with open(combinedFilename, newline = "") as csvfile:
        csvReader = csv.reader(csvfile, delimiter = ",")
        # First row is headers
        headers = next(csvReader)
        # Only take the headers that we care about (look below for explanation)
        headers = [headers[36]] + headers[49:]

        for row in csvReader:
            if (row[24] not in ["1", "2", "3"]):
                labels.append(0)
            else:
                #labels.append(int(row[24]))
                labels.append(1)

            if row[36] == "":
                row[36] = "30.122"

            row = [row[36]] + row[49:]

            arr = np.array(row)
            #print(arr)
            arr = np.asfarray(arr)
            features.append(np.array(arr))

    matrix = np.vstack(features)




    # Split into train, cv, and test sets
    numTrain = math.floor(matrix.shape[0] * 0.7)
    numCv = math.floor(matrix.shape[0] * 0.15)
    trainData = matrix[:numTrain]
    trainLabels = labels[:numTrain]
    cvData = matrix[numTrain + 1:numTrain + numCv]
    cvLabels = labels[numTrain + 1:numTrain + numCv]
    testData = matrix[numTrain + numCv + 1:]
    testLabels = labels[numTrain + numCv + 1:]
    return trainData, trainLabels, cvData, cvLabels, testData, testLabels, headers

def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

def PrintPrecisionAccuracy(model, cvData, cvLabels):
    y_pred = model.predict_classes(cvData)
    print(classification_report(cvLabels, y_pred))

def w_categorical_crossentropy(y_true, y_pred):
    # Customize weight array to punish for misclassifying 1, 2, or 3 as a 1
    # weights = np.ones((4, 4))
    # weights[1, 0] = 70
    # weights[2, 0] = 70
    # weights[3, 0] = 70
    weights = np.ones((2, 2))
    weights[1, 0] = 70
    weights[0, 0] = 0
    weights[1, 1] = 0

    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def BetOnRaces(model, cvData, cvLabels):
    # Go through each race that has a horse's odds (30.122 is the average
    # that we used to fill in missing values)
    skipCount = 0
    showCount = 0
    noCount = 0

    withoutModelWinnings = 0
    withModelWinnings = 0

    for i in range(len(cvData)):
        if (cvData[i][0] == 30.122):
            skipCount += 1
            continue
        # Run model on all the examples with real odds
        row = np.array([cvData[i]])
        result = model.predict(row)
        # Odds for a show are normally about half of win odds
        showOdds = row[0, 0] / 2
        if result[0] > 0.5:
            showCount += 1
        else:
            noCount += 1

        # Betting strategies
        # Without model: bet $100 on favorites to show
        # With model: bet $100 on all horses we predict will show

        # Is favorite, so w/o model bet 100
        if row[0, 1] == 1:
            if cvLabels[i] == 1:
                withoutModelWinnings += (100 * showOdds)
            else:
                withoutModelWinnings -= 100
        # Predict show, so w/ model bet 100
        #print(result)
        if result[0] > 0.8:
            if cvLabels[i] == 1:
                withModelWinnings += (100 * showOdds)
            else:
                withModelWinnings -= 100

    # print("Skipped:", skipCount, "/", len(cvData))
    # print("Show:", showCount, "/", len(cvData))
    # print("Not show:", noCount, "/", len(cvData))

    print("Without model winnings:", withoutModelWinnings)
    print("With model winnings:", withModelWinnings)



def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

def f1Score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * (p * r) / (p + r)



trainData, trainLabels, cvData, cvLabels, testData, testLabels, headers = CreateDataSets(DATAFILE)
print("created data sets")
numInputFeatures = len(headers)
print("Num input features:", numInputFeatures)


model = Sequential()
model.add(Dense(30, activation = "relu", input_dim = numInputFeatures))
model.add(Dense(1, activation = "sigmoid"))


model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy", precision, recall, f1Score])

#oneHotLabels = to_categorical(trainLabels, num_classes = 2)
# Train the model on the training data
history = model.fit(trainData, trainLabels, epochs = 10, batch_size = 100, verbose = 1)

#oneHotLabels = to_categorical(testLabels, num_classes = 2)
score = model.evaluate(testData, testLabels, batch_size = 100)
print(score)


PrintPrecisionAccuracy(model, cvData, cvLabels)

BetOnRaces(model, cvData, cvLabels)
