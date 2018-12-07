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
            # If the horse did not finish in top 3, label 0
            if (row[24] not in ["1", "2", "3"]):
                labels.append(0)
            else:
                # If finished in top 3, label 1
                labels.append(1)

            # column 36 is the betting odds
            if row[36] == "":
                # If horse did not have odds, then make the odds 30.122, which
                # if the average for horses that did have odds
                row[36] = "30.122"

            # Everything from 49 on are the features that we added to the model
            row = [row[36]] + row[49:]

            arr = np.array(row)
            arr = np.asfarray(arr)
            features.append(np.array(arr))

    matrix = np.vstack(features)

    # Split into train, cv, and test sets (70/15/15 split)
    numTrain = math.floor(matrix.shape[0] * 0.7)
    numCv = math.floor(matrix.shape[0] * 0.15)
    trainData = matrix[:numTrain]
    trainLabels = labels[:numTrain]
    cvData = matrix[numTrain + 1:numTrain + numCv]
    cvLabels = labels[numTrain + 1:numTrain + numCv]
    testData = matrix[numTrain + numCv + 1:]
    testLabels = labels[numTrain + numCv + 1:]
    return trainData, trainLabels, cvData, cvLabels, testData, testLabels, headers

def PrintPrecisionAccuracy(model, cvData, cvLabels):
    y_pred = model.predict_classes(cvData)
    print(classification_report(cvLabels, y_pred))

def BetOnRaces(model, cvData, cvLabels):
    withoutModelWinnings = 0
    withModelWinnings = 0

    # Go through each example from the cv set
    for i in range(len(cvData)):
        # Skip the horses that did not originally have odds
        if (cvData[i][0] == 30.122):
            continue
        # Run model on all the examples with real odds
        row = np.array([cvData[i]])
        result = model.predict(row)
        # Odds for a show are roughly half of win odds
        showOdds = row[0, 0] / 2

        # Betting strategies
        # Without model: bet $100 on favorites to show
        # With model: bet $100 on all horses we predict with at least probability
        # 90% will show

        # Is favorite, so w/o model bet 100
        if row[0, 1] == 1:
            if cvLabels[i] == 1:
                withoutModelWinnings += (100 * showOdds)
            else:
                withoutModelWinnings -= 100
        # Predict show, so w/ model bet 100
        if result[0] > 0.9:
            if cvLabels[i] == 1:
                withModelWinnings += (100 * showOdds)
            else:
                withModelWinnings -= 100

    print("Without model winnings:", withoutModelWinnings)
    print("With model winnings:", withModelWinnings)


# From: https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras
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
    # Calcuates the F1 score from precision and accuracy
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * (p * r) / (p + r)


# Split the datasets
trainData, trainLabels, cvData, cvLabels, testData, testLabels, headers = CreateDataSets(DATAFILE)
print("created data sets")
numInputFeatures = len(headers)
print("Num input features:", numInputFeatures)

# Define the model
model = Sequential()
model.add(Dense(30, activation = "relu", input_dim = numInputFeatures))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy", precision, recall, f1Score])

# Train the model on the training data
history = model.fit(trainData, trainLabels, epochs = 100, batch_size = 100, verbose = 1)

# Evaluate and print out testing data
score = model.evaluate(testData, testLabels, batch_size = 100)
print(score)

PrintPrecisionAccuracy(model, cvData, cvLabels)

BetOnRaces(model, cvData, cvLabels)
