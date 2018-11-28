# Final project

# Colleen Caveney and Brendan Leech

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import csv
import math

HORSE_INFO_FILE = "./horse-racing-dataset-for-experts-hong-kong/horse_info.csv"
RESULTS_FILE = "./horse-racing-dataset-for-experts-hong-kong/results.csv"
OUTFILE = "combined_data.csv"


def JoinData(horseInfoFile, resultsFile, outFile):
    horseInfo = pd.read_csv(horseInfoFile)
    results = pd.read_csv(resultsFile)
    merged = results.merge(horseInfo, on = "horse")
    merged.to_csv(outFile, index = False)

def CreateDataSets(combinedFilename):
    features = []
    labels = []
    headers = None
    with open(combinedFilename, newline = "") as csvfile:
        csvReader = csv.reader(csvfile, delimiter = ",")
        # First row is headers
        headers = next(csvReader)
        # Only take the headers that we care about (look below for explanation)
        headers = [headers[1]] + headers[3:]
        for row in csvReader:
            # 0th element is an index value we can ignore

            # 2nd element is the place (our label)
            # We are classifying as first, second, third, or worse places
            # So worse than third place will be represented as 0
            if (row[2] not in [1, 2, 3]):
                labels.append(0)
            else:
                lables.append(row[2])
            row = [row[1]] + row[3:]
            arr = np.array(row)
            features.append(np.array(arr))

    matrix = features[0]
    count = 1
    for row in features[1:]:
        print(count)
        matrix = np.vstack((matrix, row))
        count += 1

    numRows = math.floor(len(features) * 0.8)
    trainData = matrix[:numRows]
    trainLabels = labels[:numRows]
    testData = matrix[numRows + 1:]
    testLabels = labels[numRows + 1:]
    return trainData, trainLabels, testData, testLabels, headers

def CreateModel(numInputFeatures):
    model = Sequential()
    model.add(Dense(50, activation = "relu", input_dim = numInputFeatures))
    model.add(Dense(4, activation = "sigmoid"))
    return model

def CompileModel(model):
    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def TrainModel(model, trainData, trainLabels):
    oneHotLabels = to_categorical(trainLabels, num_classes = 4)
    model.fit(trainData, oneHotLabels, epochs = 10, batch_size = 100)
    return model

def EvaluateModel(model, testData, testLabels):
    score = model.evaluate(testData, testLabels)
    print(score)


print("hi")
JoinData(HORSE_INFO_FILE, RESULTS_FILE, OUTFILE)
print("hi")
trainData, trainLabels, testData, testLabels, headers = CreateDataSets(OUTFILE)
print("hi")
numInputFeatures = len(headers)
print(numInputFeatures)
model = CreateModel(numInputFeatures)
print("hi")
model = CompileModel(model)
print("hi")
model = TrainModel(model, trainData, trainLabels)
print("hi")
# EvaluateModel(model, testData, testLabels)
