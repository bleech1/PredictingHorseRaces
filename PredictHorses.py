# Final project

# Colleen Caveney and Brendan Leech

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import keras.backend as K
import pandas as pd
import numpy as np
import csv
import math

HORSE_INFO_FILE = "./horse-racing-dataset-for-experts-hong-kong/horse_info.csv"
RESULTS_FILE = "./horse-racing-dataset-for-experts-hong-kong/results.csv"
OUTFILE = "combined_data.csv"


def JoinData(horseInfoFile, resultsFile, outFile):
    # Join the results file and the horse info file on the horse's name
    horseInfo = pd.read_csv(horseInfoFile)
    results = pd.read_csv(resultsFile)
    merged = results.merge(horseInfo, on = "horse")
    merged.to_csv(outFile, index = False)

def CategoricalFeatureToNum(filename, featureName):
    categoricalValues = []
    numericalValues = []
    indexHorse = 0
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        if(row[featureName] in categoricalValues):
            numericalValues.append(categoricalValues.index(row[featureName]))
        else:
            categoricalValues.append(row[featureName])
            numericalValues.append(indexHorse)
            indexHorse += 1

    df[featureName + "Numerical"] = pd.Series(numericalValues, index=df.index)
    df.to_csv(filename, index=False)


def CreateDataSets(combinedFilename):
    features = []
    labels = []
    headers = None
    with open(combinedFilename, newline = "") as csvfile:
        csvReader = csv.reader(csvfile, delimiter = ",")
        # First row is headers
        headers = next(csvReader)
        # Only take the headers that we care about (look below for explanation)
        headers = [headers[1]] + [headers[3]] + [headers[43]] + [headers[44]] + [headers[45]] + [headers[46]] + [headers[47]] + [headers[48]] + [headers[49]]
        #headers = [headers[1]] + headers[3:]
        for row in csvReader:
            # 0th element is an index value we can ignore

            #row = [row[1]] + row[3:]
            # Just choosing 2 float values for our parameters at the beginning
            # Skip if the value doesn't exist
            if (row[1] != "" and row[3] != ""):
                # 2nd element is the place (our label)
                # We are classifying as first, second, third, or worse places
                # So worse than third place will be represented as 0
                if (row[2] not in ["1", "2", "3"]):
                    labels.append(0)
                else:
                    labels.append(row[2])

                row = [row[1]] + [row[3]] + [row[43]] + [row[44]] + [row[45]] + [row[46]] + [row[47]] + [row[48]] + [row[49]]
                arr = np.array(row)
                #print(arr)
                arr = np.asfarray(arr)
                features.append(np.array(arr))


    # Use vstack to make into a matrix that keras can use
    matrix = features[0]
    count = 1
    for row in features[1:]:
        #print(count)
        matrix = np.vstack((matrix, row))
        count += 1

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

def CreateModel(numInputFeatures):
    # Create a model
    model = Sequential()
    model.add(Dense(50, activation = "relu", input_dim = numInputFeatures))
    model.add(Dense(4, activation = "sigmoid"))
    return model

def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

def CompileModel(model):
    # Compile the model
    model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = [get_categorical_accuracy_keras])
    return model

def TrainModel(model, trainData, trainLabels):
    # Turn the labels into one hot labels
    oneHotLabels = to_categorical(trainLabels, num_classes = 4)
    # Train the model on the training data
    model.fit(trainData, oneHotLabels, epochs = 10, batch_size = 100)
    return model

def EvaluateModel(model, testData, testLabels):
    # Evaluate the model on the testing set
    oneHotLabels = to_categorical(testLabels, num_classes = 4)
    score = model.evaluate(testData, oneHotLabels)
    print(score)


print("start")
JoinData(HORSE_INFO_FILE, RESULTS_FILE, OUTFILE)
print("joined")
CategoricalFeatureToNum(OUTFILE, "horse")
CategoricalFeatureToNum(OUTFILE, "jockey")
CategoricalFeatureToNum(OUTFILE, "trainer_x")
CategoricalFeatureToNum(OUTFILE, "venue")
CategoricalFeatureToNum(OUTFILE, "country")
CategoricalFeatureToNum(OUTFILE, "sire")
CategoricalFeatureToNum(OUTFILE, "dam")
trainData, trainLabels, cvData, cvLabels, testData, testLabels, headers = CreateDataSets(OUTFILE)
print("created data sets")
numInputFeatures = len(headers)
print(numInputFeatures)
model = CreateModel(numInputFeatures)
print("created model")
model = CompileModel(model)
print("compiled model")
model = TrainModel(model, trainData, trainLabels)
print("trained model")
EvaluateModel(model, cvData, cvLabels)
print("evaluated")
