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
BARRIER_FILE = "./horse-racing-dataset-for-experts-hong-kong/barrier.csv"
OUTFILE = "combined_data.csv"


def JoinData(horseInfoFile, resultsFile, barrierFile, outFile):
    # Combine the results and barrier files
    results = pd.read_csv(resultsFile)
    barrier = pd.read_csv(barrierFile)
    merged = pd.concat([results, barrier], axis = 0, ignore_index = True)

    # Join the combined file and the horse info file on the horse's name
    horseInfo = pd.read_csv(horseInfoFile)
    merged = merged.merge(horseInfo, on = "horse")

    # Shuffle the rows in the dataset
    merged = merged.reindex(np.random.permutation(merged.index))

    # Write out to a CSV file
    merged.to_csv(outFile, index = False)

def AddCategoricalFeature(filename, feature):
    df = pd.read_csv(filename)
    featureValues = set()

    # Get all of the unique horse names
    for index, row in df.iterrows():
        #print(row[feature])
        featureValues.add(row[feature])

    featureList = list(featureValues)

    # Make column of values for whether a horse name = given horse name
    count = 1
    for horseName in featureList:
        horseNameCol = []
        print(count, "/", len(featureList))
        for index, row in df.iterrows():
            name = row[feature]
            if (horseName == name):
                horseNameCol += [1]
            else:
                horseNameCol += [0]
        df[feature + horseName] = pd.Series(horseNameCol, index = df.index)
        count += 1
    df.to_csv(filename, index = False)

def AddNumberOpponents(filename):
    df = pd.read_csv(filename)

    # For each horse's race,
    length = len(df)
    numOpponents = dict()
    for index, row in df.iterrows():
        # Find the number of opponents for the horse
        identifier = str(row["date"]) + str(row["raceno"]) + str(row["venue"])
        if identifier in numOpponents:
            numOpponents[identifier] += 1
        else:
            numOpponents[identifier] = 1
    horseNameCol = []
    for index, row in df.iterrows():
        identifier = str(row["date"]) + str(row["raceno"]) + str(row["venue"])
        horseNameCol += [numOpponents[identifier]]
    # Add the column to the dataset
    df["numOpponents"] = pd.Series(horseNameCol, index = df.index)
    df.to_csv(filename, index = False)

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
        #headers = [headers[10]] + [headers[25]] + [headers[49]]
        headers = [headers[1]] + [headers[3]] + [headers[43]] + [headers[44]] + [headers[45]] + [headers[46]] + [headers[47]] + [headers[48]] + [headers[49]]
        #headers = [headers[1]] + headers[3:]
        for row in csvReader:
            # 0th element is an index value we can ignore

            # Just choosing 2 float values for our parameters at the beginning
            # Skip if the value doesn't exist

            # 2nd element is the place (our label)
            # We are classifying as first, second, third, or worse places
            # So worse than third place will be represented as 0
            if (row[24] not in ["1", "2", "3"]):
                labels.append(0)
            else:
                labels.append(int(row[24]))

            #row = [row[10]] + [row[25]] + [row[49]]
            row = [row[1]] + [row[3]] + [row[43]] + [row[44]] + [row[45]] + [row[46]] + [row[47]] + [row[48]] + [row[49]]

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
    history = model.fit(trainData, oneHotLabels, epochs = 10, batch_size = 100, verbose = 1)
    return model, history

def EvaluateModel(model, testData, testLabels):
    # Evaluate the model on the testing set
    oneHotLabels = to_categorical(testLabels, num_classes = 4)
    score = model.evaluate(testData, oneHotLabels)
    print(score)

print("start")
JoinData(HORSE_INFO_FILE, RESULTS_FILE, BARRIER_FILE, OUTFILE)
print("joined")
# AddCategoricalFeature(OUTFILE, "horse")

# Get the other horses in the race
AddNumberOpponents(OUTFILE)


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
model, history = TrainModel(model, trainData, trainLabels)

print(history)

print("trained model")
EvaluateModel(model, cvData, cvLabels)
print("evaluated")
