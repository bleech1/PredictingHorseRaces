# Final project

# Colleen Caveney and Brendan Leech

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.metrics import classification_report
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
        feature = str(row[featureName])
        if (len(feature) != 0):
            feature = feature.lower()
        if(feature in categoricalValues):
            numericalValues.append(categoricalValues.index(feature))
        else:
            categoricalValues.append(feature)
            numericalValues.append(indexHorse)
            indexHorse += 1

    print(len(set(numericalValues)))

    df[featureName + "Numerical"] = pd.Series(numericalValues, index=df.index)
    df.to_csv(filename, index=False)

def AddNumTimesWon(filename, featureName):
    df = pd.read_csv(filename)
    values = []

    numTimesWon = dict()
    # Go through dataset and find how many times each has won
    for index, row in df.iterrows():
        feature = str(row[featureName])
        if (row["plc"] == "1"):
            if feature in numTimesWon:
                numTimesWon[feature] += 1
            else:
                numTimesWon[feature] = 1
    # Fill in feature array with how many times it has won
    for index, row in df.iterrows():
        feature = str(row[featureName])
        if (feature in numTimesWon):
            values += [numTimesWon[feature]]
        else:
            values += [0]


    df[featureName + "WinTimes"] = pd.Series(values, index=df.index)
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
        headers = [headers[10]] + [headers[25]] + [headers[36]] + headers[49:] + [headers[24]]

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

            # If empty, fill in winodds with 30.122 because that is the mean
            # of the winodds for the horses that do have odds
            if (row[36] == ""):
                row[36] = "30.122"

            if (row[24] == "" or not (row[24][0].isdigit() and row[24][-1].isdigit())):
                row[24] = "4"

            row = [row[10]] + [row[25]] + [row[36]] + row[49:] + [row[24]]

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
    model.add(Dense(10, activation = "relu", input_dim = numInputFeatures))
    model.add(Dense(4, activation = "softmax"))
    return model

def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

def CompileModel(model):
    # Compile the model
    model.compile(optimizer = "rmsprop", loss = "mean_squared_error", metrics = [get_categorical_accuracy_keras])
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


def PrintPrecisionAccuracy(model, testData, testLabels):
    y_pred = model.predict_classes(testData)
    print(classification_report(testLabels, y_pred))


def BetOnRaces(model, cvData, cvLabels):
    # Go through each race that has a horse's odds (30.122 is the average
    # that we used to fill in missing values)
    for row in cvData:
        if (row[2] == 30.122):
            continue
        # Run model on all the examples with real odds
        row = np.array([row])
        result = model.predict(row)
        #print(result)


print("start")
JoinData(HORSE_INFO_FILE, RESULTS_FILE, BARRIER_FILE, OUTFILE)
print("joined")
# AddCategoricalFeature(OUTFILE, "horse")

# Get the other horses in the race
AddNumberOpponents(OUTFILE)

# Convert string features to numbers
print("Adding numerical horse variable")
CategoricalFeatureToNum(OUTFILE, "horse")
print("Adding numerical trainer variable")
CategoricalFeatureToNum(OUTFILE, "trainer_x")
print("Adding numerical country variable")
CategoricalFeatureToNum(OUTFILE, "country")
print("Adding numerical sire variable")
CategoricalFeatureToNum(OUTFILE, "sire")
# print("Adding numerical dam variable")
# CategoricalFeatureToNum(OUTFILE, "dam")
# print("Adding numerical owner variable")
# CategoricalFeatureToNum(OUTFILE, "owner")
print("Adding numerical sex variable")
CategoricalFeatureToNum(OUTFILE, "sex")
print("Adding numerical course variable")
CategoricalFeatureToNum(OUTFILE, "course")
# print("Adding numerical date variable")
# CategoricalFeatureToNum(OUTFILE, "date")
print("Adding numerical going variable")
CategoricalFeatureToNum(OUTFILE, "going")
print("Adding numerical venue variable")
CategoricalFeatureToNum(OUTFILE, "venue")
print("Adding numerical jockey variable")
CategoricalFeatureToNum(OUTFILE, "jockey")
print("Adding numerical import variable")
CategoricalFeatureToNum(OUTFILE, "import_type")

# Add features for number of times horse/trainer/jockey has won
AddNumTimesWon(OUTFILE, "horse")
AddNumTimesWon(OUTFILE, "trainer_x")
AddNumTimesWon(OUTFILE, "jockey")


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

print("predictions made")
PrintPrecisionAccuracy(model, cvData, cvLabels)


BetOnRaces(model, cvData, cvLabels)
