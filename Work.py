# Final project

# Colleen Caveney and Brendan Leech

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from itertools import product
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



def AddIsFavorite(filename):
    df = pd.read_csv(filename)
    bestOdds = dict()
    for index, row in df.iterrows():
        # Find the best odds for each specific race
        identifier = str(row["date"]) + str(row["raceno"]) + str(row["venue"])
        # If this horse has better odds, then update the best odds for the race
        if identifier in bestOdds:
            if bestOdds[identifier] > row["winodds"]:
                bestOdds[identifier] = row["winodds"]
        else:
            bestOdds[identifier] = row["winodds"]
    favoriteValues = []
    for index, row in df.iterrows():
        identifier = str(row["date"]) + str(row["raceno"]) + str(row["venue"])
        if row["winodds"] == bestOdds[identifier]:
            favoriteValues += [1]
        else:
            favoriteValues += [0]

    df["isFavorite"] = pd.Series(favoriteValues, index=df.index)
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

def PrintPrecisionAccuracy(model, testData, testLabels):
    y_pred = model.predict_classes(testData)
    print(classification_report(testLabels, y_pred))

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
                withoutModelWinnings -= (100 * showOdds)
        # Predict show, so w/ model bet 100
        if result[0] > 0.8:
            if cvLabels[i] == 1:
                withModelWinnings += (100 * showOdds)
            else:
                withModelWinnings -= (100 * showOdds)

    # print("Skipped:", skipCount, "/", len(cvData))
    # print("Show:", showCount, "/", len(cvData))
    # print("Not show:", noCount, "/", len(cvData))

    print("Without model winnings:", withoutModelWinnings)
    print("With model winnings:", withModelWinnings)



print("start")
JoinData(HORSE_INFO_FILE, RESULTS_FILE, BARRIER_FILE, OUTFILE)
print("joined")

print("Adding favorites")
AddIsFavorite(OUTFILE)

# print("Adding numerical horse variable")
# CategoricalFeatureToNum(OUTFILE, "horse")
# print("Adding numerical trainer variable")
# CategoricalFeatureToNum(OUTFILE, "trainer_x")
print("Adding numerical country variable")
CategoricalFeatureToNum(OUTFILE, "country")
print("Adding numerical sire variable")
CategoricalFeatureToNum(OUTFILE, "sire")
print("Adding numerical sex variable")
CategoricalFeatureToNum(OUTFILE, "sex")
print("Adding numerical course variable")
CategoricalFeatureToNum(OUTFILE, "course")

AddNumTimesWon(OUTFILE, "horse")
AddNumTimesWon(OUTFILE, "trainer_x")
AddNumTimesWon(OUTFILE, "jockey")

trainData, trainLabels, cvData, cvLabels, testData, testLabels, headers = CreateDataSets(OUTFILE)
print("created data sets")
numInputFeatures = len(headers)
print("Num input features:", numInputFeatures)

model = Sequential()
model.add(Dense(30, activation = "relu", input_dim = numInputFeatures))
model.add(Dense(1, activation = "sigmoid"))


model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = [get_categorical_accuracy_keras, "accuracy"])

#oneHotLabels = to_categorical(trainLabels, num_classes = 2)
# Train the model on the training data
history = model.fit(trainData, trainLabels, epochs = 10, batch_size = 100, verbose = 1)

#oneHotLabels = to_categorical(testLabels, num_classes = 2)
score = model.evaluate(testData, testLabels, batch_size = 100)
print(score)


PrintPrecisionAccuracy(model, cvData, cvLabels)

BetOnRaces(model, cvData, cvLabels)
