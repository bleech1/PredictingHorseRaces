# Final project

# Prepare data file

# Colleen Caveney and Brendan Leech

import pandas as pd
import numpy as np
import csv
import math
import tensorflow as tf
import sys

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


def AddNumTimesandPercentWon(filename, featureName):
    df = pd.read_csv(filename)
    values = []
    percentages = []

    numTimesWon = dict()
    numTimesRaced = dict()
    # Go through dataset and find how many times each has won
    for index, row in df.iterrows():
        feature = str(row[featureName])
        if (row["plc"] == "1"):
            if feature in numTimesWon:
                numTimesWon[feature] += 1
            else:
                numTimesWon[feature] = 1
        if feature in numTimesRaced:
            numTimesRaced[feature] += 1
        else:
            numTimesRaced[feature] = 1
    # Fill in feature array with how many times it has won
    for index, row in df.iterrows():
        feature = str(row[featureName])
        if (feature in numTimesWon):
            values += [numTimesWon[feature]]
            if (feature in numTimesRaced):
                percentages += [str(int(numTimesWon[feature])/int(numTimesRaced[feature]))]
        else:
            values += [0]
            percentages += [0]

    df[featureName + "WinTimes"] = pd.Series(values, index=df.index)
    df[featureName + "WinPercentage"] = pd.Series(percentages, index=df.index)
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

AddNumTimesandPercentWon(OUTFILE, "horse")
AddNumTimesandPercentWon(OUTFILE, "trainer_x")
AddNumTimesandPercentWon(OUTFILE, "jockey")
