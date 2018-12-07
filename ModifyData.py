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
    print("Adding number of times", featureName, "won")
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
    print("Adding numerical", featureName, "variable")
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

def takeFirst(elem):
    return elem[0]

#if a horse is racing in a distance over 1 mile, and 3/4 or 4/4 of it's previous
#races are <1 mile, horse might not do as well with new dist
def AddNewDistIndicator(filename):
    df = pd.read_csv(filename)
    #keep track for each horse: race distance and it's date
    previousDists = dict()
    for index, row in df.iterrows():
        horseName = str(row["horse"])
        if horseName in previousDists:
            previousDists[horseName].append([str(row["date"]), str(row["distance"])])
        else:
            previousDists[horseName] = [[str(row["date"]), str(row["distance"])]]
    #order by dates
    for element in previousDists:
        previousDists[element].sort(key=takeFirst)
    #iterate again, horse/race combo, find that date, and mark 1 if 3/4 or 4/4 or less than a mile and this dist is > mile
    newDistIndicator = []
    for index, row in df.iterrows():
        horseName = str(row["horse"])
        raceDate = str(row["date"])
        raceDist = str(row["distance"])
        previousRaces = previousDists[horseName];
        mostRecentRaceDists = []
        for element in previousRaces:
            if element[0] < raceDate:
                mostRecentRaceDists.append(element[1])
                if len(mostRecentRaceDists) > 4:
                    mostRecentRaceDists.pop(0)
        # if current race distance is greater than 1 mile
        if raceDist > '1600':
            # keep track of recent races under 1 mile
            underMileDistCount = 0
            for element in mostRecentRaceDists:
                if element < '1600':
                    underMileDistCount += 1
            # if majority of most recent races are under 1 mile
            if underMileDistCount > 2:
                newDistIndicator += [1]
            else:
                newDistIndicator += [0]
        else:
            newDistIndicator += [0]
    df["newDistIndicator"] = pd.Series(newDistIndicator, index=df.index)
    df.to_csv(filename, index=False)


def AddAvgSpeedRating(filename):
    df = pd.read_csv(filename)
    #for each item, calculate time/dist = speed
    #keep track of speeds for specific date/dist/venue combo
    allSpeeds = dict()
    for item, row in df.iterrows():
        finishtime = str(row['finishtime'])
        if(finishtime == "---" or finishtime == "nan" or finishtime == "10"):
            finishtime = "0.0.00"
        finishtime = float(finishtime[:-3])
        speed = finishtime/row['distance']
        identifier = str(row["date"]) + str(row["distance"]) + str(row["venue"])
        if(speed != 0):
            if identifier in allSpeeds:
                allSpeeds[identifier].append(speed)
            else:
                allSpeeds[identifier] = [speed]
    #calculate pars for dist/venue/date with averagetime/dist
    speedRating = []
    for item, row in df.iterrows():
        identifier = str(row["date"]) + str(row["distance"]) + str(row["venue"])
        if(identifier in allSpeeds):
            count = len(allSpeeds[identifier])
            if count > 0:
                parSpeed = sum(allSpeeds[identifier])/count
                finishtime = str(row['finishtime'])
                if(finishtime == "---" or finishtime == "nan" or finishtime == "10"):
                    finishtime = "0.0.00"
                finishtime = float(finishtime[:-3])
                val = finishtime/row['distance']
                speedRating.append(val - parSpeed)
        #if finishtime wasn't available, give speed=parSpeed, so difference = 0
        else:
            speedRating.append(0)
    df["speedRating"] = pd.Series(speedRating, index=df.index)
    df.to_csv(filename, index=False)




print("start")
JoinData(HORSE_INFO_FILE, RESULTS_FILE, BARRIER_FILE, OUTFILE)
print("joined")

print("Adding favorites")
AddIsFavorite(OUTFILE)

print("Adding speed rating")
AddAvgSpeedRating(OUTFILE)

print("Adding new distance indicator")
AddNewDistIndicator(OUTFILE)

# print("Adding numerical horse variable")
# CategoricalFeatureToNum(OUTFILE, "horse")
# print("Adding numerical trainer variable")
# CategoricalFeatureToNum(OUTFILE, "trainer_x")
CategoricalFeatureToNum(OUTFILE, "country")
CategoricalFeatureToNum(OUTFILE, "sire")
CategoricalFeatureToNum(OUTFILE, "sex")
CategoricalFeatureToNum(OUTFILE, "course")

AddNumTimesandPercentWon(OUTFILE, "horse")
AddNumTimesandPercentWon(OUTFILE, "trainer_x")
AddNumTimesandPercentWon(OUTFILE, "jockey")
