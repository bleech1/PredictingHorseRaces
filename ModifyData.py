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

#helper function
def takeFirst(elem):
    return elem[0]

#This is an indicator for a horse racing at a distance longer than it is used to
#from the past few races. Could have a negative impact on horse's outcome.
#Adds a column with value 1 if the horse is racing in a race with distance over 1 mile,
#and the majority of its most recent races (at most 4 previous races) are less than 1 mile.
#Has value 0 otherwise.
def AddNewDistIndicator(filename):
    df = pd.read_csv(filename)
    #keep track for each horse: race distance and that race's date
    previousDists = dict()
    for index, row in df.iterrows():
        horseName = str(row["horse"])
        if horseName in previousDists:
            previousDists[horseName].append([str(row["date"]), str(row["distance"])])
        else:
            previousDists[horseName] = [[str(row["date"]), str(row["distance"])]]
    #order each horse's races by dates
    for element in previousDists:
        previousDists[element].sort(key=takeFirst)
    #iterate dataset again
    #for each horse/race combo, if the distance is >1 mile, find that date in previousDists
    #create a list of the 4 most recent races
    #mark 1 if 3/4 or 4/4 or less than a mile, 0 otherwise
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


#Calculates an average speed rating for each horse/race combo given the horse's previous races
#Speed rating = difference between horse's speed in that specific race and the par speed for that course
#Par speed = average speed of horses on particular date on a particular course (unique date, dist, venue)
#A horse's speed rating will be positive if they raced faster than the average of all horses on that course on that day
def AddAvgSpeedRating(filename):
    df = pd.read_csv(filename)
    #for each item, calculate speed = time/dist
    #keep track of all speeds for specific date/dist/venue combo
    allSpeeds = dict()
    for item, row in df.iterrows():
        #clean the data for finishtime
        finishtime = str(row['finishtime'])
        if(finishtime == "---" or finishtime == "nan" or finishtime == "10"):
            finishtime = "0.0.00"
        finishtime = float(finishtime[:-3])
        #calculate speed
        speed = finishtime/row['distance']
        identifier = str(row["date"]) + str(row["distance"]) + str(row["venue"])
        if(speed != 0):
            #append speed to array for the specific date/dist/venue combo
            if identifier in allSpeeds:
                allSpeeds[identifier].append(speed)
            else:
                allSpeeds[identifier] = [speed]
    #calculate pars for dist/venue/date with average time/dist
    #calculate speed rating for each horse/race combo by (speed - parSpeed for that date/course)
    speedRating = []
    for item, row in df.iterrows():
        identifier = str(row["date"]) + str(row["distance"]) + str(row["venue"])
        if(identifier in allSpeeds):
            count = len(allSpeeds[identifier])
            if count > 0:
                #calculate par speed
                parSpeed = sum(allSpeeds[identifier])/count
                #clean finishtime data
                finishtime = str(row['finishtime'])
                if(finishtime == "---" or finishtime == "nan" or finishtime == "10"):
                    finishtime = "0.0.00"
                finishtime = float(finishtime[:-3])
                #calculate speed for this horse/race combo
                val = finishtime/row['distance']
                #append difference in par and speed to speedRating array
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
