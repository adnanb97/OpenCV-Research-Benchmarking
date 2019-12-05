import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import os
from scipy import optimize
import math

# Get data for plotting from files

dirNameCD = './ResultsCD'
dirNameFPS = './ResultsFPS'
dirNameIOU = './ResultsIOU'
dirNameDataset = './Dataset'
dirNameBB = './TrackerBoundingBoxes'
pathToDSFolder = os.listdir(dirNameDataset)
pathToDSFolder.sort()
pathToCalculatedBBFolder = os.listdir(dirNameBB)
pathToCalculatedBBFolder.sort()


# first function to return tracking benchmark
def center_distance(boxA, boxB):
    # determine the (x, y)-coordinates of the centers of rectangle
    centerAx = boxA[0] + boxA[2] / 2
    centerAy = boxA[1] + boxA[3] / 2
    centerBx = boxB[0] + boxB[2] / 2
    centerBy = boxB[1] + boxB[3] / 2
    xKvadrat = (centerAx - centerBx) * (centerAx - centerBx)
    yKvadrat = (centerAy - centerBy) * (centerAy - centerBy) 
    # compute the distance
    distance = math.sqrt(xKvadrat + yKvadrat)
    
 
    # return the distance between centers
    return distance

# second function to return tracking benchmark
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou
    

def getDataFromTxtFile(boundingBoxFile):
    # get the bounding boxes for the rest of the video so we have a 
    # ground truth to compare it with
    groundTruth = []
    for line in boundingBoxFile.readlines():
        line = line.replace('\t', ',')
        line = line.replace(' ', ',')
        line = line.rstrip() 
        groundTruth.append(line)
    return groundTruth

def addToSum(row1, row2):
    return intersection_over_union((int(row1[0]),int(row1[1]),int(row1[0])+int(row1[2]),int(row1[1])+int(row1[3])), 
                                   (int(row2[0]),int(row2[1]),int(row2[0])+int(row2[2]),int(row2[1])+int(row2[3])))

def plotDataOnGraph(arrayToPlot, arrayOfAverages, className = None, classLength = None):
    x = np.arange(0, 1.1, 0.2)
    y = []
    for i in arrayToPlot: 
        y.append(i)
    # normalize values from 0 to 1
    for i in range(len(y)):
        divisor = y[i][0]
        for j in range(len(y[i])):
            y[i][j] /= divisor

    # remove empty space in plot
    plt.xlim(0)
    plt.ylim(0)

    # add labels 
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if (className == None):
        plt.title('Success plots of OPE')
    else:
        plt.title('Success plots of OPE - ' + className + ' (' + str(classLength) + ')')
    # plot each tracker
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(trackerList)): 
        plt.plot(x, y[i], color[i], label=(trackerList[i] + ' [' + str(round(arrayOfAverages[i], 3)) + ']'))
    plt.legend()
    plt.savefig('./Plots/OPESuccessPlots.eps', format='eps')
    plt.show()

IOUAverage = [[],[],[],[],[],[],[]]
IOU = [[],[],[],[],[],[],[]]
plotList = [[],[],[],[],[],[],[]]
# order is IV SV OCC DEF MB FM IPR OPR OV BC LR
namesOfClasses = ['Illumination Variation', 'Scale Variation', 'Occlusion', 'Deformation', 'Motion Blur', 'Fast Motion', 'In-Plane Rotation', 'Out-of-Plane Rotation', 'Out-of-View', 'Background Clutters', 'Low Resolution']
videoClasses = [['Basketball','Box','Car1','Car2','Car24','Car4','CarDark','Coke','Crowds','David','Doll','FaceOcc2','Fish','Human2','Human4.2','Human7','Human8','Human9','Ironman','KiteSurf','Lemming','Liquor','Man','Matrix','Mhyang','MotorRolling','Shaking','Singer1','Singer2','Skating1','Skiing','Soccer','Sylvester','Tiger1','Tiger2','Trans','Trellis','Woman'],
                ['Biker','BlurBody','BlurCar2','BlurOwl','Board','Box','Boy','Car1','Car24','Car4','CarScale','ClifBar','Couple','Crossing','Dancer','David','Diving','Dog','Dog1','Doll','DragonBaby','Dudek','FleetFace','Freeman1','Freeman3','Freeman4','Girl','Girl2','Gym','Human2','Human3','Human4.2','Human5','Human6','Human7','Human8','Human9','Ironman','Jump','Lemming','Liquor','Matrix','MotorRolling','Panda','RedTeam','Rubik','Shaking','Singer1','Skater','Skater2','Skating1','Skating2.1','Skating2.2','Skiing','Soccer','Surfer','Toy','Trans','Trellis','Twinnings','Vase','Walking','Walking2','Woman'],
                ['Basketball','Biker','Bird2','Bolt','Box','CarScale','ClifBar','Coke','Coupon','David','David3','Doll','DragonBaby','Dudek','FaceOcc1','FaceOcc2','Football','Freeman4','Girl','Girl2','Human3','Human4.2','Human5','Human6','Human7','Ironman','Jogging.1','Jogging.2','Jump','KiteSurf','Lemming','Liquor','Matrix','Panda','RedTeam','Rubik','Singer1','Skating1','Skating2.1','Skating2.2','Soccer','Subway','Suv','Tiger1','Tiger2','Trans','Walking','Walking2','Woman'],
                ['Basketball','Bird1','Bird2','BlurBody','Bolt','Bolt2','Couple','Crossing','Crowds','Dancer','Dancer2','David','David3','Diving','Dog','Dudek','FleetFace','Girl2','Gym','Human3','Human4.2','Human5','Human6','Human7','Human8','Human9','Jogging.1','Jogging.2','Jump','Mhyang','Panda','Singer2','Skater','Skater2','Skating1','Skating2.1','Skating2.2','Skiing','Subway','Tiger1','Tiger2','Trans','Walking','Woman'],
                ['Biker','BlurBody','BlurCar1','BlurCar2','BlurCar3','BlurCar4','BlurFace','BlurOwl','Board','Box','Boy','ClifBar','David','Deer','DragonBaby','FleetFace','Girl2','Human2','Human7','Human9','Ironman','Jump','Jumping','Liquor','MotorRolling','Soccer','Tiger1','Tiger2','Woman'],
                ['Biker','Bird1','Bird2','BlurBody','BlurCar1','BlurCar2','BlurCar3','BlurCar4','BlurFace','BlurOwl','Board','Boy','CarScale','ClifBar','Coke','Couple','Deer','DragonBaby','Dudek','FleetFace','Human6','Human7','Human9','Ironman','Jumping','Lemming','Liquor','Matrix','MotorRolling','Skater2','Skating2.1','Skating2.2','Soccer','Surfer','Tiger1','Tiger2','Toy','Vase','Woman'],
                ['Bird2','BlurBody','BlurFace','BlurOwl','Bolt','Box','Boy','CarScale','ClifBar','Coke','Dancer','David','David2','Deer','Diving','Dog1','Doll','DragonBaby','Dudek','FaceOcc2','FleetFace','Football','Football1','Freeman1','Freeman3','Freeman4','Girl','Gym','Ironman','Jump','KiteSurf','Matrix','MotorRolling','MountainBike','Panda','RedTeam','Rubik','Shaking','Singer2','Skater','Skater2','Skiing','Soccer','Surfer','Suv','Sylvester','Tiger1','Tiger2','Toy','Trellis','Vase'],
                ['Basketball','Biker','Bird2','Board','Bolt','Box','Boy','CarScale','Coke','Couple','Dancer','David','David2','David3','Dog','Dog1','Doll','DragonBaby','Dudek','FaceOcc2','FleetFace','Football','Football1','Freeman1','Freeman3','Freeman4','Girl','Girl2','Gym','Human2','Human3','Human6','Ironman','Jogging.1','Jogging.2','Jump','KiteSurf','Lemming','Liquor','Matrix','Mhyang','MountainBike','Panda','RedTeam','Rubik','Shaking','Singer1','Singer2','Skater','Skater2','Skating1','Skating2.1','Skating2.2','Skiing','Soccer','Surfer','Sylvester','Tiger1','Tiger2','Toy','Trellis','Twinnings','Woman'],
                ['Biker','Bird1','Board','Box','ClifBar','DragonBaby','Dudek','Human6','Ironman','Lemming','Liquor','Panda','Suv','Tiger2'],
                ['Basketball','Board','Bolt2','Box','Car1','Car2','Car24','CarDark','ClifBar','Couple','Coupon','Crossing','Crowds','David3','Deer','Dudek','Football','Football1','Human3','Ironman','Liquor','Matrix','Mhyang','MotorRolling','MountainBike','Shaking','Singer2','Skating1','Soccer','Subway','Trellis'],
                ['Biker','Car1','Freeman3','Freeman4','Panda','RedTeam','Skiing','Surfer','Walking'],
]
trackerList = ['Boosting', 'CSRT', 'KCF', 'MedianFlow', 'MIL', 'MOSSE', 'TLD']
videoName = []
IndexOfClass = 10
classLengthCounter = 0
for x in pathToDSFolder:
    pathToBBFolder = dirNameDataset + '/' + x
    boundingBoxFile = open(pathToBBFolder + '/groundtruth_rect.txt', "r")
    videoName.append(x)
    boostingBB = dirNameBB + '/' + x + 'BBboosting.txt'
    csrtBB = dirNameBB + '/' + x + 'BBcsrt.txt'
    kcfBB = dirNameBB + '/' + x + 'BBkcf.txt'
    medianflowBB = dirNameBB + '/' + x + 'BBmedianflow.txt'
    milBB = dirNameBB + '/' + x + 'BBmil.txt'
    mosseBB = dirNameBB + '/' + x + 'BBmosse.txt'
    tldBB = dirNameBB + '/' + x + 'BBtld.txt'
    if (x in videoClasses[IndexOfClass]): 
        classLengthCounter = classLengthCounter + 1
    boostingFile = open(boostingBB, "r")
    csrtFile = open(csrtBB, "r")
    kcfFile = open(kcfBB, "r")
    medianflowFile = open(medianflowBB, "r")
    milFile = open(milBB, "r")
    mosseFile = open(mosseBB, "r")
    tldFile = open(tldBB, "r")

    groundTruthArray = getDataFromTxtFile(boundingBoxFile)
    boostingBBArray = getDataFromTxtFile(boostingFile)
    csrtBBArray = getDataFromTxtFile(csrtFile)
    kcfBBArray = getDataFromTxtFile(kcfFile)
    medianflowBBArray = getDataFromTxtFile(medianflowFile)
    milBBArray = getDataFromTxtFile(milFile)
    mosseBBArray = getDataFromTxtFile(mosseFile)
    tldBBArray = getDataFromTxtFile(tldFile)
   
    initLen = len(groundTruthArray)
    trackLen = len(boostingBBArray)

    
    sums = [0,0,0,0,0,0,0]  

    for i in range(trackLen - initLen, trackLen): 
        rowOfGroundTruth = groundTruthArray[i - (trackLen - initLen)].split(',')
        rowOfBoosting = boostingBBArray[i].split(',')
        rowOfCsrt = csrtBBArray[i].split(',')
        rowOfKcf = kcfBBArray[i].split(',')
        rowOfMedianFlow = medianflowBBArray[i].split(',')
        rowOfMil = milBBArray[i].split(',')
        rowOfMosse = mosseBBArray[i].split(',')
        rowOfTld = tldBBArray[i].split(',')
        #if (x in videoClasses[IndexOfClass]): 
        sums[0] += addToSum(rowOfGroundTruth, rowOfBoosting)
        sums[1] += addToSum(rowOfGroundTruth, rowOfCsrt)
        sums[2] += addToSum(rowOfGroundTruth, rowOfKcf)
        sums[3] += addToSum(rowOfGroundTruth, rowOfMedianFlow)
        sums[4] += addToSum(rowOfGroundTruth, rowOfMil)
        sums[5] += addToSum(rowOfGroundTruth, rowOfMosse)
        sums[6] += addToSum(rowOfGroundTruth, rowOfTld)

        IOU[0].append(addToSum(rowOfGroundTruth, rowOfBoosting))
        IOU[1].append(addToSum(rowOfGroundTruth, rowOfCsrt))
        IOU[2].append(addToSum(rowOfGroundTruth, rowOfKcf))
        IOU[3].append(addToSum(rowOfGroundTruth, rowOfMedianFlow))
        IOU[4].append(addToSum(rowOfGroundTruth, rowOfMil))
        IOU[5].append(addToSum(rowOfGroundTruth, rowOfMosse))
        IOU[6].append(addToSum(rowOfGroundTruth, rowOfTld))
# get the average for each tracker
# append to IOUAverage
# IOUAverage[i] represents average IOU for each video of tracker trackerList[i]


        for i in range(7): 
            IOUAverage[i].append(sums[i] / trackLen)
    
    boundingBoxFile.close()
    boostingFile.close()
    csrtFile.close()
    kcfFile.close()
    medianflowFile.close()
    milFile.close()
    mosseFile.close()
    tldFile.close()
    
print(len(IOUAverage))
# compute average IOU for each tracker
averagesForAllVideos = [0,0,0,0,0,0,0]
for i in range(len(trackerList)):
    ans = sum(IOUAverage[i])
    averagesForAllVideos[i] = ans / len(IOUAverage[i])

# find Success Plot Rate for each tracker

plottingList = []
for thresh in np.arange(0, 1.01, 0.2): 
    threshold = round(thresh, 1)
    for x in IOU: 
        brojac = 0
        for y in x: 
            if (y >= threshold): 
                brojac += 1
        plottingList.append(brojac)


for i in range(7): 
    for j in range(6):
        #print(plottingList[j * 7 + i])
        plotList[i].append(plottingList[j * 7 + i])
    
#print (str(plotList))

plotDataOnGraph(plotList, averagesForAllVideos) #className=namesOfClasses[IndexOfClass], classLength=classLengthCounter)
