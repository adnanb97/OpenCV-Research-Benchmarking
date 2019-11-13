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

def test_func(x, a, c, d):
    return a*np.exp(-c*x)+d
def plotDataOnGraph(s):
    # Data for plotting
    t = np.arange(0.0, 1.01, 0.2)


    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='Overlap Threshold', ylabel='Success Rate', title='Success Plots of OPE')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()
    
IOUAverage = [[],[],[],[],[],[],[]]
IOU = [[],[],[],[],[],[],[]]
plotList = [[],[],[],[],[],[],[]]

for x in pathToDSFolder:
    pathToBBFolder = dirNameDataset + '/' + x
    boundingBoxFile = open(pathToBBFolder + '/groundtruth_rect.txt', "r")
    

    boostingBB = dirNameBB + '/' + x + 'BBboosting.txt'
    csrtBB = dirNameBB + '/' + x + 'BBcsrt.txt'
    kcfBB = dirNameBB + '/' + x + 'BBkcf.txt'
    medianflowBB = dirNameBB + '/' + x + 'BBmedianflow.txt'
    milBB = dirNameBB + '/' + x + 'BBmil.txt'
    mosseBB = dirNameBB + '/' + x + 'BBmosse.txt'
    tldBB = dirNameBB + '/' + x + 'BBtld.txt'

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
    
    '''
    print(len(groundTruthArray))
    print(len(boostingBBArray))
    print(len(csrtBBArray))
    print(len(kcfBBArray))
    print(len(medianflowBBArray))
    print(len(milBBArray))
    print(len(mosseBBArray))
    print(len(tldBBArray))
    '''
    initLen = len(groundTruthArray)
    trackLen = len(boostingBBArray)

    sumOfBoosting = sumOfCsrt = sumOfKcf = sumOfMedianFlow = sumOfMil = sumOfMosse = sumOfTld = 0

    for i in range(trackLen - initLen, trackLen): 
        rowOfGroundTruth = groundTruthArray[i - (trackLen - initLen)].split(',')
        rowOfBoosting = boostingBBArray[i].split(',')
        rowOfCsrt = csrtBBArray[i].split(',')
        rowOfKcf = kcfBBArray[i].split(',')
        rowOfMedianFlow = medianflowBBArray[i].split(',')
        rowOfMil = milBBArray[i].split(',')
        rowOfMosse = mosseBBArray[i].split(',')
        rowOfTld = tldBBArray[i].split(',')
        sumOfBoosting += addToSum(rowOfGroundTruth, rowOfBoosting)
        sumOfCsrt += addToSum(rowOfGroundTruth, rowOfCsrt)
        sumOfKcf += addToSum(rowOfGroundTruth, rowOfKcf)
        sumOfMedianFlow += addToSum(rowOfGroundTruth, rowOfMedianFlow)
        sumOfMil += addToSum(rowOfGroundTruth, rowOfMil)
        sumOfMosse += addToSum(rowOfGroundTruth, rowOfMosse)
        sumOfTld += addToSum(rowOfGroundTruth, rowOfTld)

        IOU[0].append(addToSum(rowOfGroundTruth, rowOfBoosting))
        IOU[1].append(addToSum(rowOfGroundTruth, rowOfCsrt))
        IOU[2].append(addToSum(rowOfGroundTruth, rowOfKcf))
        IOU[3].append(addToSum(rowOfGroundTruth, rowOfMedianFlow))
        IOU[4].append(addToSum(rowOfGroundTruth, rowOfMil))
        IOU[5].append(addToSum(rowOfGroundTruth, rowOfMosse))
        IOU[6].append(addToSum(rowOfGroundTruth, rowOfTld))
        
    sumOfBoosting /= trackLen
    sumOfCsrt /= trackLen
    sumOfKcf /= trackLen
    sumOfMedianFlow /= trackLen
    sumOfMil /= trackLen
    sumOfMosse /= trackLen
    sumOfTld /= trackLen
    IOUAverage[0].append(sumOfBoosting)
    IOUAverage[1].append(sumOfCsrt)
    IOUAverage[2].append(sumOfKcf)
    IOUAverage[3].append(sumOfMedianFlow)
    IOUAverage[4].append(sumOfMil)
    IOUAverage[5].append(sumOfMosse)
    IOUAverage[6].append(sumOfTld)
    boundingBoxFile.close()
    boostingFile.close()
    csrtFile.close()
    kcfFile.close()
    medianflowFile.close()
    milFile.close()
    mosseFile.close()
    tldFile.close()


for x in IOUAverage:
    suma = 0 
    for y in x: 
        suma += y
    suma /= 98
    break

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
    
print (str(plotList))
#plotDataOnGraph(plotList[0])
