import os
import numpy as np
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import math

folderName = './Dataset'
pathToSrcFolder = 0

pathToDSFolder = os.listdir(folderName)
pathToDSFolder.sort()


(major, minor) = cv2.__version__.split(".")[:2]


# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

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

 
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# this is a function to scale a bounding box regarding which way and with which factor of scaling
def scaleBoundingBox(bb, scaleNumber, factorOfScaling, width, height):
    x = bb[0]
    y = bb[1]
    w = bb[2]
    h = bb[3]
    procentW = w * factorOfScaling
    procentH = h * factorOfScaling
    if (scaleNumber == 1): # left
        x = int(bb[0] - procentW)
    elif (scaleNumber == 2): # up
        y = int(bb[1] - procentH)
    elif (scaleNumber == 3): # right
        w = int(bb[2] + procentW)
    elif (scaleNumber == 4): # down
        h = int(bb[3] + procentH)
    elif (scaleNumber == 5): # left up
        x = int(bb[0] - procentW)
        y = int(bb[1] - procentH)
    elif (scaleNumber == 6): # right up
        w = int(bb[2] + procentW)
        y = int(bb[1] - procentH)
    elif (scaleNumber == 7): # left down
        x = int(bb[0] - procentW)
        h = int(bb[3] + procentH)
    elif (scaleNumber == 8): # right down 
        w = int(bb[2] + procentW)
        h = int(bb[3] + procentH)
    x = max(0, x)
    y = max(0, y)
    w = max(0, w)
    h = max(0, h)
    if (x + w > width):
        w = w - (x + w - width)
    # w = min(x + w, height)
    if (y + h > height): 
        h = h - (y + h - height)
    # h = min(y + h, width)

    return (x, y, w, h)


# parameters are video stream, what tracker to use, its name, initial BB, key is always 0 by default and
# an array of bounding boxes which represent ground truth and the name of the video file
# so we can make all the txt files needed for later visualisation 

# scaleNumber is an integer representing how to scale the initial bounding box: 
# 0 nothing, 1 left 2 up 3 right 4 down 5 left up 6 right up 7 left down 8 right down


def doTracking(vs, tracker, trackerName, initBB, key, groundTruth, nameOfVideoFile, scaleNumber, factorOfScaling = 0.1):
    fps = None
    frameNumber = 0
    #fileName1 = 'ResultsCD/' + nameOfVideoFile + 'CenterDistance' + trackerName + '.txt'
    #fileName2 = 'ResultsIOU/' + nameOfVideoFile + 'IntersectionOverUnion' + trackerName + '.txt'
    #fileName3 = 'ResultsFPS/' + nameOfVideoFile + 'FPS' + trackerName + '.txt'
    #fileName4 = 'TrackerBoundingBoxes/' + nameOfVideoFile + 'BB' + trackerName + '.txt'
    fileName5 = 'TrackerBoundingBoxesScaled/' + str(scaleNumber) + '/' + nameOfVideoFile + 'BB' + trackerName + '.txt' 
    os.makedirs(os.path.dirname(fileName5), exist_ok=True)
    if (os.path.isfile(fileName5)): 
        return
    #file1 = open(fileName1, "w")
    #file2 = open(fileName2, "w")
    #file3 = open(fileName3, "w")
    #file4 = open(fileName4, "w")
    with open(fileName5, "w") as file5:
    #file5 = open(fileName5, "w")

        while True:
            # grab the current frame
            frame = vs.read()
            frame = frame[1]
            
            # resize the frame (so we can process it faster) and grab the
            # frame dimensions

            #frame = imutils.resize(frame, width=500)
            
            # check to see if we have reached the end of the stream
            if frame is None:
                break
            
            (H, W) = frame.shape[:2]
            if key == 0:
                initBB = scaleBoundingBox(initBB, scaleNumber, factorOfScaling, W, H)
                print(initBB)
                print((H, W))
                tracker.init(frame, initBB)
                fps = FPS().start()
                key = 1
                # if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                break

            
            # draw the ground truth BB
            if (frameNumber < len(groundTruth)):
                (xbb, ybb, wbb, hbb) = [int(v) for v in groundTruth[frameNumber].split(',')]
                frameNumber = frameNumber + 1
                cv2.rectangle(frame, (xbb, ybb), (xbb + wbb, ybb + hbb), (0, 255, 255), 2)

            successStr = 'No'
            # check to see if we are currently tracking an object
            if initBB is not None:
                # grab the new bounding box coordinates of the object
                (success, box) = tracker.update(frame)
                # check to see if the tracking was a success
                upis = '99999 99999 99999 99999\n'
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    upis = str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)
                    # get the benchmark data 
                    distanceBetweenCenters = center_distance((x,y,x+w,y+h), (xbb,ybb,xbb+wbb,ybb+hbb))
                    intersectionOverUnion = intersection_over_union((x,y,x+w,y+h), (xbb,ybb,xbb+wbb,ybb+hbb))
                    successStr = 'Yes'
                    
                else: 
                    intersectionOverUnion = 0
                    distanceBetweenCenters = 999999
                
                
                # update the FPS counter
                fps.update()
                fps.stop()
                fpsStr = str("{:.2f}".format(fps.fps()))
                #file1.write(str(distanceBetweenCenters) + '\n')
                #file2.write(str(intersectionOverUnion) + '\n')
                #file3.write(str(fpsStr + ' ' + successStr) + '\n')
                #file4.write(upis)
                file5.write(upis)
                # initialize the set of information we'll be displaying on
                # the frame
                info = [
                    ("Tracker", trackerName),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps())),
                    ("Ground Truth: ", "Yellow")
                ]
        
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
    #file1.close()
    #file2.close()
    #file3.close()
    #file4.close()
    file5.close()
    vs.release()

for x in pathToDSFolder:
    pathToBBFolder = folderName + '/' + x
    boundingBoxFile = open(pathToBBFolder + '/groundtruth_rect.txt', "r")
    
    # get a reference to the video file
    videoFile = './Videos/' + x + '.avi'
    
    listInitialBB = boundingBoxFile.readline()
    # some were separated by tab and some by comma - converting them all to commas
    listInitialBB = listInitialBB.replace('\t', ',')
    listInitialBB = listInitialBB.split(',')
    listInitialBB = [x.rstrip() for x in listInitialBB]
    initBB = (int(listInitialBB[0]),int(listInitialBB[1]),int(listInitialBB[2]),int(listInitialBB[3]))
    
    print(x)
    
    # get the bounding boxes for the rest of the video so we have a 
    # ground truth to compare it with

    groundTruth = []
    for line in boundingBoxFile.readlines():
        line = line.replace('\t', ',')
        line = line.replace(' ', ',')
        line = line.rstrip() 
        groundTruth.append(line)

    
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    for trackerType in OPENCV_OBJECT_TRACKERS:
        
        for scaleNumber in range(8): 
            tracker = OPENCV_OBJECT_TRACKERS[trackerType]()
            vs = cv2.VideoCapture(videoFile)
            doTracking(vs, tracker, trackerType, initBB, 0, groundTruth, x, scaleNumber + 1)
            vs.release()
 
#cv2.destroyAllWindows()
