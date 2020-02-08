import os
import numpy as np
import cv2

def makeVideo(nameOfVideo, arrayOfImages):
    pathIn = './Dataset/' + nameOfVideo + '/img/'
    fps = 30
    frame_array = []

    for i in range(len(arrayOfImages)):
        filename = pathIn + arrayOfImages[i]
        #print(filename)
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter('./Videos/' + nameOfVideo + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

    out.release()
    return out

folderName = './Dataset'
pathToSrcFolder = 0

pathToDSFolder = os.listdir(folderName)
pathToDSFolder.sort()

for x in pathToDSFolder:
    pathToImgFolder = folderName + '/' + x + '/img'
    sortedImgs = os.listdir(pathToImgFolder)
    sortedImgs.sort() 
    print ("Video: " + x)
    # make a video and save it to folder Videos/
    video = makeVideo(x, sortedImgs)
    