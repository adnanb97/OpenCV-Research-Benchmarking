import os

file = open("videoClasses.txt", "r")
text = []
for line in file.readlines():
    if (line != '\n'):
        text.append(line.split(', '))
    #    groundTruth.append(line)
print (str(text))