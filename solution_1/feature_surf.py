
import cv2
import os
import csv


train_path = "../data-train"

surf = cv2.xfeatures2d.SURF_create()

des_list = []
for img_path in os.listdir(train_path):
    img = cv2.imread(train_path + "/"+ img_path,0)
    print "load image: ", (train_path + "/"+ img_path)

    kp, des = surf.detectAndCompute(img,None)
    #des_list.append((img_path, des))

    with open('extract-feature/'+img_path[:-4], 'wb') as csvfile:
        matrixwriter = csv.writer(csvfile, delimiter=' ')
        a = str(train_path + "/"+ img_path)
        matrixwriter.writerow(a)
        for row in des:
            matrixwriter.writerow(row)

