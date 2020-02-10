import cv2
import sys
import pytesseract
import os
from time import sleep
from PIL import Image, ImageSequence
import numpy
import csv

if __name__ == '__main__':

    #path_save = './OCRS/'
    file_read = open('/media/bscuser/bsc/labels/train_intersection_prefix.csv', "rU")
    reader = csv.reader(file_read, delimiter=',')

    #Original label csv reading into list
    new_rows_list = []
    for row in reader:
        #full_row = [row[0], row[1]]
        new_rows_list.append(row)
    file_read.close()

    file_write = open('/media/bscuser/bsc/labels/BigTobacco_cover_train_intersection.csv', "a")
    writer = csv.writer(file_write, delimiter=',')

    counter = 0
    for element in new_rows_list:
        counter += 1
        if counter%50==0:
            print(counter)
        if counter >= 72202:#last time
            img_dir = element[0]
            #imPath = os.path.join(root,file)
            #print(img_dir)


            config = ('tesseract image.jpg output -l eng --oem 1 --psm 3')

            # Read image from disk
            im = cv2.imread(img_dir, cv2.IMREAD_COLOR)

            # Run tesseract OCR on image
            try:
                text = pytesseract.image_to_string(im, config=config)
                file = open(os.path.join(img_dir[:-4] + ".txt"),"w")
                file.write(text)
                new_row = [element[0],element[1],img_dir[:-4] + ".txt"]
            except:
                new_row = ' '
                print('',img_dir)
            #txt_name = os.path.join(imPath, image_name + ".txt")
            #txt_name = imPath + ".txt"




            #print(new_row)

            writer.writerow(new_row)
            #file.close()

        # Print recognized text
    file_write.close()
