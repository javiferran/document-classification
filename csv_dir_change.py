import numpy as np
import h5py
from random import shuffle
import glob
import csv
import cv2


file_read = open('./Data/SmallTobacco_ocr.csv', "rU")
reader = csv.reader(file_read, delimiter=',')

file_write = open('./Data/Small_Tobacco_cte.csv', "w")
writer = csv.writer(file_write, delimiter=',')

#Original label csv reading into list
addrs = []
labels = []
segmentation = []
ocr_dirs = []
reader_rows_list = []
length_pre_dir = len('/media/bscuser/bsc')
for row in reader:
      adress = row[0]
      new_adress = '.' + adress[length_pre_dir:]
      lab = int(row[1])
      seg = int(row[2])
      ocr = row[3]
      new_ocr = '.' + ocr[length_pre_dir:]

      #label = np.array(label).astype(int)
      addrs.append(new_adress)
      labels.append(lab)
      segmentation.append(seg)
      ocr_dirs.append(ocr)
      full_row = [new_adress, lab, seg, new_ocr]
      reader_rows_list.append(full_row)


for element in reader_rows_list:
      #print(element)
      new_row = element
      writer.writerow(new_row)

file_read.close()