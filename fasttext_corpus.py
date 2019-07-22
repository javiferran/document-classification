import numpy as np
from random import shuffle
import glob
import csv
import cv2


file_read = open('./Data/Small_Tobacco_cte.csv', "rU")
reader = csv.reader(file_read, delimiter=',')

#Original label csv reading into list
addrs = []
labels = []
segmentation = []
ocr_dirs = []

#reader_rows_list = []
for row in reader:
      adress = row[0]
      lab = int(row[1])
      seg = int(row[2])
      ocr = row[3]
      #label = np.array(label).astype(int)
      addrs.append(adress)
      labels.append(lab)
      segmentation.append(seg)
      ocr_dirs.append(ocr)

      #full_row = [adress, lab, seg, ocr]

      #reader_rows_list.append(full_row)

file_read.close()


# file_write = open('./Data/Small_Tobacco_cover.csv', "w")
# writer = csv.writer(file_write, delimiter=',')
#
# for element in reader_rows_list:
#       #print(element)
#       new_row = element
#       ocr_addr = new_row[3]
#       if '_' in ocr_addr[-20:] and (ocr_addr[-5:][0] != '0' or ocr_addr[-6:][0] != '_'):
#           #print(addr)
#           pass
#
#       else:
#           new_row = element
#           writer.writerow(new_row)



# train_dir = ocr_dirs[0:int(0.6*len(ocr_dirs))]
# test_dir = ocr_dirs[int(0.6*len(ocr_dirs)):int(0.8*len(ocr_dirs))]
# dev_dir = ocr_dirs[int(0.8*len(ocr_dirs)):]
#
# train_labels = labels[0:int(0.6*len(labels))]
# test_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
# dev_labels = labels[int(0.8*len(labels)):]

#Papers split 800/200/rest
train_dir = ocr_dirs[0:800]
test_dir = ocr_dirs[1000:]
dev_dir = ocr_dirs[800:1000]

train_labels = labels[0:800]
test_labels = labels[1000:]
dev_labels = labels[800:1000]



forbidden_chars = ['_1','_2','_3','_4','_5','_6','_7','_8','_9','_1']
count = 0
for i, dir in enumerate(train_dir):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print('File: {}/{}'.format(i, len(ocr_dirs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB

    addr = str(dir)

    if '_' in addr[-20:] and (addr[-5:][0] != '0' or addr[-6:][0] != '_'):
        #print(addr)
        pass
    else:
        print(addr)
        count += 1
        label = str(train_labels[i])

        f = open(dir, "r")
        text = f.read()
        if text != '':
            text = text.replace('\n',' ')
            text = '\n' + '__label__' + label + ' ' + text
            #print(addr)
            #if i < 123:
            receptive_file = open("train.txt","a")
            receptive_file.write(text)


for i, dir in enumerate(test_dir):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print('File: {}/{}'.format(i, len(ocr_dirs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB


    addr = str(dir)
    #print(addr[-5:][0])

    if '_' in addr[-20:] and (addr[-5:][0] != '0' or addr[-6:][0] != '_'):
        #print(addr)
        pass

    else:
        count += 1
        label = str(test_labels[i])
        #print(addr)
        f = open(dir, "r")
        text = f.read()
        if text != '':
            text = text.replace('\n',' ')
            text = '\n' + '__label__' + label + ' ' + text
            #print(text)
            receptive_file = open("test.txt","a")
            receptive_file.write(text)

for i, dir in enumerate(dev_dir):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print('File: {}/{}'.format(i, len(ocr_dirs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB

    addr = str(dir)
    #print(addr[-6:][0])


    if '_' in addr[-20:] and (addr[-5:][0] != '0' or addr[-6:][0] != '_'):
        #print(addr)
        pass

    else:
        print(addr)
        count += 1
        label = str(dev_labels[i])
        #print(addr)
        f = open(dir, "r")
        text = f.read()
        if text != '':
            text = text.replace('\n',' ')
            text = '\n' + '__label__' + label + ' ' + text

            #print(text)
            receptive_file = open("dev.txt","a")
            receptive_file.write(text)
print(count)
