import csv
import os
import cv2
from PIL import Image, ImageSequence


root = './'
file_read = open('./Data/labels_full.csv', "rU")
destination_path = './BigTobacco/'
reader = csv.reader(file_read, delimiter=',')

#Original labels_full (image directory and class of BigTobacco) csv reading and save into a list
new_rows_list = []
for row in reader:
      full_row = [row[0], row[1]]
      new_rows_list.append(full_row)
file_read.close()


#Open new csv file if does not exist, otherwise append to existing one -> new label datset csv
#Iterate over labels list and get path of image
#Extract new images from multipages tiff into pngs (due to problems with some tiffs)
#Add new png image path + class + segmentation
if os.path.exists('./Data/new_labels.csv'):
    file_write = open('./Data/new_labels.csv', "a")
    writer = csv.writer(file_write, delimiter=',')
else:
    file_write = open('./Data/new_labels.csv', "w")
    writer = csv.writer(file_write, delimiter=',')
for element in new_rows_list:
    imPath = element[0]
    if 'imagesa/a/a' in imPath or 'imagesa/a/b' in imPath: #Just to test with one folder, delete for entire Tobacco
        imPath = os.path.join(root,imPath)
        print(imPath)
        img = Image.open(imPath)
        img.load()
        save_path = destination_path + imPath[10:][:-4]
        pos = save_path.rfind('/')
        doc_title = save_path[pos:][1:]
        save_path = save_path[:pos+1]
        print(save_path)
        if os.path.exists(save_path):
            pass
        else:
            try:  
                os.makedirs(save_path)
            except OSError:  
                print ("Creation of the directory %s failed" % save_path)
        if img.n_frames != 1:
            for i, page in enumerate(ImageSequence.Iterator(img)):
                #page.save(imPath[:-4] + '_' + str(i) + '.png')
                page.save(save_path + doc_title + '_' + str(i) + '.png')
                if i == 0:
                    new_row = [save_path + doc_title + '_' + str(i) + '.png', element[1], 1]
                else:
                    new_row = [save_path + doc_title + '_' + str(i) + '.png', element[1], 0]
                writer.writerow(new_row)
            img.close()
        else:
            img.save(save_path + doc_title + '.png')
            new_row = [save_path + doc_title + '.png', element[1],1]
            writer.writerow(new_row)
file_write.close()