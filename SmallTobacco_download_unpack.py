# import requests

import urllib.request  as urllib2
import string
import os
from pyunpack import Archive
import shutil
import re

import csv
import os
import cv2
from PIL import Image, ImageSequence

hard_disk_path = '/media/bscuser/bsc/' #Change roots with hard_disk_path
alphabet = list(string.ascii_lowercase)
save_path = '/media/bscuser/bsc/Tobacco/'
save_path_extracted = '/media/bscuser/bsc/Tobacco_extracted/'

file_read = open('./Data/SmallTobacco_files.csv', "rU") #BigTobacco csv dataset with labels
reader = csv.reader(file_read, delimiter=',')

#Original labels_full (image directory and class of BigTobacco) csv reading and save into a list
new_rows_list = []
for row in reader:
    full_row = [row[0], row[1]]
    new_rows_list.append(full_row)
file_read.close()

def main():
    # file_to_split = 'images' + 'a' + '/' + 'a' + '/' + 'a'
    # print(file_to_split)
    # split(file_to_split)


    counter = 0
    for letter in alphabet:
        for letter_2 in alphabet:
            counter += 1
            url = "https://ir.nist.gov/cdip/cdip-images/images" + "." + letter + "." + letter_2 + "." + "cpio"
            print(url)
            file_name = url.split('/')[-1]
            print('Counter',counter)
            if (letter == 'f' and letter_2 == 'o') or (letter == 'f' and letter_2 == 'p') or (letter == 'z' and letter_2 == 'p') or (letter == 's'):
                pass
            else:
            #if counter > 2: #663 = z.n
                print(file_name)
                u = urllib2.urlopen(url)
                meta = u.info()
                file_dir = os.path.join(save_path,file_name)
                print(file_dir)
                f = open(file_dir, 'wb')
                print(meta)
                #file_size = int(meta.getheaders("Content-Length")[0])
                #print("Downloading: %s Bytes: %s" % (file_name, file_size))
                file_size_dl = 0
                block_sz = 8192
                while True:
                    buffer = u.read(block_sz)
                    if not buffer:
                        break

                    file_size_dl += len(buffer)
                    f.write(buffer)
                    #status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                    #status = status + chr(8)*(len(status)+1)
                    if file_size_dl%100000 == 0:
                        print(file_size_dl/1048576)
                        print('MB')

                f.close()

                #Unpack
                Archive('/media/bscuser/bsc/Tobacco/' + file_name).extractall(save_path_extracted)
                os.remove('/media/bscuser/bsc/Tobacco/' + file_name)
                file_to_split = 'images' + letter + '/' + letter + '/' + letter_2
                print(file_to_split)
                split(file_to_split) #Send folder, i.e imagesb/b/d
                #print(counter)
            
            #Else below used for individual cases, set counter to a high value >700
            # else:
            #     letter = 'a'
            #     letter_2 = 'a'
            #     file_to_split = 'images' + letter + '/' + letter + '/' + letter_2
            #     print(file_to_split)
            #     split(file_to_split)


#Checks every document in file (folder, i.e imagesb/b/d) with BigTobacco csv dataset
#For every coincidence, split into multiple images
#Store new images in new_labels.csv

def split(file):
    path_extracted = '/media/bscuser/bsc/Tobacco_extracted/' + file
    destination_path = '/media/bscuser/bsc/SmallTobacco/'
    path_3482 = './Tobacco3482'


    #Open new csv file if does not exist, otherwise append to existing one -> new label datset csv
    #Iterate over labels list and get path of image
    #Extract new images from multipages tiff into pngs (due to problems with some tiffs)
    #Add new png image path + class + segmentation
    if os.path.exists('./Data/SmallTobacco.csv'):
        file_write = open('./Data/SmallTobacco.csv', "a")
        writer = csv.writer(file_write, delimiter=',')
    else:
        file_write = open('./Data/SmallTobacco.csv', "w")
        writer = csv.writer(file_write, delimiter=',')

    #Important to loop over directory in first order
    for root, dirs, files in os.walk(path_extracted):
            counter=0
            for extracted_file in files:
                for element in new_rows_list:
                    file_3482 = element[0] #Name of file
                    class_3482 = element[1] #Class of the file: News, Letter...

                    if file_3482 == extracted_file:
                        imPath = os.path.join(root,extracted_file) #Path from Tobacco_extracted folder
                        print(imPath)
                        bar_locations = []
                        penultimate_bar = 0
                        for m in re.finditer('/', imPath):
                            bar_locations.append(m.end())

                        penultimate_bar = bar_locations[-2]
                        last_bar =  bar_locations[-1]
                        ocr_path = imPath[:last_bar]

                        folder = imPath[penultimate_bar:]
                        bar_position = folder.find('/')
                        folder = folder[:bar_position]

                        
                        img = Image.open(imPath) #Open image
                        img.load()
                        save_path = destination_path + imPath[len(root):][:-4]
                        pos = save_path.rfind('/')
                        doc_title = save_path[pos:][1:]

                        save_path = root.replace('Tobacco_extracted','SmallTobacco') + '/'

                        if os.path.exists(save_path):
                            pass
                        else:
                            try:  
                                os.makedirs(save_path)
                            except OSError:  
                                print ("Creation of the directory %s failed" % save_path)
                        try:
                            os.rename(ocr_path +folder + '.xml', save_path + doc_title + '.xml')
                        except:
                                print("This file has no OCR")
                        if img.n_frames != 1:
                            for i, page in enumerate(ImageSequence.Iterator(img)):
                                #page.save(imPath[:-4] + '_' + str(i) + '.png')
                                page.save(save_path + doc_title + '_' + str(i) + '.png')
                                if i == 0:
                                    new_row = [save_path + doc_title + '_' + str(i) + '.png', class_3482, 1]
                                else:
                                    new_row = [save_path + doc_title + '_' + str(i) + '.png', class_3482, 0]
                                writer.writerow(new_row)
                            img.close()
                            #os.remove(imPath)
                        else:
                            img.save(save_path + doc_title + '.png')
                            new_row = [save_path + doc_title + '.png', class_3482, 1]
                            writer.writerow(new_row)
    file_write.close()
    # try:
    #     shutil.rmtree(root + file)
    # except:
    #     print('Couldnt delete')

    

main()