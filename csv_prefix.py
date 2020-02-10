import csv
import os

file_write = open('/media/bscuser/bsc/labels/train_intersection_prefix.csv', "w")
writer = csv.writer(file_write, delimiter=',')

file_read = open('/media/bscuser/bsc/labels/train_intersection.csv', "rU")
reader = csv.reader(file_read, delimiter=',')
reader_rows_list = []
for row in reader:
    full_row = [row[0], row[1]]
    reader_rows_list.append(full_row)
file_read.close()


for element in reader_rows_list:
    imPath = element[0]
    #new_imPath = imPath.replace("PEEKBOX", "bsc")
    new_imPath = '/media/bscuser/bsc/images/' + imPath
    #print(new_imPath)
    new_row = [new_imPath, element[1]]
    writer.writerow(new_row)
