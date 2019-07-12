import csv
import os

file_write = open('./Data/ModifiedBigTobacco.csv', "w")
writer = csv.writer(file_write, delimiter=',')

file_read = open('./Data/BigTobacco_labels_no_duplicates.csv', "rU")
reader = csv.reader(file_read, delimiter=',')
reader_rows_list = []
for row in reader:
    full_row = [row[0], row[1], row[2]]
    reader_rows_list.append(full_row)
file_read.close()


for element in reader_rows_list:
    imPath = element[0]
    new_imPath = imPath.replace("PEEKBOX", "bsc")
    #print(new_imPath)
    new_row = [new_imPath, element[1],element[2]]
    writer.writerow(new_row)

