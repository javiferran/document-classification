import csv
import os

# hard_disk_root = '/media/bscuser/bsc/BigTobacco/'

file_read = open('./Data/labels_full.csv', "rU") #BigTobacco csv dataset with labels
reader = csv.reader(file_read, delimiter=',')

path = './Tobacco3482'

# file_read_split = open('./Data/ModifiedBigTobacco.csv', "rU") #BigTobacco csv dataset with labels
# reader_split = csv.reader(file_read_split, delimiter=',')

# file_write = open('./Data/BigTobacco_not_founded.csv', "w")
# writer = csv.writer(file_write, delimiter=',')


#Original labels_full (image directory and class of BigTobacco) csv reading and save into a list
new_rows_list = []
for row in reader:
    full_row = [row[0], row[1]]
    new_rows_list.append(full_row)
file_read.close()

# #Splited labels_full (image directory and class of BigTobacco)
# split_rows_list = []
# for row in reader_split:
#     full_row_split = [row[0], row[1]]
#     split_rows_list.append(full_row_split)
# file_read_split.close()

# counter = 0
# number_rows = 0
# #print(len(split_rows_list))
# for element in new_rows_list:
#     number_rows+=1
#     if number_rows%100==0:
#         print(number_rows)
#     coincidences = 0
#     imPath = element[0]
#     if 'imagesz/z/p' in imPath:
#         #print(imPath[:-4])
#         for element_split in split_rows_list:
#             imPath_split = element_split[0]
#             file_split = imPath_split[len(hard_disk_root):]
#             #print(file_split[:-4])
#             if imPath[:-4] in file_split[:-4]:
#                 coincidences += 1
#         if coincidences == 0:
#             counter += 1
#             print('file not found', element)
#             new_row = [element]
#             writer.writerow(new_row)
# print('Elements not found', counter)


counter = 0
for element in new_rows_list:
    #print(element)
    imPath = element[0]
    #print('imPath', imPath)
    #print(counter)
    if counter==100:
        print(counter)
    #print(imPath)
    #print(len(hard_disk_root))
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[:-4] in imPath:
                print(file)
                print(imPath)
                counter += 1
print(counter)