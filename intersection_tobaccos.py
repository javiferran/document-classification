import csv
import os
from more_itertools import unique_everseen
from PIL import Image, ImageSequence

hard_disk_root = '/media/bscuser/bsc/labels/'

# file_read_big = open(hard_disk_root + 'train.csv', "rU") #Split SmallTobacco
# reader_big = csv.reader(file_read_big, delimiter=',')
# big_list = []
# for row in reader_big:
#     full_row = [row[0], row[1]]
#     big_list.append(full_row)
# file_read_big.close()
#
# file_read_small = open('./Data/Small_Tobacco_cover_final.csv', "rU") #Split SmallTobacco
# reader_small = csv.reader(file_read_small, delimiter=',')
# small_list = []
# for row in reader_small:
#     full_row = [row[0], row[1]]
#     small_list.append(full_row)
# file_read_small.close()
#
# file_write = open(hard_disk_root + 'intersection.csv', "w")
# writer = csv.writer(file_write, delimiter=',')
#
#
# counter = 0
#
# for row_big in big_list:
#     #print('Big Tobacco',row_big[0][14:][:8])
#     # print()
#
#     for row_small in small_list:
#
#         if row_big[0][14:][:8] == row_small[0][29:][:8]:
#             writer.writerow(row_big)
#             counter += 1
#             print(counter)
# print(counter)


#################################################################################################################################3
file_read_inter= open(hard_disk_root + 'intersection.csv', "rU") #Split SmallTobacco
reader_inter = csv.reader(file_read_inter, delimiter=',')
inter_list = []
for row in reader_inter:
    full_row = [row[0], row[1]]
    inter_list.append(full_row)
file_read_inter.close()

file_read_big = open(hard_disk_root + 'train.csv', "rU") #Split SmallTobacco
reader_big = csv.reader(file_read_big, delimiter=',')
big_list = []
for row in reader_big:
    full_row = [row[0], row[1]]
    big_list.append(full_row)
file_read_big.close()

file_write = open(hard_disk_root + 'train_intersection.csv', "w")
writer = csv.writer(file_write, delimiter=',')

for row_big in big_list:
    #print('Big Tobacco',row_big[0][14:][:8])
    # print()
    counter = 0

    for row_inter in inter_list:

        if row_big[0] != row_inter[0]:
            counter += 1

    if counter == len(inter_list):
        writer.writerow(row_big)
