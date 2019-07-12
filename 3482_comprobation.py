import csv
import os
from more_itertools import unique_everseen
from PIL import Image, ImageSequence

hard_disk_root = '/media/bscuser/bsc/BigTobacco/'

file_read_split = open('./Data/SmallTobacco.csv', "rU") #Split SmallTobacco
reader_split = csv.reader(file_read_split, delimiter=',')
split_rows_list = []
for row in reader_split:
    full_row = [row[0], row[1], row[2]]
    split_rows_list.append(full_row)
file_read_split.close()

#######################################################################################

file_write = open('./Data/SmallTobacco_nums.csv', "w")
writer = csv.writer(file_write, delimiter=',')

for element in split_rows_list:
    clase = element[1]
    if(clase == 'Resume'): 
        cat = 0
    elif(clase == 'News'):
        cat = 1
    elif(clase == 'Scientific'):
        cat = 2
    elif(clase == 'Email'):
        cat = 3
    elif(clase == 'ADVE'):
        cat = 4
    elif(clase == 'Memo'):
        cat = 5
    elif(clase == 'Report'):
        cat = 6
    elif(clase == 'Form'):
        cat = 7
    elif(clase == 'Note'):
        cat = 8
    elif(clase == 'Letter'):
        cat = 9
    new_row = [element[0],cat, element[2]]
    writer.writerow(new_row)

#######################################################################################


# file_read_orig = open('./Data/SmallTobacco_files.csv', "rU") #Split SmallTobacco
# reader_orig_3482 = csv.reader(file_read_orig, delimiter=',')
# orig_3482_rows_list = []
# for row in reader_orig_3482:
#     orig_row = [row[0]]
#     orig_3482_rows_list.append(full_row)
# file_read_orig.close()



# counter = 0
# for element_orig in orig_3482_rows_list:
#     if counter%100==0:
#         print(counter)
#     orig_imPath = element_orig[0]
#     for element in split_rows_list:
#         split_imPath = element[0]
#         if orig_imPath[:-4] in split_imPath:
#             counter += 1
#             break

# print(counter)