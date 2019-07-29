import csv
import os
from more_itertools import unique_everseen

path_3482 = './Tobacco3482'


file_read = open('./Data/Copies/SmallTobacco_files.csv', "rU") #BigTobacco csv dataset with labels
reader = csv.reader(file_read, delimiter=',')

file_read_2 = open('./Data/Small_Tobacco_cover.csv', "rU") #BigTobacco csv dataset with labels
reader_2 = csv.reader(file_read_2, delimiter=',')

file_write = open('./Data/Small_Tobacco_cover_final.csv', "w")
writer = csv.writer(file_write, delimiter=',')

images_list = []
for row in reader:
    image = row[0]
    images_list.append(image)
file_read.close()

paths_list = []
new_rows_list = []
for row in reader_2:
    full_row = row
    path = row[0]
    new_rows_list.append(full_row)
    paths_list.append(path)
file_read_2.close()

total_counter = 0
final_list = []
for element in images_list:
    #print(element[:-4])
    counter = 0
    for row in new_rows_list:
        if element[:-4] in row[0]:
            counter += 1
            if counter == 1:
                #final_list.append(row)
                writer.writerow(row)
            else:
                total_counter += 1
                #print(img_path)

print(total_counter)






# for root_3482, dirs_3482, files_3482 in os.walk(path_3482):
#     for file in files_3482:
#         print(file)
#         new_row = [file,root_3482[14:]]
#         writer.writerow(new_row)
