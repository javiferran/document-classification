import csv
import os
from more_itertools import unique_everseen

path_3482 = './Tobacco3482'

file_write = open('./Data/SmallTobacco_files.csv', "w")
writer = csv.writer(file_write, delimiter=',')

for root_3482, dirs_3482, files_3482 in os.walk(path_3482):
    for file in files_3482:
        print(file)
        new_row = [file,root_3482[14:]]
        writer.writerow(new_row)

