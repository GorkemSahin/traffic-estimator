import re
import pandas as pd
import csv
from dateutil.parser import parse

file_names = [
    "Aug",
    "Dec",
    "Jun",
    "Nov",
    "Oct",
    "Sep"
]

for name in file_names:
    file = open("C:/Users/gbot/Desktop/yagiz/web-traffic-estimation/BITIRME DATA/direnc.net-ssl_log-" + name + "-2020/direnc.net-ssl_log-" + name + "-2020")
    with open(name + '.csv', 'w', newline='') as newfile:
        spamwriter = newfile.write("date_time,size\n")

    #lines = file.readlines(1024 * 1024 * 10)
    #lines = file.readline()
    #print(len(lines))
    day = 0

    # for line in lines:
    #     words = re.split(r'[()]',line)
    #     for word in words:
    #         print(word)
    while 1:
        lines = file.readlines(1024 * 1024 * 1024)
        if not lines:
            break
        rowList = []
        for line in lines:
            words = line.split()
            date = words[3][1:]
            size = words[9]
            if (str.isdigit(size)):
                row = date + "," + size  + "\n"
                rowList.append(row)
            else:
                print(date + "," + size  + " PROBLEMATIC DATA\n")

        with open(name + '.csv', 'a+', newline='') as newfile:
            spamwriter = newfile.writelines(rowList)
        print(date)