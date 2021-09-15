import pandas as pd
import os
import shutil

df = pd.read_csv('paper.csv')
files = df['File Attachments']
paths = []
for i in range(len(files)):
    temp = files[i].split(';')
    for item in temp:
        if '.pdf' in item:
            paths.append(str(item))
title = df['Title']
time = df['Publication Year']
conference = df['Publication Title']
n = len(title)
names = []
for i in range(n):
    names.append(('('+str(time[i])+' '+conference[i]+' ) '+str(title[i])).replace(':','：'))
if not os.path.exists('papers'):
    os.makedirs('papers')
os.chdir('papers')
# for name in names:
#     if not os.path.exists(name):
#         try:
#             os.mkdir(name)
#         except:
#             print(name)
for i in range(n):
    if not os.path.exists(names[i]):
        os.mkdir(names[i])
        try:
            shutil.copy(paths[i],names[i]+'\\全文.pdf')
        except:
            print(names[i])
