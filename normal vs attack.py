###### -*- coding: utf-8 -*-
#####"""
#####Created on Sun Apr 21 11:51:15 2019
#####
#####@author: user
#####"""
#####
######to separate normal and attack 
######
######with open('KDDTrain.csv') as rad_file, open("out.csv","w") as out_file:
######    header = next(rad_file)
######    out_file.write(header)
######    for rad_line in rad_file:
######        if rad_line.endswith('normal'):
######            out_file.write(rad_line)
#####
#import csv
###
#with open('kd100normal.csv', 'r') as i, open('out.csv', 'w') as o:
#   r = csv.reader(i, delimiter=',')
#   w = csv.writer(o, delimiter=',')
#   print(r)
#   for row in r:
#       rx=list(row)
#       if rx[41]=='normal.':
#          w.write(row) 
###
###
#######convert .gz to csv
######import gzip
######kd10=open('kd100.csv','w')
######csvFile = gzip.open('kddcup.data.gz', 'rt', newline='') 
###### # Open in text mode, not binary, no line ending translation
######c=0
######for r in csvFile:
######    kd10.write(r)
######    c=c+1
######print(c)    
#####import csv
######import pandas as pd
#####import gzip
######kd10=open('kd100at3.csv','w')
######csvFile =open('kd100attack.csv', 'r')
######csvFile1=csv.reader(csvFile)
######kd10=csv.writer(kd10,delimiter=',')
####### 
####### # Open in text mode, not binary, no line ending translation
######c=0
######k=0
######for r in csvFile1:
######    if c>6000000 and c<7851300:
####
######        kd10.writerow(r)
######        k=k+1
######    c=c+1
######    
######print(k)
########    if c==5000000:
######print(c)
########        break 
######
import csv
###import pandas as pd
import gzip
kd=open('attackpt3.csv','w')

kd10 = gzip.open('kddcup.data_10_percent.gz', 'rt' )
#kd10=open(csvFile,'r')
c=0
csvFile=csv.reader(kd10)    
kdnor10=csv.writer(kd,delimiter=',')

#kdnor10label=kdnor10.writeheader()     
for row in csvFile:
    r=list(row )
    if not (r[41]=='buffer_overflow.'or r[41]=='loadmodule.' or  r[41]=='perl.' or  r[41]=='guess_passwd.'  or r[41]=='pod.' or  r[41]=='land.' or  r[41]=='ftp_write.' or  r[41]=='imap.' or  r[41]=='phf.' or  r[41]=='nmap.' or  r[41]=='multihop.' or  r[41]=='warezmaster.' or  r[41]=='spy.' or  r[41]=='rootkit.' or r[41]=='normal.') :
        kdnor10.writerow(row)
        c=c+1
print(c)
####import gzip
####import pandas as pd
####df=pd.read_csv('kddcup.data_10_percent.gz',compression='gzip',header=None)
####col=df.columns
####print(col)
####
#####import csv
#####inp=open('kd100normal.csv','r')
#####op=open('kdnwb.csv','w')
#####writer=csv.writer(op)
#####for row in csv.reader(inp):
#####    if any(field.strip() for field in row):
#####        writer.writerow(row)
#####inp.close()
#####op.close()
####
####
####import csv
####from csv import Dialect
####
####class LiferayDialect(Dialect):
####    delimiter = ';'
####    quotechar = '"'
####    doublequote = True
####    lineterminator = '\n'
####    q uoting = csv.QUOTE_MINIMAL
####
####csv.register_dialect("liferayDialect", LiferayDialect)
####
####cr = csv.reader(open("kd100normal.csv", "r"), dialect="liferayDialect")
####cw = csv.writer(open("clean_data.csv", "w"), dialect="excel")
####
####crlf = '\r\n'
####
####for row in cr:
####    row2 = []
####    for col in row:
####        while crlf in col or '\n' in col:
####            col = col.replace(crlf, ' ')
####            col = col.replace('\n', ' ')
####        row2.append(col)
####    cw.writerow(row2)
##import csv
##file_object = open('kd100normal.csv', 'r')
##lines = csv.reader(file_object, delimiter=',', quotechar='"')
##flag = 0
##data=[]
##for line in lines:
##    if line == []:
##        flag =1
##        continue
##    else:
##        data.append(line)
##file_object.close()   
##if flag ==1: #if blank line is present in file
##    file_object = open('kd100normal.csv', 'w')
##    for line in data:
##        str1 = ','.join(line)
##        file_object.write(str1+"\n")
##    file_object.close() 
#import pandas as pd
#d=pd.read_csv('kd100normal.csv')
#d.to_csv('out.csv',index=False)
