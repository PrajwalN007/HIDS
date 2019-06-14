# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:31:27 2019

@author: user
"""
'''import pandas as pd
dataset=pd.read_csv('kd100part1.csv')
y = dataset.iloc[:,41].values
def histogram(y):
    d=dict()
    i=0
    for c in y:
        if c not in d:
            if c=='normal.':
                d[c]=1
        else:
            if c=='normal.':
                d[c]+=1
        i=i+1
    ij=i-d['normal.']
    return d,ij
x,y=histogram(y)
print(x)
print(y)'''
#{'normal.': 562386, 'buffer_overflow.': 5, 'loadmodule.': 2, 'perl.': 2, 'neptune.': 204815, 'smurf.': 212363, 'guess_passwd.': 53, 'pod.': 40, 'teardrop.': 199, 'portsweep.': 2782, 'ipsweep.': 7579, 'land.': 17, 'ftp_write.': 8, 'back.': 2002, 'imap.': 12, 'satan.': 5389, 'phf.': 3, 'nmap.': 2316, 'multihop.': 6, 'warezmaster.': 20}
import pandas as pd
dataset=pd.read_csv('kd100at1.csv')
y = dataset.iloc[:,41].values
def histogram(y):
    d=dict()
    for c in y:
        if c not in d:
            d[c]=1
        else:
            d[c]+=1
    return d
x=histogram(y)
print(x)