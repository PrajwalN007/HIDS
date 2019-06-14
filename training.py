# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:03:30 2019

@author: user
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)
from sklearn.preprocessing import Normalizer 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd
from keras.layers import Convolution2D,Flatten
from keras.layers import LSTM, GRU, SimpleRNN
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
from keras import callbacks 
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.utils.np_utils import to_categorical

dataset=pd.read_csv('attackpt1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 41].values

labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
labelencoder_y = LabelEncoder()

labelencoder_x_1 = labelencoder_x_1.fit(['icmp', 'tcp', 'udp'])
labelencoder_x_2 = labelencoder_x_2.fit(['IRC', 'X11', 'Z39_50', 'aol', 'auth' ,'bgp', 'courier', 'csnet_ns', 'ctf',
 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i' ,'efs',
 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames' ,'http',
 'http_2784', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell',
 'ldap' ,'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns',
 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'other', 'pm_dump' ,'pop_2', 'pop_3',
 'printer', 'private', 'remote_job', 'rje', 'shell', 'smtp' ,'sql_net' ,'ssh',
 'sunrpc', 'supdup' ,'systat', 'telnet', 'tim_i', 'time' ,'urp_i', 'uucp',
 'uucp_path', 'vmnet', 'whois'])
labelencoder_x_3 = labelencoder_x_1.fit(['OTH' ,'REJ' ,'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF','SH'])
labelencoder_y1 = labelencoder_y.fit(['buffer_overflow.' ,'ftp_write.' ,'guess_passwd.', 'imap.','land.', 'loadmodule.', 'multihop.', 'nmap.', 'perl.','phf.', 'pod.' ,'rootkit.', 'spy.', 'warezmaster.'])
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
X[:, 3] = labelencoder_x_3.fit_transform(X[:, 3])
y_train= labelencoder_y1.fit_transform(Y)

scaler = Normalizer().fit(X)
x_train= scaler.transform(X)

y_train1 = np.array(y_train)
y_train= to_categorical(y_train1)

       
def cnnload(testX):
    x_trainr = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1],1))
    cnn = Sequential()
    cnn.add(Convolution2D(64, 3,3, border_mode="same",activation="relu",input_shape=(1,41,1),W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(64, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3,border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(14, activation="softmax"))
    cnn.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
    checkpointer =ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
    csv_logger = CSVLogger('cnn3trainanalysis3.csv',separator=',', append=False)
    cnn.fit(x_trainr, y_train, batch_size=2, nb_epoch=100, show_accuracy=True,callbacks=[checkpointer,csv_logger])
    cnn.save("cnn1.hdf5")
    
def lstmload(testX):
    x_trainr = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
    lstm = Sequential()
    lstm.add(LSTM(128,input_dim=41, return_sequences=True))  
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128,return_sequences=True))
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128, return_sequences=True))  
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128, return_sequences=False))  
    lstm.add(Dropout(0.1))
    lstm.add(Dense(14))
    lstm.add(Activation('softmax'))
    lstm.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
    csv_logger = CSVLogger('training_set_iranalysis3.csv',separator=',', append=False)
    lstm.fit(x_trainr, y_train, batch_size=2, nb_epoch=100,show_accuracy=True,callbacks=[checkpointer,csv_logger])
    lstm.save("lstm1.hdf5")
    
def gruload(testX):
    x_trainr = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
    gru = Sequential()
    gru.add(GRU(64,input_dim=41, return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64,return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64, return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64, return_sequences=False))  
    gru.add(Dropout(0.1))
    gru.add(Dense(14))
    gru.add(Activation('softmax'))
    gru.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
    csv_logger = CSVLogger('training_set_iranalysis3.csv',separator=',', append=False)
    gru.fit(x_trainr, y_train, batch_size=2,nb_epoch=100,show_accuracy=True, callbacks=[checkpointer,csv_logger])
    gru.save("gru1.hdf5")
    
def dnnload(testX):
    x_trainr = np.reshape(x_train, (x_train.shape[0],x_train.shape[1]))
    dnn = Sequential()
    dnn.add(Dense(1024,input_dim=41,activation='relu'))  
    dnn.add(Dropout(0.01))
    dnn.add(Dense(768,activation='relu'))  
    dnn.add(Dropout(0.01))
    dnn.add(Dense(512,activation='relu'))  
    dnn.add(Dropout(0.01))
    dnn.add(Dense(256,activation='relu'))  
    dnn.add(Dropout(0.01))
    dnn.add(Dense(128,activation='relu'))  
    dnn.add(Dropout(0.01))
    dnn.add(Dense(14))
    dnn.add(Activation('softmax'))
    dnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('training_set_dnnanalysis.csv',separator=',', append=False)
    dnn.fit(x_trainr, y_train,batch_size=2, nb_epoch=100,show_accuracy=True, callbacks=[checkpointer,csv_logger])
    dnn.save("dnn1.hdf5")


def rnnload(testX):
    x_trainr = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
    rnn = Sequential()
    rnn.add(SimpleRNN(128,input_dim=41, return_sequences=True))  
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128,return_sequences=True)) 
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128, return_sequences=True))  
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128, return_sequences=False))  
    rnn.add(Dropout(0.1))
    rnn.add(Dense(14))
    rnn.add(Activation('softmax'))
    rnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
    csv_logger = CSVLogger('training_set_iranalysis3.csv',separator=',', append=False)
    rnn.fit(x_trainr, y_train, batch_size=2, nb_epoch=100,show_accuracy=True,callbacks=[checkpointer,csv_logger])
    rnn.save("rnn1.hdf5")
    

def signatureM(x_train):
    cnnload(x_train)
    lstmload(x_train)
    gruload(x_train)
    dnnload(x_train)
    rnnload(x_train)
    print("Signature based model saved to disk")
    

signatureM(x_train)