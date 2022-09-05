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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  
from keras.models import model_from_json
import matplotlib.pyplot as plt

dataset=pd.read_csv('attackpt1.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 41].values
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
labelencoder_y = LabelEncoder()

labelencoder_x_1 = labelencoder_x_1.fit(['icmp' 'tcp' 'udp'])
labelencoder_x_2 = labelencoder_x_2.fit(['IRC' 'X11' 'Z39_50' 'aol' 'auth' 'bgp' 'courier' 'csnet_ns' 'ctf'
 'daytime' 'discard' 'domain' 'domain_u' 'echo' 'eco_i' 'ecr_i' 'efs'
 'exec' 'finger' 'ftp' 'ftp_data' 'gopher' 'harvest' 'hostnames' 'http'
 'http_2784' 'http_443' 'http_8001' 'imap4' 'iso_tsap' 'klogin' 'kshell'
 'ldap' 'link' 'login' 'mtp' 'name' 'netbios_dgm' 'netbios_ns'
 'netbios_ssn' 'netstat' 'nnsp' 'nntp' 'other' 'pm_dump' 'pop_2' 'pop_3'
 'printer' 'private' 'remote_job' 'rje' 'shell' 'smtp' 'sql_net' 'ssh'
 'sunrpc' 'supdup' 'systat' 'telnet' 'tim_i' 'time' 'urp_i' 'uucp'
 'uucp_path' 'vmnet' 'whois'])
labelencoder_x_3 = labelencoder_x_1.fit(['OTH' 'REJ' 'RSTO' 'RSTOS0' 'RSTR' 'S0' 'S1' 'S2' 'S3' 'SF' 'SH'])
labelencoder_y = labelencoder_y.fit(['back.' 'buffer_overflow.' 'ftp_write.' 'guess_passwd.' 'imap.'
 'ipsweep.' 'land.' 'loadmodule.' 'multihop.' 'neptune.' 'nmap.' 'perl.'
 'phf.' 'pod.' 'portsweep.' 'rootkit.' 'satan.' 'smurf.' 'spy.'
 'teardrop.' 'warezclient.' 'warezmaster.'])

X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
X[:, 3] = labelencoder_x_3.fit_transform(X[:, 3])
testY= labelencoder_y.fit_transform(Y)

scaler = Normalizer().fit(X)
testX= scaler.transform(X)
labels=['back.' ,'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.',
 'ipsweep.' ,'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.',
 'phf.', 'pod.', 'portsweep.', 'rootkit.' ,'satan.' ,'smurf.', 'spy.',
 'teardrop.', 'warezclient.' ,'warezmaster.']

def con_mat(y_pred,y_test):
    cm=confusion_matrix(y_test,y_pred)
    print("done") 
    #print(labels[int(y_pred)])
    print("\n"+classification_report(y_test, y_pred))
    from mlxtend.plotting import plot_confusion_matrix
    fig,ax=plot_confusion_matrix(conf_mat=cm,figsize=(15,15))
    plt.show()
    fig,ax=plt.subplots()
    ax.scatter(y_test,y_pred )
    ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('predicted')
    fig.show()
 
   
    
def cnnload(testX,op):
    cnn = Sequential()
    cnn.add(Convolution2D(64, 3,3, border_mode="same",activation="relu",input_shape=(1,41,1),W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(64, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3,border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(op, activation="softmax"))
    cnn.load_weights("cnn1.hdf5")
    print("loaded cnn")
    testX=np.array(testX)
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1],1))
    y_pred = cnn.predict_classes(testXR)
    print(y_pred)
    #con_mat(y_pred,testY)
    return cnn,y_pred

def lstmload(testX,op):
    lstm = Sequential()
    lstm.add(LSTM(128,input_dim=41, return_sequences=True))  
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128,return_sequences=True))
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128, return_sequences=True))  
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128, return_sequences=False))  
    lstm.add(Dropout(0.1))
    lstm.add(Dense(op))
    lstm.add(Activation('softmax'))
    lstm.load_weights("lstm1.hdf5")
    print("loaded lstm")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    y_pred = lstm.predict_classes(testXR)
    con_mat(y_pred,testY)
    return lstm,y_pred
    
def gruload(testX,op):
    gru = Sequential()
    gru.add(GRU(64,input_dim=41, return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64,return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64, return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64, return_sequences=False))  
    gru.add(Dropout(0.1))
    gru.add(Dense(op))
    gru.add(Activation('softmax'))
    gru.load_weights("gru1.hdf5")
    print("loaded gru")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    y_pred = gru.predict_classes(testXR)
    con_mat(y_pred,testY)
    return gru,y_pred
    

def dnnload(testX,op):
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
    dnn.add(Dense(op))
    dnn.add(Activation('softmax'))
    dnn.load_weights('dnn1.hdf5')
    print("loaded dnn")
    testXR = np.reshape(testX, (testX.shape[0],testX.shape[1]))
    y_pred = dnn.predict_classes(testXR)
    con_mat(y_pred,testY)
    return dnn,y_pred


def rnnload(testX,op):
    rnn = Sequential()
    rnn.add(SimpleRNN(128,input_dim=41, return_sequences=True))  
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128,return_sequences=True)) 
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128, return_sequences=True))  
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128, return_sequences=False))  
    rnn.add(Dropout(0.1))
    rnn.add(Dense(op))
    rnn.add(Activation('softmax'))
    rnn.load_weights('rnn1.hdf5')
    print("loaded rnn")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    y_pred = rnn.predict_classes(testXR)
    con_mat(y_pred,testY)
    return rnn,y_pred

def signatureM(testX):
    cnn,yc=cnnload(testX,14)
    lstm,yl=lstmload(testX,14)
    gru,yg=gruload(testX,14)
    dnn,yd=dnnload(testX,14)
    rnn,yr=rnnload(testX,14)
    '''cnn.summary()
    lstm.summary()
    gru.summary()
    dnn.summary()
    rnn.summary()'''
    
def anamoly():
    dataset=pd.read_csv('kd10.csv')
    X = dataset.iloc[:, :-1].values
    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()
    labelencoder_x_1 = labelencoder_x_1.fit(['icmp' 'tcp' 'udp'])
    labelencoder_x_2 = labelencoder_x_2.fit(['IRC' 'X11' 'Z39_50' 'aol' 'auth' 'bgp' 'courier' 'csnet_ns' 'ctf'
 'daytime' 'discard' 'domain' 'domain_u' 'echo' 'eco_i' 'ecr_i' 'efs'
 'exec' 'finger' 'ftp' 'ftp_data' 'gopher' 'harvest' 'hostnames' 'http'
 'http_2784' 'http_443' 'http_8001' 'imap4' 'iso_tsap' 'klogin' 'kshell'
 'ldap' 'link' 'login' 'mtp' 'name' 'netbios_dgm' 'netbios_ns'
 'netbios_ssn' 'netstat' 'nnsp' 'nntp' 'other' 'pm_dump' 'pop_2' 'pop_3'
 'printer' 'private' 'remote_job' 'rje' 'shell' 'smtp' 'sql_net' 'ssh'
 'sunrpc' 'supdup' 'systat' 'telnet' 'tim_i' 'time' 'urp_i' 'uucp'
 'uucp_path' 'vmnet' 'whois'])
    labelencoder_x_3 = labelencoder_x_1.fit(['OTH' 'REJ' 'RSTO' 'RSTOS0' 'RSTR' 'S0' 'S1' 'S2' 'S3' 'SF' 'SH'])
    dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')
    T = dataset.iloc[:, 41].values
    labelencoder_yBN = LabelEncoder()
    labelencoder_yBN=labelencoder_yBN.fit(['attack','normal.'])
    y_test=labelencoder_yBN.fit_transform(T)
    X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_x_3.fit_transform(X[:, 3])
    scaler = Normalizer().fit(X) 
    testX= scaler.transform(X)

    classifier = Sequential()
    classifier.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', input_dim = 41))
    classifier.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    #classifier.summary() 
    json_file = open('binaryAnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("binaryAnn.h5")
    print("Loaded model binaryAnn")
    y_pred = classifier.predict(testX)
    print(y_pred)
    y_pred1= (y_pred > 0.6)
    print(y_pred1)
    con_mat(y_pred1,y_test)
   
anamoly()    
signatureM(testX)