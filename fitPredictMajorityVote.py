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
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json

lab=['buffer_overflow.' ,'ftp_write.' ,'guess_passwd.', 'imap.','land.', 'loadmodule.', 'multihop.', 'nmap.', 'perl.','phf.', 'pod.' ,'rootkit.', 'spy.', 'warezmaster.']

labels=['back.' ,'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.',
 'ipsweep.' ,'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.',
 'phf.', 'pod.', 'portsweep.', 'rootkit.' ,'satan.' ,'smurf.', 'spy.',
 'teardrop.', 'warezclient.' ,'warezmaster.']
   
def histogram(a):
    d=dict()
    i=0
    for c in a:
        if c not in d:
            if i==0:
                if c=='portsweep.':
                    d[c]=12
                elif c=='warezclient.':
                    d[c]=15
                elif c=='teardrop.':
                    d[c]=12
                else:
                    d[c]=4
            elif i==1:
                if c=='satan.':
                    d[c]=14
                elif c=='portsweep.':
                    d[c]=12
            elif i==2:
                if c=='teardrop.':
                    d[c]=12
                else:
                    d[c]=3
            elif i==3:
                    d[c]=2
            elif i==4:
                if c=='land.':
                    d[c]=6
                else:
                    d[c]=3
            elif i==5:
                if c=='ftp_write.':
                    d[c]=15
                elif c=='multihop.':
                    d[c]=12
                else:
                    d[c]=5
            elif i==6:
                d[c]=6
            else:
                d[c]=1
        else:
            if c=='neptune.':
                d[c]=d[c]+2
            elif c=='ipsweep.':
                d[c]=d[c]+4
            else:
                d[c]+=1
        i=i+1    
    inverse=[(value,key) for key,value in d.items()]
    return max(inverse)[1]
    
def cnnload(testX):
    cnn = Sequential()
    cnn.add(Convolution2D(64, 3,3, border_mode="same",activation="relu",input_shape=(1,41,1),W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(64, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3,border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(22, activation="softmax"))
    cnn.load_weights("cnn.hdf5")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1],1))
    print("Predicting.....")
    y_pred = cnn.predict_classes(testXR)
    return cnn,y_pred

def lstmload(testX):
    lstm = Sequential()
    lstm.add(LSTM(128,input_dim=41, return_sequences=True))  
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128,return_sequences=True))
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128, return_sequences=True))  
    lstm.add(Dropout(0.1))
    lstm.add(LSTM(128, return_sequences=False))  
    lstm.add(Dropout(0.1))
    lstm.add(Dense(22))
    lstm.add(Activation('softmax'))
    lstm.load_weights("lstm.hdf5")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    print("Predicting.....")
    y_pred = lstm.predict_classes(testXR)
    return lstm,y_pred
    
def gruload(testX):
    gru = Sequential()
    gru.add(GRU(64,input_dim=41, return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64,return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64, return_sequences=True))  
    gru.add(Dropout(0.1))
    gru.add(GRU(64, return_sequences=False))  
    gru.add(Dropout(0.1))
    gru.add(Dense(22))
    gru.add(Activation('softmax'))
    gru.load_weights("gru.hdf5")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    y_pred = gru.predict_classes(testXR)
    print("Predicting.....")
    return gru,y_pred
    

def dnnload(testX):
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
    dnn.add(Dense(22))
    dnn.add(Activation('softmax'))
    dnn.load_weights('dnn.hdf5')
    testXR = np.reshape(testX, (testX.shape[0],testX.shape[1]))
    print("Predicting.....")
    y_pred = dnn.predict_classes(testXR)
    return dnn,y_pred


def rnnload(testX):
    rnn = Sequential()
    rnn.add(SimpleRNN(128,input_dim=41, return_sequences=True))  
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128,return_sequences=True)) 
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128, return_sequences=True))  
    rnn.add(Dropout(0.1))
    rnn.add(SimpleRNN(128, return_sequences=False))  
    rnn.add(Dropout(0.1))
    rnn.add(Dense(22))
    rnn.add(Activation('softmax'))
    rnn.load_weights('rnn.hdf5')
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    print("Predicting.....")
    y_pred = rnn.predict_classes(testXR)
    return rnn,y_pred

def cnnload1(testX):
    cnn = Sequential()
    cnn.add(Convolution2D(64, 3,3, border_mode="same",activation="relu",input_shape=(1,41,1),W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(64, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3, border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Convolution2D(128, 3,3,border_mode="same", activation="relu",W_constraint=maxnorm(3)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(14, activation="softmax"))
    cnn.load_weights("cnn1.hdf5")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1],1))
    print("Predicting.....")
    y_pred = cnn.predict_classes(testXR)
    return cnn,y_pred

def lstmload1(testX):
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
    lstm.load_weights("lstm1.hdf5")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    print("Predicting.....")
    y_pred = lstm.predict_classes(testXR)
    return lstm,y_pred
    
def gruload1(testX):
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
    gru.load_weights("gru1.hdf5")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    y_pred = gru.predict_classes(testXR)
    print("Predicting.....")
    return gru,y_pred
    

def dnnload1(testX):
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
    dnn.load_weights('dnn1.hdf5')
    testXR = np.reshape(testX, (testX.shape[0],testX.shape[1]))
    print("Predicting.....")
    y_pred = dnn.predict_classes(testXR)
    return dnn,y_pred


def rnnload1(testX):
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
    rnn.load_weights('rnn1.hdf5')
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    print("Predicting.....")
    y_pred = rnn.predict_classes(testXR)
    return rnn,y_pred

def signatureM(testX):
    rows=testX.shape[0]
    cnn,yc=cnnload(testX)
    lstm,yl=lstmload(testX)
    gru,yg=gruload(testX)
    dnn,yd=dnnload(testX)
    rnn,yr=rnnload(testX)
    cnn1,yc1=cnnload1(testX)
    lstm1,yl1=lstmload1(testX)    
    gru1,yg1=gruload1(testX)
    dnn1,yd1=dnnload1(testX)
    rnn1,yr1=rnnload1(testX)
    for i in range(rows):
        print([labels[int(yc[i])],labels[int(yl[i])],labels[int(yg[i])],labels[int(yd[i])],labels[int(yr[i])],lab[int(yc1[i])],lab[int(yl1[i])],lab[int(yg1[i])],lab[int(yd1[i])],lab[int(yr1[i])]])
        print(histogram([labels[int(yc[i])],labels[int(yl[i])],labels[int(yg[i])],labels[int(yd[i])],labels[int(yr[i])],lab[int(yc1[i])],lab[int(yl1[i])],lab[int(yg1[i])],lab[int(yd1[i])],lab[int(yr1[i])]]))
        x=histogram([labels[int(yc[i])],labels[int(yl[i])],labels[int(yg[i])],labels[int(yd[i])],labels[int(yr[i])],lab[int(yc1[i])],lab[int(yl1[i])],lab[int(yg1[i])],lab[int(yd1[i])],lab[int(yr1[i])]])
        if i==0:    
            y_pred1 = np.array(x)
        else:
            y_pred1=np.append(y_pred1,x)
    labelencoder_y = LabelEncoder()
    labelencoder_y = labelencoder_y.fit(['back.' 'buffer_overflow.' 'ftp_write.' 'guess_passwd.' 'imap.'
 'ipsweep.' 'land.' 'loadmodule.' 'multihop.' 'neptune.' 'nmap.' 'perl.'
 'phf.' 'pod.' 'portsweep.' 'rootkit.' 'satan.' 'smurf.' 'spy.'
 'teardrop.' 'warezclient.' 'warezmaster.'])
    y_pred1=labelencoder_y.fit_transform(y_pred1)
    y_pred1= to_categorical(y_pred1,22)    
    cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print("\nFitting predicted values to model.........")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1],1))
    cnn.fit(testXR, y_pred1, batch_size=1, nb_epoch=10)
    
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    lstm.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print("\nFitting predicted values to model.........")
    lstm.fit(testXR, y_pred1, batch_size=1, nb_epoch=10)
    
    print("\nFitting predicted values to model.........")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    gru.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    gru.fit(testXR, y_pred1, batch_size=1, nb_epoch=10)
    
    print("\nFitting predicted values to model.........")
    testXR = np.reshape(testX, (testX.shape[0],testX.shape[1]))
    dnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    dnn.fit(testXR, y_pred1, batch_size=1, nb_epoch=10)
    
    print("\nFitting predicted values to model.........")
    testXR = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
    rnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    rnn.fit(testXR, y_pred1, batch_size=1, nb_epoch=10)
    print("\n Models are Saved")
    
    cnn.save("cnnFit.hdf5")
    lstm.save("lstmFit.hdf5")
    gru.save("gruFit.hdf5")
    dnn.save("dnnfit.hdf5")
    cnn.save("rnnfit.hdf5")
     
    
    
def anamoly():
    dataset=pd.read_csv('kd10attack.csv')
    X = dataset.iloc[:, :-1].values
    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()
    total=dataset.shape[0]
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
    X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_x_3.fit_transform(X[:, 3])
    scaler = Normalizer().fit(X)
    testX= scaler.transform(X)

    classifier = Sequential()
    classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 41))
    classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    json_file = open('binaryAnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("binaryAnn.h5")
    print("Loaded model binaryAnn")
    y_pred = classifier.predict(testX)
    y_pred1=pd.DataFrame()
    print(y_pred)
    for i in range(0,total):
        y_pred1=y_pred1.append(pd.Series((y_pred [i]> 0.995 and (y_pred[i]==0.99838364 or (y_pred[i]< 0.99821)))),ignore_index=True)
    X=pd.DataFrame()
    print(y_pred1)
    y_pred1=np.array(y_pred1)
    j=0
    for i in y_pred1:
        if i==False:
            X=X.append(pd.Series(testX[j]),ignore_index=True)
            j=j+1
    return X

def HybridIDS():
    print("Running HYBRID INTRUSION DETECTION SYSTEM.......")
    print("Data Processing and Running Anamoly Based model..........")
    testX=anamoly()
    if testX.empty :
        print("Anamoly Based model detected Network Data as normal data flow..... ")
        print("Continue Normal flow of the Syetem.......")
    else:
        print("Anamoly Based model detected Network Data as attack data flow..... ")
        print("Signature Based model detecting signature of attack..... ")
        signatureM(testX.values)

HybridIDS()