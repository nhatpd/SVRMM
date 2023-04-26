from commonutil import svm_read_problem
from DCA_SAGA import *
from DCA_SVRG import *
from SDCA import *
from mmSAGA import *
from mmSVRG import *
from mmSARAH import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import json
from itertools import count

from joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[1], data[0]



methods = ['DCA-SAGA','DCA-SVRG','SDCA','MM-SAGA','MM-SVRG','MM-SARAH','DCA']

import argparse

flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="SVRMM")

flags.add_argument('--loss',type=str,default='NN2',help='Type of loss')

flags.add_argument('--typedata',type=str,default='small',help='Type of loss')
flags.add_argument('--epoch',type=int,default=20,help='Number of epochs.')
flags.add_argument('--run',type=int,default=10,help='Number of runs.')

FLAGS = flags.parse_args()

loss = FLAGS.loss

if loss == 'multilogistic':
    # datasets = ['shuttle','Sensorless','connect-4','dna']
    datasets = ['dna']
else:
    if FLAGS.typedata == 'small':
        datasets = ['a9a', 'w7a', 'w8a', 'rcv1']
    else:
        datasets = ['epsilon_normalized','avazu-app','kddb-raw','real-sim','url_combined']

    
    # datasets = ['a9a']

for dataset, ind in zip(datasets,count()):
    path = './data/'
    
    if loss == 'multilogistic':
        if dataset == 'connect-4':
            path += dataset+'.txt'
        else:
            path += dataset+'.scale'
    else:
        if dataset == 'covtype':
            path += dataset+'.libsvm.binary'
        elif dataset == 'rcv1':
            path += dataset+'_train.binary'
        elif FLAGS.typedata == 'small':
            path += dataset+'.txt'
        else:
            path += dataset 

    
    
    y_train, X_train = get_data(path)


    print('check read data')

    X_train = normalize(X_train, 'l2', axis=1, copy=False)
    print('check normalize data')
    
    # patht = './data/'
    # if (loss == 'NN2' and dataset != 'real-sim' and dataset !='url_combined') or (loss == 'multilogistic' and ind == 3):
    
        
    #     if dataset == 'rcv1':
    #         patht += dataset+'_test.binary'
    #     elif dataset == 'dna':
    #         patht += 'scale.t'
    #     else:
    #         patht += dataset + '.t'

    #     y_test, X_test = get_data(patht)

    #     print('check read test data')

    #     X_test = normalize(X_test, 'l2', axis=1, copy=False)
    

    
    theta = 5
    maxtime = 5
    batch = 1
    reg = 'exp'
    maxiter = FLAGS.epoch
    
    

    if loss == 'logistic':
        ll = 1/4
    elif loss == 'sigmoind':
        ll = 1/(6*np.sqrt(3))
    elif loss == 'NN2':
        ll = (39+55*np.sqrt(33))/2304
    elif loss == 'multilogistic':
        K = len(np.unique(y_train))
        ll = np.sqrt(K*8/27)
    print('check',X_train.shape)

    l = np.max(np.sum(X_train.multiply(X_train),1))

    l = l*ll

    print("preprocesing data")

    data = {}
    for method in methods:
        data[method] = {}
        data[method]['obj'] = []
        data[method]['time'] = []
        data[method]['train_acc'] = []
        data[method]['test_acc'] = []
    

    for iterrun in range(FLAGS.run):

        # if patht == './data/':
        test_ratio = 0.1
        total_size = X_train.shape[0]

        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size

        rnd_indices = np.random.permutation(total_size)

        X = X_train
        y = y_train
        X_train = X[rnd_indices[:train_size]]
        y_train = np.c_[y][rnd_indices[:train_size]]
        X_test = X[rnd_indices[-test_size:]]
        y_test = np.c_[y][rnd_indices[-test_size:]]
        del X
        del y
        m = X_train.shape[0]
        n = X_train.shape[1]

        alpha = 1/m 
        # X_train = np.c_[np.ones((m, 1)), X_train]
        if loss == 'multilogistic':
            w0 =  0*np.random.randn(n, K)
        else:
            w0 = 0*np.random.randn(n).reshape(-1,1)

        method = 'DCA-SAGA'
        print('run',iterrun,'method',method)
        batch = int((m**(3/4))*2*(2**(1/4)))
        model = DCA_SAGA(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=2*l, loss = loss)
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)

        method = 'DCA-SVRG'
        print('run',iterrun,'method',method)
        inner_lopp = int(m**(1/3)/(4*np.sqrt(np.exp(1)-1)))
        batch = int(m**(2/3))
        model = DCA_SVRG(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=2*l, loss = loss, m = inner_lopp)
        
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)

        method = 'SDCA'
        print('run',iterrun,'method',method)
        batch = int(m/10)
        model = SDCA(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=1.1*l, loss = loss)
        
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)

        method = 'MM-SVRG'
        print('run',iterrun,'method',method)
        inner_lopp = int(m**(1/3)/4)
        batch = int(m**(2/3))
        model = mmSVRG(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=l, loss = loss, m = inner_lopp)
        
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)

        method = 'MM-SARAH'
        print('run',iterrun,'method',method)
        inner_lopp = int(m**(1/2))
        batch = int(m**(1/2))
        model = mmSARAH(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=l, loss = loss, m = inner_lopp)
        
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)

        method = 'MM-SAGA'
        print('run',iterrun,'method',method)
        batch = int((4*m)**(2/3))
        model = mmSAGA(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=l, loss = loss)
        
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)


        method = 'DCA'
        print('run',iterrun,'method',method)
        batch = m
        model = mmSAGA(w0,eta = 0, alpha = alpha, theta =theta, maxIter=maxiter)
        model.fit(X_train,y_train,X_test,y_test,batch_size=batch, L=l, loss = loss)
        
        data[method]['obj'].append(model.obj)
        data[method]['time'].append(model.Time)
        data[method]['train_acc'].append(model.acc_train)
        data[method]['test_acc'].append(model.acc_test)

    dfjson = json.dumps(data)

    # open file for writing, "w" 
    f = open('./results/'+dataset+".json","w")

    # write json object to file
    f.write(dfjson)

    # close file
    f.close()










