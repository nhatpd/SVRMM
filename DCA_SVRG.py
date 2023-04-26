from sklearn.metrics import accuracy_score
import numpy as np
import time 
from sklearn.utils import shuffle
from commonutil import svm_read_problem
#from libsvm.svmutil import svm_read_problem
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
import numpy as np 
import random
import warnings

warnings.filterwarnings('ignore')


class DCA_SVRG:
    def __init__(self, w0, eta = 0.1, maxIter = 1000, alpha = 0.1, theta = 5, maxTime = None, tol = None, inertial = True):

        self.w = w0
        self.eta = eta
        self.maxIter = maxIter
        self.alpha = alpha
        self.maxTime = maxTime
        self.tol = tol
        self.Time = []
        self.obj = []
        self.acc_train = []
        self.nnz = []
        self.inertial = inertial 
        self.theta = theta 
        self.acc_test =[]

    def soft_thresholding(self, v, gamma):
        return np.sign(v)*np.maximum(np.abs(v) - gamma,0)

    def update_w(self,v,gamma,loss):
        if loss == 'multilogistic':
            return self.ProxL21(v, gamma)
        else:
            return self.soft_thresholding(v,gamma)

    def fit(self, X_train, y_train, X_test =None, y_test=None, batch_size = 30, L = 0.1, loss = 'sigmoind', eval = True, inertial = False, m = 1, pr = False):

        n, d = X_train.shape
        self.K = len(np.unique(y_train))
        self.Kmin = int(np.unique(y_train).min())
        # X_train = np.c_[np.ones((n, 1)), X_train]

        t = 1
        wp = self.w
        w0 = self.w

        batch = []
        S = {}
        if loss == 'multilogistic' or loss == 'logistic':
            C = 1-1e-5
        else:
            C = 0.5 - 1e-5

        m = m 
        
        # v = 0 
        for i in range(n//batch_size):
            batch.append((i*batch_size,(i+1)*batch_size))
            # X = X_train[batch[i][0]:batch[i][1]]
            # y = y_train[batch[i][0]:batch[i][1]]
            
        v = self.grad(self.w, X_train, y_train, L, loss)/n
            
        # v = self.grad(self.w, X_train, y_train, L, loss)/n
            
        if eval:
            f_val = self.obj_func(self.w, X_train, y_train, loss)
            
            self.obj.append(f_val + self.obj_reg(self.w,loss))
            if pr:
                print('obj: ',self.obj[-1])

            self.acc_train.append(self.Comp_acc(X_train,y_train,loss))
            if X_test != None:
                self.acc_test.append(self.Comp_acc(X_test, y_test, loss))
        self.ngrad = [0]
        full_bach = 0
        for iter in range(self.maxIter):
            
            # X_train, y_train = shuffle(X_train, y_train, random_state=0)

            start_time = time.time()

            for i in range(n//batch_size):
                ttt = iter*(n//batch_size) + i
            # print('test',t)
            
            # eta = self.eta*self.learning_schedule(ttt)
                reg = self.grad_reg(self.w,loss)

                
                
                
                w_bar = self.w
                if ttt%m != 0:
                    # i = random.sample(range(n//batch_size),1)[0]

                    X = X_train[batch[i][0]:batch[i][1]]
                    y = y_train[batch[i][0]:batch[i][1]]
                    
                    # print('check',ttt,m)
                    # reg = self.alpha * self.theta * np.exp(-self.theta * np.abs(self.w))
                    
                    grad_bp = self.grad(w0, X, y, L, loss)
                    grad_b = self.grad(w_bar, X, y, L, loss)
                    grad = v + (grad_b-grad_bp)/batch_size + self.eta*L*self.w
                    full_bach += 1/(n//batch_size)
                else:
                    full_bach += 1
                    # v = 0
                    # for j in range(n//batch_size):
                    #     X = X_train[batch[j][0]:batch[j][1]]
                    #     y = y_train[batch[j][0]:batch[j][1]]
                        
                    v = self.grad(w_bar, X_train, y_train, L, loss)/n
                    # v = self.grad(w_bar, X_train, y_train, L, loss)/n
                    w0 = w_bar 
                    grad = v
                wp = self.w 
                self.w = self.update_w(grad/(self.eta*L+L),reg/(self.eta*L+L),loss)

            # self.Time.append(time.time() - start_time)
                if eval and full_bach >=1:
                    
                    self.Time.append(time.time() - start_time)
                    self.ngrad.append(self.ngrad[-1] + full_bach)
                    full_bach = 0
                    f_val = self.obj_func(self.w, X_train, y_train, loss)

                    self.obj.append(f_val + self.obj_reg(self.w,loss))
                    if pr:
                        print('iter',iter,'obj: ',self.obj[-1])
                    # self.nnz.append(np.count_nonzero(self.w))

                    # Xw = X_train.dot(self.w)
                    # y_pred = np.sign(Xw)
                    # y_pred[y_pred==0] = 1
                    self.acc_train.append(self.Comp_acc(X_train, y_train, loss))
                    if X_test != None:
                        self.acc_test.append(self.Comp_acc(X_test, y_test, loss))
            #stopping conditions
            if self.ngrad[-1] > self.maxIter:
                break
            if self.maxTime is not None:
                if np.sum(self.Time) > self.maxTime:
                    print('Stopped by MaxTime at ',iter,'-th iterattion')
                    break
            if self.tol is not None:
                if np.abs(self.obj[-1] - self.obj[-2]) <= self.tol:
                    print('Stopped by the objective value at ',iter,'-th iterattion')
                    break

    def Comp_acc(self, X, y, loss):
        n, d = X.shape
        if d == self.w.shape[0] - 1:
            X = np.c_[np.ones((n, 1)), X]
        if loss == 'multilogistic':
            y = np.array(y).astype(int)
            y = y - self.Kmin
            prob = self.softmax(X.dot(self.w))

            y_pred = np.argmax(prob, axis=1)
        else: 
            y = np.c_[y]
            y_pred = np.sign(X.dot(self.w))
            y_pred[y_pred==0] = 1
        return np.mean(y == y_pred)

    def learning_schedule(self, t):
        return 1 / (t + 1)

    def grad(self, w, X, y, L, loss):
        
        batch_size = X.shape[0]
        if loss == 'sigmoind':
            y = np.c_[y]
            aa = y*X.dot(w)
            prob = 1/(1+np.exp(aa))
            b = y*(1-prob)*prob
            grad_b = self.w*batch_size*L - X.T.dot(-b)
        elif loss == 'logistic':
            y = np.c_[y]
            aa = y*X.dot(w)
            prob = 1/(1+np.exp(-aa))
            b = y*(1-prob)
            grad_b = self.w*batch_size*L - X.T.dot(-b)
        elif loss =='NN2':
            y = np.c_[y]
            aa = y*X.dot(w)
            prob = 1/(1+np.exp(aa))
            b = 2*y*(1-prob)*(prob**2)
            grad_b = self.w*batch_size*L - X.T.dot(-b)
        elif loss == 'multilogistic':
            y = np.array(y).astype(int)
            y = y - self.Kmin
            y_train_one_hot = self.to_one_hot(y)
            XW = X.dot(w)
            prob = self.softmax(XW)
            error = prob - y_train_one_hot
            grad_b = self.w*batch_size*L - X.T.dot(error)

        return grad_b 
    def obj_func(self,w, X, y, loss):
        if loss == 'sigmoind':
            y = np.c_[y]
            Xw = X.dot(w)
            aa = y*Xw
            prob = 1/(1+np.exp(aa))
            f_val = prob.mean()
        elif loss == 'logistic':
            y = np.c_[y]
            Xw = X.dot(w)
            aa = y*Xw
            bb = np.maximum(-aa,0)
            prob = np.log(np.exp(-bb) +  np.exp(-aa-bb)) + bb
            f_val = prob.mean()
        elif loss =='NN2':
            y = np.c_[y]
            Xw = X.dot(w)
            aa = y*Xw
            prob = 1/(1+np.exp(aa))**2
            f_val = prob.mean()
        elif loss == 'multilogistic':
            y = np.array(y).astype(int)
            y = y - self.Kmin
            y_train_one_hot = self.to_one_hot(y)
            XW = X.dot(w)
            prob = self.softmax(XW)
            f_val = -np.mean(np.sum(y_train_one_hot * np.log(prob + 1e-7), axis=1))
        
        return f_val 

    def obj_reg(self,w,loss):
        if loss == 'multilogistic':
            norm2 = np.linalg.norm(w,axis=1)
            return self.alpha*(1-np.exp(-self.theta*np.abs(norm2))).sum()
        else:
            return self.alpha*(1-np.exp(-self.theta*np.abs(w))).sum()

    def grad_reg(self, w, loss):
        if loss == 'multilogistic':
            norm2 = np.linalg.norm(w,axis=1)
            reg = self.alpha * self.theta * np.exp(-self.theta * np.abs(norm2))
        else:
            reg = self.alpha * self.theta * np.exp(-self.theta * np.abs(w))
            
        return reg
    def to_one_hot(self, y):
        n = len(y)
        Y_one_hot = np.zeros((n, self.K))
        Y_one_hot[np.arange(n), y] = 1
        return Y_one_hot
    def ProxL21(self, V, gamma):
        norm2 = np.linalg.norm(V,axis=1)
        V[norm2 < gamma] = 0
        # print('len', V.shape,norm2.shape)
        V[norm2 >= gamma] = (1-gamma[norm2 >= gamma]/norm2[norm2 >= gamma]).reshape(-1,1)*V[norm2 >= gamma]
        return V
    def softmax(self, XW):
        exps = np.exp(XW)
        exp_sums = np.sum(exps, axis=1, keepdims=True)
        return exps / exp_sums
