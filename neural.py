from nwCostFunc import *
from scipy.optimize import fmin_cg
import numpy as np
import cPickle

class nw_Struct():
    def __init__(self,size,lam):
        self.in_size , self.hi_size , self.label_size = size
        self.lam = lam


class neural_network():
    def __init__(self,nw_S):
        self.nw_Struct = nw_s
        self.Cost = nwCostFunc.nwCost(nw_S)
        self.compute_plen()

    def compute_plen(self):
        nS = self.nw_Struct
        n1 , n2 , n3 = nS.in_size , nS.hide_size , nS.label.size
        self.p_len = n2*(n1 + 1) + n3*(n2 + 1)

    def unroll(self,param):
        return self.Cost.unroll(param)

    def enroll(self,t):
        return self.Cost.enroll(t)

    def load_train_data(self,train_data):
        self.X ,self.y = train_data
        self.Cost.load_data(train_data)

    def train(self,train_data):
        init_p = rand_arr(self.p_len)
        self.load_train_data(train_data)
        nw_f = self.Cost.Jcost
        nw_fprime = self.Cost.JG
        opt_param = fmin_cg(nw_f,init_p,nw_fprime)
        J  = self.Cost.Jcost(opt_param)
        print 'finished training the neural network'
        print 'Cost on training data set is %f'%J
        self.param = opt_param
        Theta1 , Theta2 = self.unroll(self.param)
        self.Theta1 , self.Theta2 = Theta1 , Theta2

    def store_data(self,fname = 'neural_data.txt'):
        with open(fname,'w') as f:
            cPickle.dump(self.nw_Struct,f)
            cPickle.dump(self.param,f)

    def predit(self,X):
        m = np.size(X,0)
        T1 , T2 = self.Theta1 , self.Theta2
        a0 = np.hstack(ones(m,1),X)
        h1 = sigmoid( np.dot(a0 , T1.transpose()) ) 
        a1 = np.hstack(ones(m,1),h1)
        h2 = sigmoid( np.dot(a1 , T2.transpose()) )
        pred = np.argmax(h2,1)
        return pred 


class nw_model(neural_network):
    def __init__(self,fname):
        self.load_from_file(self,fname)
        self.Cost = nwCostFunc.nwCost(self.nw_Struct)
        self.Theta1 , self.Theta2 = self.unroll(self.param)
        self.compute_plen()

    def load_from_file(self,fname):
        with open(fname,'r') as f:
            self.nw_Struct = cPickle.load(f)
            self.param = cPickle.load(f)


       










