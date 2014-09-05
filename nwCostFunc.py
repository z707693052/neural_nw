import numpy as np


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoidG(z):
    return sigmoid(z)*(1 - sigmoid(z))

def rand_arr(arr_len):
    return np.random.randn(arr_len)

# a 3 layer neural network now

class nwCost():
    def __init__(self,nw_Struct):
        self.in_size = nw_Struct.in_size
        self.hide_size = nw_Struct.hi_size
        self.label_size = nw_Struct_label_size
        self.lam = nw_Struct.lam
        self.m = np.size(X,0)


    def unroll(self,param):
        n1 , n2 , n3 = self.in_size , self.hide_size ,self.label_size
        shape1 = (n2 , n1 + 1)
        shape2 = (n3 , n1 + 1)
        Theta1 = np.reshape(param[:n1*(n2+1)],shape1)
        Theta2 = np.reshape(param[n1*(n2+1):],shape2)
        return Theta1,Theta2

    def enroll(self,t):
        return t.ravel()

    def load_data(self,train_data):
        self.X , self.y = train_data

    def cost_grad(self,Theta1,Theta2):
        m = self.m
        lam = self.lam
        X ,y = self.X , self.y
        a0 = np.hstack((ones(m,1),X)).transpose()
        z1 = np.dot(Theta1 , a0)
        a1 = sigmoid(z1)
        a1 = np.vstack(ones(1,m),a1)
        z2 = np.dot(Theta2,a1)
        hx = sigmoid(z2)
        Y = np.identity(len(y))
        Y = Y[y,].transpose()
        J = -Y*np.log10(hx) - (1 - Y)*np.log10(1 - hx)
        J = np.sum(J)/m
        J += ( lam/(2.0*m) )*(np.sum(Theta1[:,1:]) + np.sum(Theta2[:,1:]))
        d2 = hx - Y
        d1 = np.dot(Theta2.transpose(),d2)*(a1*(1 - a1)) 
        d0 = np.dot(Theta1.transpose(),d1[1:,:])*sigmoidG(a0)
        delta1 = np.dot(d2,a1.transpose())
        delta0 = np.dot(d1,a0.transpose())
        ThetaG1 = delta0[1:,:]/m
        ThetaG2 = delta1[1:,:]/m
        ThetaG = ThetaG1 , ThetaG2
        return J,ThetaG
    
    def nwcostFunction(self,param):
        Theta1 , Theta2 = self.unroll(param)
        J , (ThetaG1, ThetaG2) = self.cost_grad(Theta1,Theta2)
        TG1 , TG2 = self.enroll(ThetaG1) , self.enroll(ThetaG2)
        TG = np.vstack(T1,T2)
        self.cache = [param,J,TG]
        return J,TG

    def Jcost(self,param):
        if param == self.cache[0]:
            return self.cache[1] 
        return nwcostFunction(param)[0]

    def JG(self,param):
        if param == self.cache[0]:
            return self.chche[2]
        return nwcostFunction(param)[1]

    def num_check(self,param):
        n = np.size(param)
        numG = np.zeros(n)
        pertub = np.zeros(n)
        e = 10**-4
        for i in range(n):
            pertub[i] = e
            loss1 = Jcost(param - pertub)
            loss2 = Jcost(param + pertub)
            numG[i] = (loss2 - loss1)/(2*e)
            pertub[i] = 0
        computG = self.nwcostFunction(param)
        errG = np.abs(computG - numG)
        return np.sum(errG)/n










        

