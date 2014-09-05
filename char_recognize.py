import neural
from imgprocessor import *

class trans():
    def __init__(self):
        trans= {i:str(i) for i in range(10)}
        trans.update((i,chr(i-10 + ord(a))) for i in range(10,36))
        self.trans = trans
        inv_trans = {}
        for k,v in trans.iteritems():
            inv_trans[v] = k
        self.inv_trans = inv_trans
        self.ind2chars = np.frompyfunc(self.ind2char,1,1)
        self.char2inds = np.frompyfunc(self.char2ind,1,1)

    def ind2char(self,ind):
        return self.trans[ind]

    def char2ind(self,char):
        return self.in_trans[char]


class recognizer():
    def __init__(self,model):
        self.nw_model = neural.neural_network(model)
        self.translater = trans()

    def recognize(self,img):
        X = img2mat(img)
        p = self.nw_model.predict(X)
        result = self.trans.ind2char(p)
        res = ''.join(result)
        print 'result is %s'%res
        return res



