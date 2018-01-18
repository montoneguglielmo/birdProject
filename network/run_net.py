import gzip
import cPickle as pickle
import sys
import numpy as np
import time

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Net(object):

    actNameToActFun = {'softmax' : softmax}
    
    def __init__(self, W0, b0, act0, W1, b1, act1):

        self.W0   = W0
        self.b0   = b0
        self.act0 = None##actNameToActFun[act0]
        self.W1   = W1
        self.b1   = b1
        self.act1 = self.actNameToActFun[act1]
        

    def forward(self, input):

        output = np.dot(input, W0) + b0
        
        if self.act0 is not None:
            output = self.act0(output)

        output = self.act1(np.dot(output, W1) + b1)
        
        return output


    

if __name__ == "__main__":

    name_net   = '180437.pkl.gz' #sys.argv[1]
    f          = gzip.open(name_net,'r')
    net_params = pickle.load(f)
    f.close()

    W0   = net_params[0]['hidden_layer0']['W']
    b0   = net_params[0]['hidden_layer0']['b']
    act0 = None 
    
    W1   = net_params[1]['output0']['W']
    b1   = net_params[1]['output0']['b']
    act1 = 'softmax'

    net = Net(W0, b0, act0, W1, b1, act1)


    ##load a file from bird
    with gzip.open('dataSingleBirdTrashNN.pkl.gz', 'r') as f:
        inputs, labels = pickle.load(f)

    indx   = np.random.randint(0, high=200, size=20)
    inputs = inputs[indx,:]
    labels = labels[indx]
    print "Labels for the dataset (0,1,2,3 birds; 4 is trash):", labels

    outputs = net.forward(inputs)
    print "Outputs of the network (0 is any bird; 1 is trash):", np.argmax(outputs, axis=1)


    s_time = time.time()
    for cnt in range(1000):
        output = net.forward(inputs)
    print "done in (ms):", ((time.time()-s_time)/float(cnt) * 1000.0)
