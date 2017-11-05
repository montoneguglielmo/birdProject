import wave
import numpy as np
import gzip
import cPickle as pickle
from wavestream import WaveStream
from feature import *
from llaf import *
import matplotlib.pyplot as plt
from utils import file_in_subfold

if __name__ == '__main__':

    nameBirdsTrash = ['1296_calls', '1440_calls', '1511_calls', '1513_calls', 'trash']
    
    fileBirdsTrash = []
    fileBirdsTrash = file_in_subfold('data', fileBirdsTrash, '.wav')

    bt = [[], [], [], [], []]

    periodsize = (int)(2.0)**5 # the rate at which the file are recorder is 44000Hz
    num_der    = 0

    feature       = LLAFSimpleBank(windowsize = periodsize)
    diff_feature  = FeatureDerivative(feature = feature, n = num_der)

    
    for namefl in fileBirdsTrash:
        print namefl
        indx = [nB  in namefl for nB in nameBirdsTrash]
        indx = np.where(np.asarray(indx))[0][0]
        
        stream         = WaveStream(filename = namefl, periodsize = periodsize)
        feature_stream = FeatureStream(feature = diff_feature, stream = stream)
        stream.start()
        dat = feature_stream.read()
        speech = []
    
        while dat is not None:
            dat = feature_stream.read()
            speech.append(dat)

        speech = np.vstack(speech[0:-1])
        
        bt[indx].append(speech)

        

    #make all the data of the same lenght    
    min_shape = np.inf
    for bird in bt:
        for call in bird:
           min_shape = min(min_shape, call.shape[0])
           
    dataset = []           
    for bird in bt:
        dt = []
        for call in bird:
            indmx = np.where(call == call.max())[0][0]
            indbt = indmx
            indtp = indmx

            while True:
                if (indbt > 0):
                    indbt -= 1

                if (indtp - indbt) == min_shape:
                    break

                if (indtp < call.shape[0]):
                    indtp += 1

                if (indtp - indbt) == min_shape:
                    break

            dt.append(call[indbt:indtp,:])
        
        dataset.append(dt)


    dataBirdTrash = {}
    for nm, dt in zip(nameBirdsTrash, dataset):
        print nm
        dataBirdTrash[nm] = dt


    with gzip.open('dataBirdTrash.pkl.gz', 'wb') as f:
        pickle.dump(dataBirdTrash, f, protocol=pickle.HIGHEST_PROTOCOL)
