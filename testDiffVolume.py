import wave
import numpy as np
import gzip
import cPickle as pickle
from wavestream import WaveStream
from feature import *
from llaf import *
import matplotlib.pyplot as plt
from utils import file_in_subfold



if __name__ == "__main__":

    features = False
    
    files = ['original.wav', 'volume.wav']
    fft   = []

    periodsize = (int)(2.0)**5 # the rate at which the file are recorder is 44000Hz
    num_der    = 0

    if features:
        feature       = LLAFSimpleBank(windowsize = periodsize)
        diff_feature  = FeatureDerivative(feature = feature, n = num_der)

        
    for file in files:
        stream         = WaveStream(filename = file, periodsize = periodsize)

        if features:
            feature_stream = FeatureStream(feature = diff_feature, stream = stream)
        stream.start()

        if features:
            dat = feature_stream.read()
        else:
            dat = stream._read_()
            
        speech = []
    
        while dat is not None:
            if features:
                dat = feature_stream.read()
            else:
                dat = stream._read_()
                
            speech.append(dat)

        if features:
            speech = np.vstack(speech[0:-1])
        else:
            speech = np.hstack(speech[0:-1])
            
        fft.append(speech)

    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(1,2)
    if features:
        axarr[0].imshow(fft[0])
        axarr[1].imshow(fft[1])
    else:
        axarr[0].plot(fft[0])
        axarr[1].plot(fft[1])
        
    plt.show()
