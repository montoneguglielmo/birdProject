import numpy as np
from melmisc import meluniform, mel_scale_windows
from scipy.fftpack import dct
from feature import Feature
from scipy.signal import lfilter

class LLAF(Feature):
    def __init__(self, windowsize, *args, **kwargs):
        super(LLAF,self).__init__(*args, **kwargs)
        self.windowsize = windowsize
    def do(self, data):
        return super(LLAF,self).do(data)
    
class LLAFSimpleBank(LLAF):
    def __init__(self, *args, **kwargs):
        super(LLAFSimpleBank, self).__init__(*args, **kwargs)
        self.window = np.hamming(self.windowsize)
    def do(self, data):
        feature = super(LLAFSimpleBank,self).do(data)
        if feature is not None:
            return np.log( abs( np.fft.rfft(self.window * data) ) + 1e-50 )
        else:
            return None
    
class LLAFMelBank(LLAF):
    def __init__(self, n_mels, F, *args, **kwargs):
        super(LLAFMelBank, self).__init__(*args, **kwargs)
        self.window = np.hamming(self.windowsize)
        self.melwindows = np.asarray(mel_scale_windows(n = self.windowsize / 2 + 1, F_max = F, num = n_mels)).T
        
    def do(self, data):
        feature = super(LLAFMelBank,self).do(data)
        if feature is not None:
            E = abs( np.fft.rfft(self.window * feature) )
            M = np.dot( E, self.melwindows )
            return np.log(M + 1e-50)
        else:
            return None

class LLAFAutocorr(LLAF):
    def __init__(self, *args, **kwargs):
        super(LLAFAutocorr, self).__init__(*args, **kwargs)
        
    def do(self, data):
        #print data.shape
        feature = super(LLAFAutocorr,self).do(data)
        if feature is not None:
            ac = np.correlate(data, data, 'full')
            ac = ac[self.windowsize-1:]
            ac = ac / ac[0]
            return ac
        else:
            return None
    def reset(self):
        pass
    
class LLAF_MFCC_naive(LLAF):
    def __init__(self, n_mels, F, *args, **kwargs):
        super(LLAF_MFCC_naive, self).__init__(*args, **kwargs)
        self.window = np.hamming(self.windowsize)
        self.melwindows = np.asarray(mel_scale_windows(n = self.windowsize / 2 + 1, F_max = F, num = n_mels)).T
        
    def do(self, data):
        feature = super(LLAF_MFCC_naive,self).do(data)
        if feature is not None:
            x = self.window * data
            E = abs( np.fft.rfft( x ) )
            M = np.dot( E, self.melwindows )
            d = dct( np.log(M + 1e-50), type = 3 )
            return d
        else:
            return None

class LLAF_MFCC(LLAF):
    def __init__(self, n_cep = 13, F_max = 8000, n_mflt = 20, ceplifter = 22, k = .97, rate = 16000, *args, **kwargs):
        super(LLAF_MFCC, self).__init__(*args, **kwargs)

        self.window = np.hanning(self.windowsize)
        self.k = .97
        self.n_cep = n_cep
        self.lifter = np.hstack( [1 + .5*ceplifter * np.sin(np.arange(1,n_cep)*np.pi/ceplifter), np.asarray([1])] )

        # prepare Mel filters
        fft_freqs = np.arange(self.windowsize/2) / float(self.windowsize) * rate
        bin_freqs = meluniform(F_min = 0, F_max = F_max, num = n_mflt+2)
        mel_windows = np.zeros( (n_mflt, self.windowsize) )
        for cnt in range(n_mflt):
            left, center, right = tuple(bin_freqs[cnt:cnt+3])
            l_slope = (fft_freqs - left) / (center-left)
            r_slope = (right - fft_freqs) / (right-center)
            wnd = np.max(np.vstack( [np.zeros_like(fft_freqs), np.min(np.vstack([l_slope, r_slope]),axis=0)]), axis=0)
            mel_windows[cnt,0:wnd.shape[0]] = wnd
        self.mel_windows = mel_windows[:, :self.windowsize/2+1]

        # prepare dct filters
        dctm = []
        for cnt in range(n_cep):
            dctm.append( np.cos(np.pi*cnt*np.arange(1.,2.*n_mflt,2.)/(2.*n_mflt)) * np.sqrt(2./n_mflt) )
        self.dctm = np.asarray(dctm[1:] + [dctm[0]])
       
    def do(self, data):
        feature = super(LLAF_MFCC,self).do(data)
        if feature is not None:
            x = self.window * lfilter(np.asarray([1., -self.k]), 1., data * 2**15)
            spec = abs( np.fft.rfft(x) )
            aspec = np.dot(self.mel_windows, spec)**2
            cep = np.dot(self.dctm,np.log(aspec)) * self.lifter
            return cep
        else:
            return None
