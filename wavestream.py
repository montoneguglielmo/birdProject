import numpy as np
from sstream import *
import wave

class WaveStream(SStream):
    ''' Stream for an audiofile
    '''
    def __init__(self, filename = None, rate = None, *args, **kwargs):
        super(WaveStream, self).__init__(*args, **kwargs)
        self.set_wavefile(filename, rate)
        
    def set_wavefile(self, filename, rate = None):
        self.filename = filename
        if self.filename is None:
            self.wav = None
            self.rate = rate
            return
        self.wav = wave.open(self.filename,'rb')
        self.rate = self.wav.getframerate()
        self.channels = self.wav.getnchannels()
        if self.wav.getsampwidth() != 2:
            print 'Only 16 bit files supported'
            raise IOError
        self._setflag_('StreamEnded', False)
        self._reset_()

    def set_position(self, pos):
        if self.wav is not None:
            self.wav.setpos(pos * 1e-3 * self.rate)
            self.position = int(pos * 1e-3 * self.rate)
            self._setflag_('StreamEnded', False)
            self._reset_()
        else:
            self._setflag_('StreamEnded', True, 'No wav specified')

    def get_position(self):
        if self.wav is not None:
            return self.position * 1e3 / self.rate
        else:
            return None
            
    def start(self):
        super(WaveStream, self).start()
        if self.wav is None:
            self._setflag_('StreamEnded', True, 'No wav specified')
        
    def _read_(self, n = None):
        if self.metadata['StreamEnded'] or self.wav is None:
            return None
        if n is None:
            n = self.periodsize
        dat = self.wav.readframes(n)
        if dat == '':
            self._setflag_( 'StreamEnded', True, 'Empty wav data; possibly EOF')
            return None
        dat = np.fromstring(dat, dtype=np.int16)[::self.channels].astype(np.float32) / 2**15
        if dat.shape[0] != n:
            self._setflag_( 'StreamEnded', True, 'Wrong wav data shape: %d, expected: %d; possibly EOF' % (dat.shape[0], n) )
            return None
        return dat
        
    def close(self):
        #self.wav.close()
        pass
