import numpy as np

class Stream(object):
    ''' Base class for all streams
    '''
    def __init__(self, metadata = None):
        if metadata is None:
            metadata = {'StreamEnded': False}
        self.metadata = metadata
        self._stream_id = self.__class__.__name__ + '-' + ''.join(map(str, (np.random.rand(10)*10).astype('int')))
            
    def close(self):
        pass

    def read(self, n = None):
        pass

    def pause(self):
        pass

    def resume(self):
        pass
    
    def __iter__(self):
        return self
    
    def next(self):
        data = self.read()
        if self.metadata['StreamEnded']:
            raise StopIteration
        else:
            return data

    def __exit__(self, type, value, tb):
        self.close()
        
    def __enter__(self):
        return self

    def _setflag_(self, name, value, reason = None):
        try:
            if self.metadata[name] != value:
                self.metadata[name] = value
                self.metadata[name + 'Setter'] = self._stream_id
                self.metadata[name + 'Reason'] = reason
        except KeyError:
            self.metadata[name] = value
        
    def start(self):
        pass

    def stop(self):
        pass
    

class SStream(Stream):
    ''' Base class for sound streams
    '''
    def __init__(self, periodsize = 128, stepsize = None, *args, **kwargs):
        super(SStream, self).__init__(*args, **kwargs)
        self.periodsize = periodsize
        if not stepsize:
            stepsize = periodsize
        self.stepsize = stepsize
        self.current = 0
        if 'Reset' not in self.metadata.keys():
            self._setflag_('Reset', False)
        self._reset_()
        self.position = 0

    def read(self, n = None):
        '''
        '''
        if self.metadata['Reset'] and self.metadata['ResetSetter'] == self._stream_id and self._reset_processed_:
            self._setflag_('Reset', False)
        if self.metadata['Reset']:
            self.sstream_data = self._read_()
            self.position += self.periodsize
            self._reset_processed_ = True
            if self.metadata['StreamEnded']:
                return None
            else:
                return self.sstream_data
        else:
            data = self._read_(self.stepsize)
            if self.metadata['StreamEnded']:
                return None
            self.sstream_data = np.hstack( ( self.sstream_data[self.stepsize:], data ) )
            self.position += self.stepsize
            return self.sstream_data

    def _read_(self):
        pass

    def _reset_(self):
        if not self.metadata['Reset']:
            self._setflag_('Reset', True)
            self._reset_processed_ = False

    def start(self):
        self._setflag_('StreamEnded', False)
        
    def stop(self):
        self._setflag_('StreamEnded', True)
        
class SubStream(Stream):
    def __init__(self, stream):
        self.stream = stream
        super(SubStream,self).__init__(metadata = self.stream.metadata)
        
    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
