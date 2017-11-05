import numpy as np
from sstream import SubStream

class FeatureStream(SubStream):
    ''' Stream for features
    '''
    def __init__(self, stream, feature):
        super(FeatureStream, self).__init__(stream)
        if 'Reset' not in self.metadata.keys():
            self._setflag_('Reset', False)
        self.feature = feature
    def read(self, n = None):
        while True:
            data = self.stream.read()
            if self.metadata['StreamEnded']:
                return None
            if self.metadata['Reset']:
                self.feature.reset()
            feature = self.feature.do(data)
            if feature is not None:
                break
        return feature

    def set_position(self, pos):
        self.stream.set_position(pos)
        self.feature.set_position(pos)

    def get_position(self):
        return self.stream.get_position()
        
    def pause(self):
        self.stream.pause()

    def resume(self):
        self.stream.resume()

    def __getattr__(self, k):
        return getattr(self.stream, k)

class Feature(object):
    def __init__(self, feature = None):
        if feature is not None and not isinstance(feature,Feature):
            raise TypeError('feature must be of Feature type; received: %s' % type(feature))
        self.feature = feature
        
    def do(self, data):
        if self.feature is not None:
            return self.feature.do(data)
        else:
            return data
    
    def reset(self):
        pass

    def set_position(self, pos):
        pass

class AccumulateFeature(Feature):
    def __init__(self, n, *args, **kwargs):
        super(AccumulateFeature,self).__init__(*args, **kwargs)
        self.n = n
        if not self.n:
            raise ValueError("Can't use n = 0 in AccumulateFeature")
        self.prev = []
    def do(self, data):
        feature = super(AccumulateFeature,self).do(data)
        if feature is None:
            return None
        else:
            self.prev.append(feature)
            if len(self.prev) > self.n:
                del(self.prev[0])
            if len(self.prev) == self.n:
                return  np.concatenate( self.prev )
            else:
                return None

    def reset(self):
        self.prev = []

class FeatureDiff(Feature):
    def __init__(self, n, *args, **kwargs):
        super(FeatureDiff,self).__init__(*args, **kwargs)
        self.n = n
        self.prev = []
    def do(self, data):
        feature = super(FeatureDiff,self).do(data)
        if feature is None:
            return None
        self.prev.append( feature )
        if len(self.prev) <= self.n:
            return None
        else:
            f0 = self.prev[0]
            self.prev = self.prev[1:]
            df = [f-f0 for f in self.prev]
            return np.concatenate( [f0] + df )
    def reset(self):
        self.prev = []

class FeatureDerivative(Feature):
    def __init__(self, n, *args, **kwargs):
        super(FeatureDerivative,self).__init__(*args, **kwargs)
        self.n = n
        self.prev = []
    def do(self, data):
        feature = super(FeatureDerivative,self).do(data)
        if feature is None:
            return None
        if not len(self.prev):
            self.prev = [ [feature] ]
            return None
        else:
            self.prev[0].append(feature)
            try:
                for cnt in range(1,self.n+1):
                    if len(self.prev) == cnt:
                        self.prev.append( [] )
                    self.prev[cnt].append(self.prev[cnt-1][-1]-self.prev[cnt-1][-2])
            except:
                return None
            ret_val = np.concatenate( [prev[-1] for prev in self.prev] )
            for cnt in range(0,self.n+1):
                self.prev[cnt] = self.prev[cnt][1:]
            return ret_val

    def reset(self):
        self.prev = []

class ThresholdEventFeature(Feature):
    def __init__(self, upper, lower, *args, **kwargs):
        super(ThresholdEventFeature,self).__init__(*args, **kwargs)
        self.upper = upper
        self.lower = lower
        self.feature_active = np.zeros_like(self.upper).astype(bool)

    def do(self, data):
        feature = super(ThresholdEventFeature,self).do(data)
        if feature is None:
            return None
        event = np.logical_and( feature > self.upper, np.logical_not(self.feature_active) )
        self.feature_active = np.logical_or( np.logical_and( feature > self.lower, self.feature_active ), feature > self.upper )
        return event
