# birdProject
Nanterre Robot Bird Project


The file to start understand the way the features are created is called createBirdDataset


The way in which I create the feature is mainly in the file llaf.py, in particular the following lines:
 
 self.window = np.hamming(self.windowsize)
    def do(self, data):
        feature = super(LLAFSimpleBank,self).do(data)
        if feature is not None:
            return np.log( abs( np.fft.rfft(self.window * data) ) + 1e-50 )
            
            
            
            
In order to create a stream of data from the microphone you should use pyaudio.
Here is an example of how to do it.

import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()


It would be great if you could manage to create a class with the same functionality of WaveStream class 
(see the file wavestream.py) that takes as input instead of a wav file the stream you create from the microphone.
In case this seem too complex you can decice to create your code for evaluating the features from the audio stream.



