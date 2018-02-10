import numpy as np
import keras
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Lambda, Layer
from keras.layers import Reshape,Flatten,Activation,RepeatVector
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM
from keras import backend as K 

def LinearEncodeModel(filters=32,input_shape=(28,28,1),trainable=True):
    x = Input(shape=input_shape)
    h = Flatten()(x)
    y = Dense(filters,trainable=trainable)(h)
    return Model(inputs=[x],outputs=[y])

def LSTMDecodeModel(steps,units=32,input_size=32,trainable=True):
    x = Input(shape=(input_size,))
    h = RepeatVector(steps)(x)
    h = LSTM(units,return_sequences=True,unroll=True,trainable=trainable,implementation=2)(h)
    y = Conv1D(3,1,trainable=trainable)(h)
    return Model(inputs=[x],outputs=[y])

class DrawImageLayer(Layer):
    def __init__(self, size=28, return_sequences=False, mode="position", **kwargs):
        self.size = size
        self.return_sequences = return_sequences
        self.range = K.constant(np.arange(size).reshape(1,size)/size-0.5)
        self.mode = mode

        super(DrawImageLayer, self).__init__(**kwargs)

    def build(self, input_shape):        
        super(DrawImageLayer, self).build(input_shape) 

    def call(self, x):
        initial_states = self.get_initial_states(x)
        last_output, outputs, states = K.rnn(self.step, x, initial_states)
        if self.return_sequences:
            return outputs
        else:
            return last_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.size,self.size,1)

    def step(self, inputs, states):
        ## input: channels=3. x/vx,y/vy,touch
        ## states: len=3. x,y,image
        
        # dynamics (potision input)
        if self.mode == "position":
            x = K.reshape(inputs[:,0],(-1,1))
            y = K.reshape(inputs[:,1],(-1,1))

        # dynamics (velocity input)
        if self.mode == "velocity":
            dt = 1
            x = states[0]+dt*K.reshape(inputs[:,0],(-1,1))
            y = states[1]+dt*K.reshape(inputs[:,1],(-1,1))

        # pen profile
        sigma = 2/self.size
        g = (1/sigma)**2
        px = K.exp( -K.pow(self.range-x,2)*g)
        py = K.exp( -K.pow(self.range-y,2)*g)
        px = K.reshape(px,(-1,1,self.size))
        py = K.reshape(py,(-1,self.size,1))

        # draw
        image = K.maximum(states[2], K.reshape(inputs[:,2],(-1,1,1))*px*py )
        image = K.minimum(image, 1.0)

        new_states = [ x,y,image ]
        return K.expand_dims(image), new_states

    def get_initial_states(self,inputs):
        # build an all-zero tensor of shape (samples,)
        z = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        z = K.sum(z, axis=(1, 2))  # (samples,)

        ## dynamics 
        d = K.reshape(z,(-1,1))

        ## image
        z1 = K.expand_dims(z)
        image = K.reshape(K.stack([z]*self.size*self.size),(-1,self.size,self.size)) 

        return [d,d,image]

