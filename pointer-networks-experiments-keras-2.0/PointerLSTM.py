import tensorflow as tf
import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine import InputSpec
from keras.layers import LSTM,RNN
from keras.layers.recurrent import Recurrent
from keras.layers import TimeDistributed 

def findmaxnotinlist(pointer, prev_outs):
  for i in range(pointer.size()):
    el=pointer[pointer.size()-i-1]
    if(el not in prev_outs):
      return el       



class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):

        super(PointerLSTM, self).__init__(*args, **kwargs)
        self.hidden_shape = hidden_shape
        self.input_length = []

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]

        self.W1 = self.add_weight(name="W1",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)

        self.W2 = self.add_weight(name="W2",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)

        self.vt = self.add_weight(name="vt",
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        #initial_states = RNN.get_initial_state(inputs=x_input)

        #constants = super(PointerLSTM, self).get_constants(x_input)
        #constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                            
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        # print "x_input:", x_input, x_input.shape
        # <TensorType(float32, matrix)>

        input_shape = self.input_spec[0].shape

        prev_outs, states1=states
        en_seq = states1[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states1[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = TimeDistributed(Dense(en_seq, self.W1, output_dim=1))
        Dij = TimeDistributed(Dense(dec_seq, self.W2, output_dim=1))
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        indices=np.argsort(pointer)
        el=findmaxnotinlist(indices.array(),prev_outs)
        prev_outs.append(el)

        return el, [prev_outs,h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
