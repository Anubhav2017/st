import numpy as np 
from keras.layers import Input, TimeDistributed, LSTM, Dense
from keras.models import Model 
from attention_keras.layers.attention import AttentionLayer

l=10
batch_size=100
x=np.arange(l)/l

x=x.reshape(1,l,1)
y=2*np.arange(l)/l 
y=y.reshape(1,l,1)

encoder_inputs=Input(shape=(None,1))
encoder_lstm=LSTM(20,return_sequences=True,return_state=True)
enc_lstm_out,state_h, state_c=encoder_lstm(encoder_inputs)

enc_states=[state_h,state_c]


decoder_inputs=Input(shape=(None,1))

decoder_lstm=LSTM(20, return_sequences=True,return_state=True)

decoder_out_lstm,_,_=decoder_lstm(decoder_inputs,initial_state=enc_states)

attn_layer=AttentionLayer(name="attn_layer")
attn_out,attn_states=attn_layer([enc_lstm_out,decoder_out_lstm])


decoder_dense_out=TimeDistributed(Dense(1))
decoder_out=decoder_dense_out(decoder_out_lstm)

model=Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_dense_out)