import numpy as np 
from keras.layers import Input, TimeDistributed, LSTM, Dense
from keras.models import Model 

l=10
batch_size=100
x=np.arange(l)/l

x=x.reshape(1,l,1)
y=2*np.arange(l)/l 
y=y.reshape(1,l,1)

encoder_inputs=Input(shape=(None,1))
encoder_lstm=LSTM(20,return_state=True)
enc_lstm_out,state_h, state_c=encoder_lstm(encoder_inputs)

enc_states=[state_h,state_c]

decoder_inputs=Input(shape=(None,1))

decoder_lstm=LSTM(20, return_sequences=True,return_state=True)

decoder_out_lstm,_,_=decoder_lstm(decoder_inputs,initial_state=enc_states)

decoder_dense_out=TimeDistributed(Dense(1))
decoder_out=decoder_dense_out(decoder_out_lstm)

model=Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_dense_out)

encoder_model=Model(encoder_inputs,enc_states)

decoder_state_input_h=Input(shape=(l,))
decoder_state_input_c=Input(shape=(l,))
decoder_states_inputs=[decoder_state_input_h,decoder_state_input_c]

decoder_out_lstm, dec_state_h, dec_state_c=decoder_lstm(decoder_inputs,initial_state=decoder_states_inputs)
decoder_states=[dec_state_h,dec_state_c]

decoder_outputs=decoder_dense_out(decoder_out_lstm)

decoder_model=Model([decoder_inputs]+decoder_states_inputs,[decoder_outputs]+decoder_states)


def decode_sequence(input_seq):
	states_value=encoder_model.predict(input_seq)

