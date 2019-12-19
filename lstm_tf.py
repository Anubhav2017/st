import tensorflow as tf 
import numpy as np 

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary



vocab_size=100
n_hidden=256

dictionary, reverse_dictionary=build_dataset(vocab_size)

def RNN(x, weights, biases):
	x=tf.reshape(x, [1,n_input])
	x=tf.split(x, n_input, 1)

	rnn_cell=rnn.BasicLSTMCell(n_hidden)

	outputs, states=rnn.static_rnn(rnn_cell, x, dtype=float32)

	return tf.matmul(outputs[-1],weights['out'])+biases['out']
# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random.normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random.normal([vocab_size]))
}

symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
symbols_out_onehot = np.zeros([vocab_size], dtype=float)
symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0

session=tf.Session()

_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)