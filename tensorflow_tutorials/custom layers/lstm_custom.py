import tensorflow as tf
import numpy as np 

from tensorflow.keras.layers import LSTMCell,RNN


class HamLSTMCell(LSTMCell):

  def __init__(self,tree_size,*args, **kwargs):

    self.tree_size=tree_size
    super(HamLSTMCell, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(HamLSTMCell,self).build(input_shape)
    print(input_shape)
    self.n=input_shape[1] 
    self.embed_size=input_shape[2]
    self.ops = HAMOperations(self.embed_size, self.tree_size,self.units)
    self.tree = HAMTree(ham_ops=ops)
    self.tree.construct(n)


  def call(self, inputs, states):  

    h,c = states
  
    values = [tf.squeeze(x, [1]) for x in tf.split(inputs, self.n,1)]
    for i, val in enumerate(values):
      self.tree.leaves[i].embed(val)

    self.tree.refresh()  

    tree_out=self.tree.get_output(h)

    output, next_states=super(HAMLSTMCell,self).call(tree_out, states,training=True)

    return output,next_states


layer=RNN(HamLSTMCell(tree_size=3,units=5))

n_steps = 2
n_inputs = 3
n_neurons = 5

#n=2. b=3
X = np.array([
  
  # t = 0      t = 1
  [[[0, 1, 0], [1, 1, 0]]], # instance 0
  [[[1, 0, 5], [0, 0, 0]]], # instance 1
  [[[6, 7, 8], [6, 5, 4]]], # instance 2
  [[[9, 0, 1], [3, 2, 1]]], # instance 3
  ]
)

X=X.reshape(4,1,2,3)
print("shape2",X.shape)

y=layer(X)

#X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
# basic_cell = Wrapper(LSTMCell(n_neurons))
# outputs, states = RNN(basic_cell, X_batch, dtype=tf.float32)
# print(outputs, states)



# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   outputs_val = outputs[0].eval(feed_dict={X: X_batch})
#   print(outputs_val)

# lstmcell=LSTMCell(10)

# inputs=np.arange(5)

# x=tf.keras.Input((None,1,5))

# inputs.reshape((1,1,))

# layer=RNN(lstmcell,input_shape=(1,5))

# y=layer(x)

# print(y.shape)