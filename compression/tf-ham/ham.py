from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import LSTMCell
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#tf.enable_eager_execution()

def body(i,pointer,prev_outs):

  el=pointer[tf.size(pointer)-i-1]
  if(el not in prev_outs):
    return el 
  return tf.add(i,1)


def findmaxnotinlist(pointer, prev_outs):

  i=tf.constant(0)

  while_condition = lambda i:tf.less(i, tf.size(pointer))

  r=tf.while_loop(while_condition, body, [i,])  
    

class Transformer(object):
  '''
  Input = [b]
  Output = [d]
  '''
  def __init__(self, input_size, output_size):
    self.W = tf.Variable(tf.random.truncated_normal([input_size, output_size]))

  def __call__(self, val):
    return tf.nn.relu(tf.matmul(val, self.W))


class Join(object):
  '''
  Input = [d, d]
  Output = [d]
  '''
  def __init__(self, d):
    self.W = tf.Variable(tf.random.truncated_normal([2 * d, d]))

  def __call__(self, left, right):
    return tf.nn.relu(tf.matmul(tf.concat([left, right],1), self.W))


class Search(object):
  '''
  Input = [d, l]
  Output = [1]
  '''
  def __init__(self, d, l):
    self.W = tf.Variable(tf.random.truncated_normal([d + l, 1]))

  def __call__(self, h, control):
    return tf.nn.sigmoid(tf.matmul(tf.concat([h, control],1), self.W))


class Write(object):
  '''
  Input = [d, l]
  Output = [d]
  '''
  def __init__(self, d, l):
    self.H = tf.Variable(tf.random.truncated_normal([d + l, d]))
    self.T = tf.Variable(tf.random.truncated_normal([d + l, 1]))

  def __call__(self, h, control):
    data = tf.concat([h, control],1)
    candidate = tf.nn.sigmoid(tf.matmul(data, self.H))
    update = tf.nn.sigmoid(tf.matmul(data,  self.T))
    return update * candidate + (1 - update) * h


class HAMOperations(object):
  def __init__(self, embed_size, tree_size, controller_size):
    ###
    # From the paper,
    self.embed_size = embed_size  # b
    self.tree_size = tree_size  # d
    self.controller_size = controller_size  # l
    ##
    self.transform = Transformer(embed_size, tree_size)
    self.join = Join(tree_size)
    self.search = Search(tree_size, controller_size)
    self.write = Write(tree_size, controller_size)


class HAMTree(object):
  def __init__(self, ham_ops):
    self.ops = ham_ops
    ##
    self.transform = self.ops.transform
    self.join = self.ops.join
    self.search = self.ops.search
    self.write = self.ops.write
    self.root = None
    self.leaves = None
    self.nodes = None

  def __repr__(self):
    return 'HAMTree'

  def construct(self, total_leaves):
    # Ensure that the total number of leaves is a power of two
    depth = np.log(total_leaves) / np.log(2)
    assert depth.is_integer(), 'The total leaves must be a power of two'

    queue = [HAMNode(tree=self, left=None, right=None) for leaf in range(total_leaves)]
    self.leaves = [leaf for leaf in queue]
    self.nodes = [leaf for leaf in queue]
    while len(queue) > 1:
      l, r = queue.pop(0), queue.pop(0)
      node = HAMNode(tree=self, left=l, right=r)
      queue.append(node)
      self.nodes.append(node)
    self.root = queue[0]

  def refresh(self):
    self.root.join()

  def get_output(self, control):
    #return self.root.retrieve_ha_and_update(control)
    return self.root.retrieve_and_update(control,attention=1.0)


class HAMNode(object):
  def __init__(self, tree, left, right):
    self.tree = tree
    self.left = left
    self.right = right
    self.h = None
    self.value = None

  def __repr__(self):
    return 'HAMNode(tree={}, left={}, right={})'.format(self.tree, bool(self.left), bool(self.right))

  def embed(self, value):
    self.value = value
    self.h = self.tree.transform(value)

  def join(self):
    if self.left and self.right:
      self.left.join()
      self.right.join()
      self.h = self.tree.join(self.left.h, self.right.h)

  def retrieve_and_update(self, control, attention=1.0):
    value = None
    ###
    # Retrieve the value - left and right weighted by the value of search
    if self.left and self.right:
      move_right = self.tree.search(self.h, control)
      value = self.right.retrieve_and_update(control, attention=attention * move_right)
      value += self.left.retrieve_and_update(control, attention=attention * (1 - move_right))
    else:
      value = attention * tf.cast(self.value,tf.float32)
    ###
    # Update the values of the tree
    if self.left and self.right:
      self.h = self.tree.join(self.left.h, self.right.h)
    else:
      self.h = attention * self.tree.write(self.h, control) + (1 - attention) * self.h
    return value




class HamLSTMCell(tf.keras.layers.Layer):

  def __init__(self,tree_size,controller_size,units,*args, **kwargs):

    self.tree_size=tree_size
    super(HamLSTMCell, self).__init__()
    self.controller_size=controller_size
    self.state_size=(units,units)
    self.units=units
    self.cell=LSTMCell(units,return_states=True)
    
    #self.hlstm=tf.Variable(tf.random.normal([controller_size]))
    # self.model=Sequential()
    # self.model.add(Dense(tree_size,activation="tanh"))
    self.prev_outs=[]


    

  def build(self, input_shape):
    print(input_shape)
    self.n=input_shape[1] 
    #self.model.add(Dense(self.n,activation="softmax"))
    self.embed_size=input_shape[2]
    self.ops = HAMOperations(self.embed_size, self.tree_size,self.units)
    self.tree = HAMTree(ham_ops=self.ops)
    self.tree.construct(self.n)


  def call(self, inputs, states):  
    print(tf.executing_eagerly())

    #prev_outs,states1 = states
    #h,c=states1
    h,c=states
    values = [tf.squeeze(x, [1]) for x in tf.split(inputs, self.n,1)]
    for i, val in enumerate(values):
      self.tree.leaves[i].embed(val)

    self.tree.refresh()  

    tree_out=self.tree.get_output(h)

    opcell, next_state=self.cell.call(tree_out, states,training=True)
    #next_state=[prev_outs,next_state1]
    #output=self.model(opcell)
    
    indices=tf.argsort(opcell)

    #el=findmaxnotinlist(indices,self.prev_outs)
    #print(el)
    #self.prev_outs.append(el)
    return output,next_state


# class HamLSTMCell(tf.keras.layers.Layer):

#   def __init__(self,units,tree_size,controller_size,*args, **kwargs):

#     self.tree_size=tree_size
#     super(HamLSTMCell, self).__init__(*args, **kwargs)
#     self.controller_size=controller_size
#     self.cell=LSTMCell(units)
#     #self.hlstm=tf.Variable(tf.random.normal([controller_size]))
#     self.model=Sequential()
#     self.model.add(Dense(tree_size,activation="tanh"))
#     self.prev_outs=[]


    

#   def build(self, input_shape):
#     super(HamLSTMCell,self).build(input_shape)
#     self.cell.build(input_shape)
#     self.n=input_shape[1] 
#     self.model.add(Dense(self.n,activation="softmax"))
#     self.embed_size=input_shape[2]
#     self.ops = HAMOperations(self.embed_size, self.tree_size,self.units)
#     self.tree = HAMTree(ham_ops=self.ops)
#     self.tree.construct(self.n)



#   def call(self, inputs, states):  

#     #prev_outs,states1 = states
#     #h,c=states1
#     h,c=states
#     values = [tf.squeeze(x, [1]) for x in tf.split(inputs, self.n,1)]
#     for i, val in enumerate(values):
#       self.tree.leaves[i].embed(val)

#     self.tree.refresh()  

#     tree_out=self.tree.get_output(h)

#     opcell, next_state=self.cell.call(tree_out, states,training=True)
#     #next_state=[prev_outs,next_state1]
#     output=self.model(opcell)
#     print(tf.executing_eagerly())
#     opnum=output.numpy()

    
#     indices=tf.argsort(output)

#     el=findmaxnotinlist(indices,self.prev_outs)
#     print(el)
#     self.prev_outs.append(el)
#     return output,next_state




if __name__ == '__main__':
  ham_ops = HAMOperations(1, 2, 3)
  tree = HAMTree(ham_ops=ham_ops)
  tree.construct(2)
  print(tree.nodes)
  l, r = tree.leaves
  assert len(tree.leaves) == 2, 'Depth 2 tree should have 2 leaves'
  assert len(tree.nodes) == 3, 'Depth 2 tree should have 3 nodes (two leaves, one root)'
  assert tree.root.left == l and tree.root.right == r, 'Depth 2 tree is broken'