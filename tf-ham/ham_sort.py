from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from data import create_example
from ham import HAMOperations, HAMTree

#tf.disable_v2_behavior()

lr=0.001
batches_per_epoch=1000
max_epochs=60
batch_size=50
n=4
embed_size=10
tree_size=20
controller_size=20
weights_path='./ham.weights'
test=False

inputs = tf.placeholder(tf.float32, shape=[batch_size, n, embed_size], name='Input')
control = tf.placeholder(tf.float32, shape=[batch_size, controller_size], name='Control')
target = tf.placeholder(tf.float32, shape=[batch_size, n, embed_size], name='Target')

ham_ops = HAMOperations(embed_size, tree_size,controller_size)
tree = HAMTree(ham_ops=ham_ops)
tree.construct(n)

values = [tf.squeeze(x, [1]) for x in tf.split(inputs, n,1)]
for i, val in enumerate(values):
  tree.leaves[i].embed(val)
tree.refresh()

calculate_predicted = tf.concat(1, [tree.get_output(control) for _ in xrange(n)])
targets = tf.reshape(target, [batch_size, n * embed_size])
penalty = lambda x, y: tf.pow(x - y, 2)
#penalty = lambda x, y: tf.abs(x - y)
calculate_loss = tf.reduce_sum(penalty(calculate_predicted, targets)) / n / batch_size

optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(calculate_loss)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter('/tmp/ham-sort/', session.graph_def, flush_secs=5)

  session.run(init)

  if os.path.exists(weights_path):
    saver.restore(session, weights_path)

  # Paper uses 100 epochs with 1000 batches of batch size 50
  for epoch in xrange(max_epochs):
    total_batches = batches_per_epoch
    total_accuracy = 0.0
    total_loss = 0.0
    for i in xrange(total_batches):
      X, Y = [], []
      for b in xrange(batch_size):
        x, y = create_example(n=n, bit_length=embed_size / 2)
        X.append(x)
        Y.append(y)
      control_signal = np.zeros([batch_size, controller_size], dtype=np.float32)
      feed = {inputs: X, target: Y, control: control_signal}
      _, loss, predicted = session.run([tf.no_op() if test else train_step, calculate_loss, calculate_predicted], feed_dict=feed)
      ###
      for b in xrange(batch_size):
        y = Y[b].reshape(n * embed_size)
        y_pred = np.rint(predicted[b]).astype(int)
        total_accuracy += (y == y_pred).all()
      total_loss += loss
    train_acc = total_accuracy / (batch_size * total_batches)
    train_loss = total_loss / total_batches
    print('Epoch = {}'.format(epoch))
    print('Loss = {}'.format(train_loss))
    print('Accuracy = {}'.format(train_acc))
    print('=-=')
    ###
    summary = tf.Summary()
    summary.ParseFromString(session.run(summary_op))
    summary.value.add(tag='TrainAcc', simple_value=float(train_acc))
    summary.value.add(tag='TrainLoss', simple_value=float(train_loss))
    summary_writer.add_summary(summary, epoch)
    ###
    if not test:
      saver.save(session, weights_path)
    