from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def create_pair(bit_length=5):
  key = np.random.randint(2, size=bit_length)
  #val = np.random.randint(2, size=bit_length)
  return list(key)#, list(val)

def create_pairs(bit_length=5, n=5):
  pairs = [create_pair(bit_length=bit_length) for x in range(n)]
  return pairs

def create_example(n=5, bit_length=5):
  pairs = create_pairs(n=n, bit_length=int(bit_length))
  vals=[]
  for p in pairs:
    val = int("".join(str(x) for x in list(p)), 2)
    vals.append(val)

  correct_order = np.argsort(np.asarray(vals))
  x, y = pairs, correct_order
  x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.int32)
  # b=np.zeros((y.size, int(y.max())+1))
  # b[np.arange(y.size),y]=1
  # y=np.array(b,dtype=np.float32)
  return x, y

if __name__ == '__main__':
  print('First two bits are the key which is used for sorting')
  print('Last two bits are the value that is tied to the key')
  print('Sorting should be stable and not depend on the value')
  for i in range(5):
    print('=-=')
    x, y = create_example(bit_length=4)
    print(x)
    print('=>', y)
