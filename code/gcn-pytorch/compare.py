import tensorflow as tf
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

# input data
num_sample = 10
input_dim = 1000
output_dim = 7
support = np.random.randint(2, size=(num_sample, num_sample)).astype(np.float32)
feature = np.random.randint(2, size=(num_sample, input_dim)).astype(np.float32)

"""tensorflow version"""
# initial weights
params = dict()
params['weights_'+str(100)] = tf.ones([input_dim, output_dim], name='weights_'+str(100))
for i in range(2):
    params['weights_'+str(i)] = tf.ones([input_dim, output_dim], name='weights_'+str(i))
params['bias'] = tf.zeros([output_dim], name='bias')

# call
order_list = list()
pre_sup = tf.matmul(feature, params['weights_'+str(100)])
order_list.append(pre_sup)

for i in range(2):
    pre_sup = tf.matmul(feature, params['weights_'+str(i)])
    for j in range(i+1):
        pre_sup = tf.matmul(support, pre_sup)
    order_list.append(pre_sup)

output = tf.add_n(order_list)
output = tf.add(output, params['bias'])

sess = tf.Session()
print(sess.run(output))

"""
pytorch version
"""
bias = nn.Parameter(torch.Tensor(output_dim))
nn.init.constant(bias.data, 0)

weights = nn.Parameter(torch.Tensor(input_dim, output_dim, 2+1))
nn.init.constant(weights.data, 1)

feature = Variable(torch.from_numpy(feature))
support = Variable(torch.from_numpy(support))

output = feature.mm(weights[:,:,0])
for i in range(2):
    pre_sup = feature.mm(weights[:,:,i+1])
    for j in range(i+1):
        pre_sup = support.mm(pre_sup)
    #output.add_(support.pow(i+1).mm(feature).mm(weights[:,:,i+1]))
    output.add_(pre_sup)
output.add_(bias)
print(output.size())

print(output.data)
