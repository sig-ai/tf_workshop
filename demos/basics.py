
# coding: utf-8

# In[5]:


from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


sess = tf.InteractiveSession() # We need a session for actual values to flow through computation graph


# In[7]:


## TODO: We want to multiply 2x3 and 3x1 matrixes x and y, and add a 3x1 vector z
x = np.random.random((2,3))
y = np.random.random((3,1))
z = np.random.random((2,1))

print('x:', x, sep='\n')
print('y:', y, sep='\n')
print('z:', z, sep='\n')


# In[30]:


## numpy Version

final_np = np.mean(np.matmul(x,y) + z)
final_np


# In[31]:


# tensorflow version
final_tf = tf.reduce_mean(tf.matmul(x,y) + z)
final_tf


# In[15]:


# To get an actual value, we need to run in a session
print('sess run: ', sess.run(final_tf))
# We can also call a tensor's eval method and pass in a session
print('eval sess:', final_tf.eval(session=sess))
# When using eval with an interactive session, the session argument is optional
print('eval sess:', final_tf.eval())


# In[16]:


## Batched operations

# Here we will perform batched matrix multiplication
batch_size = 5
N, M, K = 2,3,2
# a_batch is a batch of 5 NxM matrices
# b_batch is a batch of 5 MxK matrices
a_batch = np.random.random((batch_size, N, M))
b_batch = np.random.random((batch_size, M, K))

out = tf.matmul(a_batch,b_batch)
# out is the batch of 5 NxK matrices that result from the batch of products
sess.run(out)

## Is it kind of weird that we need to sess.run to get a constant value?


# In[18]:


# In tensorflow, we typically have runtime values in our computation graph
# in some sense, each node in the graph is a function of `Placeholders` that flow into it

# So a function like this is numpy
def matmul_and_add(x,y,z):
    return np.matmul(x,y)+z

# Could be expressed as the following computational graph
N,M,K= 2,3,2
# We'll be multiplying an NxM and MxK matrices and adding a Kx1 vector
x_ph = tf.placeholder(tf.float32, shape=[N,M])
y_ph = tf.placeholder(tf.float32, shape=[M,K])
z_ph = tf.placeholder(tf.float32, shape=[K])
xy = tf.matmul(x_ph,y_ph)
out = xy+z_ph


# In[19]:


x = np.random.randint(0,5,[N,M]).astype(np.float32)
y = np.random.randint(0,5,[M,K]).astype(np.float32)
z = np.random.randint(0,5,[K])

print('x:', x, sep='\n')
print('y:', y, sep='\n')
print('z:', z, sep='\n')

tf_out = sess.run(out, feed_dict={x_ph:x, y_ph:y, z_ph:z})
np_out = matmul_and_add(x,y,z)
print("\n\nOUTPUTS \n")
print('np: \n',np_out)
print('tf: \n',tf_out)


# In[22]:


# Make sure you feed all required tensors before calling sess.run
tf_out = sess.run(out, feed_dict={x_ph:x, y_ph:y, z_ph:z})


# In[28]:


#Notices that basic indexing works
ones = tf.ones(10)
print(sess.run(ones))
print(sess.run(ones[:3]))

# But assignment to tf tensors doesn't work
#ones[:3]=5
# Could be a problem, since we want to change network weights as we learn.


# In[29]:


# Actually not a problem.
# We use variables for Tensors which have values that we wish to change
ones = tf.Variable(tf.ones([10]))
update = ones[:4].assign(np.full(4,5))

# The variables initializer is an operation to set the initial values
# We need to run it before we can use the values of a Variable

sess.run(tf.global_variables_initializer())
ones_val = sess.run(ones)
sess.run(update)
update_val = sess.run(ones)

print('ones val', ones_val)
print('updated val', update_val)

