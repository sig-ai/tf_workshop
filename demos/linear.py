
# coding: utf-8

# # Linear Regression Example
# 
# A linear regression learning algorithm example using TensorFlow library.
# 
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random


# In[2]:


# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50


# In[3]:


# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])[:,None]
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])[:,None]
n_samples = train_X.shape[0]


# In[4]:


# tf Graph Input
X = tf.placeholder("float", shape=[1])
Y = tf.placeholder("float", shape=[1])

# Set model weights


# In[5]:


# define a linear function
def linear(x, output_features, return_params=False):
    x_flat = tf.reshape(x,[-1])
    input_features=int(x_flat.get_shape()[0])
    W = tf.Variable(tf.random_uniform(shape=[input_features, output_features], name="weight"))
    b = tf.Variable(tf.random_uniform(shape=[output_features], name="bias"))
    pred = tf.add(tf.multiply(X, W), b)
    return pred, W, b
#pred = linear(X, 1)
pred, W, b = linear(X,1, return_params=True)


# In[6]:


# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[7]:


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[8]:


# Start training
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c=0
            for (x, y) in zip(train_X, train_Y):
                c+= sess.run(cost, feed_dict={X: x, Y:y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                 "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[9]:


# Regression result

