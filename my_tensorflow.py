import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np






# bunch = load_boston()
# total_features, total_prices = bunch.data , bunch.target
# print (total_prices.shape , ' , ', total_features.shape)
# train = np.hstack((total_prices[:400].reshape(400,1),total_features[:400]))
# test = np.hstack((total_prices[400:].reshape(106,1),total_features[400:]))
# print (train.shape,  '  ,  ' , test.shape)
# train = pd.DataFrame(data=train)
# test = pd.DataFrame(data=test)









beta = 0.01



def final_calc(x, y, w0, w1, w2, w3, w4, b0,b1,b2, b3, b4, const):
    # Returns predictions and error

    # first_layer = tf.nn.relu(tf.add(b0,tf.matmul(x, w0))) + tf.nn.relu(tf.matmul(x,w2))
    L1 = tf.nn.tanh(tf.matmul(x, w0)+b0)
    # L1D = tf.nn.dropout(L1, 1.0)
    L2 = tf.nn.sigmoid(tf.matmul(L1,w1)+b1)
    # L2D = tf.nn.dropout(L2, 0.7)
    L3 = tf.nn.tanh(tf.matmul(L2,w2)+b2)
    # L3D = tf.nn.dropout(L3, 0.85)
    L4 = tf.nn.relu(tf.matmul(L3,w3)+b3)
    # L4D = tf.nn.dropout(L4, 1.0)


    predictions = tf.matmul(L4, w4) + b4
    # print ('x: ' , x)
    # print ('y:' , y)
    # first_layer = tf.pow(first_layer,3)
    # second_layer = first_layer*first_layer*first_layer #tf.nn.relu(tf.matmul(first_layer, w1))
    # predictions = tf.matmul(first_layer, w1)
    # predictions = tf.nn.tanh(tf.add(b, tf.matmul(second_layer,w2)))

    # error = tf.metrics.mean_absolute_error(y , predictions)

    loss = tf.reduce_mean(tf.square(y - predictions))

    regularizer = tf.nn.l2_loss(w0)
    error = tf.reduce_mean(loss + beta * regularizer)


    # error = tf.reduce_mean(tf.squared_difference(predictions, y))



    return [ predictions, error ]


def calc(x, y, w0, w1, w2, w3, w4, b0,b1,b2, b3, b4, const):
    # Returns predictions and error

    # first_layer = tf.nn.relu(tf.add(b0,tf.matmul(x, w0))) + tf.nn.relu(tf.matmul(x,w2))
    L1 = tf.nn.tanh(tf.matmul(x, w0)+b0)
    L1D = tf.nn.dropout(L1, 0.9)
    L2 = tf.nn.sigmoid(tf.matmul(L1D,w1)+b1)
    L2D = tf.nn.dropout(L2, 0.9)
    L3 = tf.nn.tanh(tf.matmul(L2D,w2)+b2)
    L3D = tf.nn.dropout(L3, 0.9)
    L4 = tf.nn.relu(tf.matmul(L3D,w3)+b3)
    L4D = tf.nn.dropout(L4, 1.0)


    predictions = tf.matmul(L4D, w4) + b4
    # print ('x: ' , x)
    # print ('y:' , y)
    # first_layer = tf.pow(first_layer,3)
    # second_layer = first_layer*first_layer*first_layer #tf.nn.relu(tf.matmul(first_layer, w1))
    # predictions = tf.matmul(first_layer, w1)
    # predictions = tf.nn.tanh(tf.add(b, tf.matmul(second_layer,w2)))

    # error = tf.metrics.mean_absolute_error(y , predictions)

    loss = tf.reduce_mean(tf.square(y - predictions))

    regularizer = tf.nn.l2_loss(w0)
    error = tf.reduce_mean(loss + beta * regularizer)


    # error = tf.reduce_mean(tf.squared_difference(predictions, y))



    return [ predictions, error ]

def run_(train_features, train_prices, test_features, test_prices, num_steps=4000, batch_size= 100 ):
    print ('run_')

    print ('max, min: ' , np.max(test_prices), ' , ', np.min(test_prices))
    train_features, train_prices = append_bias_reshape(train_features, train_prices)
    test_features, test_prices = append_bias_reshape(test_features, test_prices)


    print (train_features.shape, ' , ', train_prices.shape, ' , ', test_features.shape, ' , ', test_prices.shape)

    print ('train_prices\n')
    print ( train_prices[:50])
    print ('test_prices\n')
    print ( test_prices[:50])
    print ('max, min: ' , np.max(test_prices), ' , ', np.min(test_prices))

    num_features = train_features.shape[1]
    num_labels = 1


    w0 = tf.Variable(tf.truncated_normal([num_features, 200], mean=0.0, stddev=0.1, dtype=tf.float32))
    # w0 = tf.Variable(tf.random_uniform([num_features, 50], -1, 1))
    w1 = tf.Variable(tf.truncated_normal([200, 50], mean=0.0, stddev=0.1, dtype=tf.float32))
    w2 = tf.Variable(tf.truncated_normal([50, 25], mean=0.0, stddev=0.1, dtype=tf.float32))
    # w1 = tf.Variable(tf.random_uniform([50,10], -1, 1))
    w3 = tf.Variable(tf.truncated_normal([25, 10], mean=0.0, stddev=0.1, dtype=tf.float32))
    w4 = tf.Variable(tf.truncated_normal([10, 1], mean=0.0, stddev=0.1, dtype=tf.float32))
    # w2 = tf.Variable(tf.random_uniform([10,1], -1, 1))
    # w2 = tf.Variable(tf.truncated_normal([10, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
    # b0 = tf.constant(1, shape=[1],dtype = tf.float32)
    b0 = tf.Variable(tf.ones(1, dtype = tf.float32))
    b1 = tf.Variable(tf.ones(1, dtype = tf.float32))
    b2 = tf.Variable(tf.ones(1, dtype = tf.float32))
    b3 = tf.Variable(tf.ones(1, dtype = tf.float32))
    b4 = tf.Variable(tf.ones(1, dtype = tf.float32))

    # const = tf.Variable(tf.ones(1, dtype = tf.float32))
    const = tf.constant(100, dtype=tf.float32)


    # w0=tf.Variable(tf.random_uniform([num_features,50],-0.01,0.01,dtype = tf.float32))
    # b0=tf.Variable(tf.random_uniform([50,1],-0.01,0.01, dtype = tf.float32))
    # w1=tf.Variable(tf.random_uniform([50,1],-0.01,0.01, dtype = tf.float32))


    tf_train_dataset = tf.placeholder(tf.float32, shape = [None, num_features])
    # tf_train_labels = tf.placeholder(tf.float32, shape = [None])
    tf_train_labels = tf.placeholder(tf.float32,[None,1])

    # step =  tf.Variable(tf.zeros((), dtype = tf.int32))


    y, cost = calc(train_features, train_prices, w0, w1, w2, w3, w4, b0, b1, b2, b3, b4, const)


    print ('here0')
    learning_rate = 0.02

    print ('here1')
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    # init = tf.local_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    print ('here2')



    with tf.Session() as sess:

        sess.run(init)

        # for i in list(range(epochs)):

        ste = 1

        cost_history = 0

        for i in range(num_steps):
            if i% int( train_prices.shape[0]/batch_size) == 0:
                train_features, train_prices = shuffle(train_features, train_prices)


            offset = (i * batch_size) % (train_prices.shape[0] - batch_size)
            minibatch_data = train_features[offset:(offset + batch_size), :]
            minibatch_labels = train_prices[offset:(offset + batch_size)]




            optimizer.run()
            # print ('minibatch_data: ' , minibatch_data)
            # print ('minibach_labels: ' , minibatch_labels)
            feed_dict = {tf_train_dataset : minibatch_data, tf_train_labels : minibatch_labels} #, tf_test_dataset: Xtest}
            _, w0_, b0_, b1_, x_ = sess.run([optimizer,w0, b0, b1, tf_train_dataset], feed_dict=feed_dict)

            if i % 10 == 0:
                # optimizer.__setattr__('learning_rate', learning_rate * 0.99)
                # print('i: ' , i , ' , ', sess.run(cost))
                # y_pred , test_cost = calc(test_features, test_prices, w0, w1, w2, b0, b1, b2,const)
                # print ('preds: ' , sess.run(y_pred))
                optimizer.__setattr__('learning_rate', learning_rate)
            if i % 100 == 0:
                y_pred , train_cost = calc(train_features, train_prices, w0, w1, w2, w3, w4,  b0, b1, b2,b3, b4, const)
                predictions = sess.run(y_pred)

                # print ('test_y: ' , test_prices)
                print('preds: ' , predictions[:5])

                tcost = sess.run(train_cost)
                print(tcost)
                print ('costs: ' , tcost)
                if tcost == cost_history and i < 5000:
                    optimizer.__setattr__('learning_rate', learning_rate * 5)
                elif tcost > (cost_history * 1.01):
                    learning_rate *= 0.9
                cost_history = tcost

            if i%500 == 0:
                learning_rate *= 0.8
                optimizer.__setattr__('learning_rate', learning_rate)
                print ('learning_rate: ' , optimizer.__getattribute__('learning_rate'))


        y_pred , test_cost = calc(test_features, test_prices, w0, w1, w2, w3, w4,  b0, b1, b2,b3, b4, const)
        # y_pred , train_cost = calc(train_features, train_prices, w0, w1, w2, b0, b1, b2,const)


        predictions = sess.run(y_pred)
        print ('test_y: ' , test_prices[:5])
        print('preds: ' , predictions[:5])

        print('test cost: ' , sess.run(cost))

        print ('mse: ', mean_squared_error(test_prices, predictions))
        print ('mae: ' ,mean_absolute_error(test_prices, predictions))

        print (tf.metrics.mean_squared_error(test_prices, predictions))


        # fig, ax = plt.subplots()
        # ax.scatter(test_prices, predictions)
        # ax.plot([test_prices.min(), test_prices.max()], [test_prices.min(), test_prices.max()], 'k--', lw=3)
        # ax.set_xlabel('Measured')
        # ax.set_ylabel('Predicted')
        # plt.show()

        return predictions



def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features, labels

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    # f = features
    # f = np.c_[np.ones(n_training_samples),features]
    # f = f.reshape([n_training_samples,n_dim + 1])
    # l = labels
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    print ('f.shape: ' , len(f) , ' , ', f.shape)
    l = np.reshape(labels,[n_training_samples,1])
    return f, l

f,l = read_boston_data()
f = feature_normalize(f)

print (f[1:2])
f, l = append_bias_reshape(f,l)
print (f[1:2])

rnd_indices = np.random.rand(len(f)) < 0.80

def initialize(features, labels):
    train_x = features[rnd_indices]
    train_y = labels[rnd_indices]
    test_x = features[~rnd_indices]
    test_y = labels[~rnd_indices]
    return run_(train_x,train_y, test_x, test_y)




# initialize(f,l)



# def initialize(train , test):
#     # batch_size = 500
#     print ('initialize')
#     train = train.sample(frac=1,random_state=0)
#
#     train_features =  scale(train.iloc[:,1:])
#     test_features = scale(test.iloc[:,1:])
#
#     train_prices = np.array(train.iloc[:,0])
#     test_prices = np.array(test.iloc[:,0])
#
#
#
#     return run_(train_features,train_prices, test_features, test_prices)
#
# initialize(train, test)
