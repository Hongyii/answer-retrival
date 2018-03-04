import cPickle as pickle 
import random
from Triplet import Triplet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import tensorflow as tf

DATA_FOLDER = '../serverfault/'

train_q = np.load(DATA_FOLDER + 'train_q_repr.npy')
train_na = np.load(DATA_FOLDER + 'train_na_repr.npy')
#valid_q = np.load(DATA_FOLDER + 'valid_q_repr.npy')
#valid_na = np.load(DATA_FOLDER + 'valid_na_repr.npy')
#test_q = np.load(DATA_FOLDER + 'test_q_repr.npy')
#test_na = np.load(DATA_FOLDER +'test_na_repr.npy')


batch_size = 100
sample_size = 20

# training
tf_q = tf.placeholder(shape=(batch_size, len(train_q[0])), dtype=tf.float32)
tf_a = tf.placeholder(shape=(len(train_na), len(train_na[0])), dtype=tf.float32)
#valid_tf_q = tf.placeholder(shape=(len(valid_q), len(valid_q[0])), dtype=tf.float32)
#valid_tf_a = tf.placeholder(shape=(len(valid_na), len(valid_na[0])), dtype=tf.float32)
#test_tf_q = tf.placeholder(shape=(len(test_q), len(test_q[0])), dtype=tf.float32)
#test_tf_a = tf.placeholder(shape=(len(test_na), len(test_na[0])), dtype=tf.float32)

dot_prod = tf.matmul(tf_q, tf.transpose(tf_a))
#valid_dot_prod = tf.matmul(valid_tf_q, tf.transpose(valid_tf_a))
#test_dot_prod = tf.matmul(test_tf_q, tf.transpose(test_tf_a))
sess = tf.Session()

'''train_q_bat = np.zeros(shape=(batch_size, len(train_q[0])), dtype=np.float32)
cnt = 0
dot_mat = np.zeros(shape=(len(train_q), len(train_na)), dtype=np.float32)
for itr in tqdm(range(len(train_q) / batch_size)):
    train_q_bat[:] = train_q[itr * batch_size:(itr + 1) * batch_size]
    result = sess.run(dot_prod, feed_dict={tf_q: train_q_bat, tf_a: train_na})
    dot_mat[itr*batch_size:(itr+1)*batch_size] = result
np.save(DATA_FOLDER +'train_cosmat', dot_mat)'''

print 'graph complete...'

#val_mat = sess.run(valid_dot_prod, feed_dict={valid_tf_q: valid_q, valid_tf_a: valid_na})
#np.save(DATA_FOLDER + 'valid_cosmat', val_mat)

#test_mat = sess.run(test_dot_prod, feed_dict={test_tf_q: test_q, test_tf_a: test_na})
#np.save(DATA_FOLDER + 'test_cosmat', test_mat)