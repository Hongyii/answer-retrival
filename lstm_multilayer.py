import tensorflow as tf
import random
import numpy as np
import time
from tqdm import tqdm 
import os,sys

max_input_len = 200
embed_dim = 200 
#reg = 10e-5
#lstm_size = 100
#batch_size = 1000
margin = 0.2
iter_log = 50

'''
_input shape[train_size, max_input_len]
_len shape[train_size]
'''

def lstm_multilayer(DATA_FOLDER, TB_FOLDER, reg = 10e-4, lstm_size=100, batch_size = 1000, iter_log = 100):
    
    # load Task1 data
    
    embed_mat = np.load(DATA_FOLDER + 'embed_mat.npy')# shape (vocab_size, embedding_dim)
    
    # training set
    train_q_input = np.load(DATA_FOLDER + 'train_q_input.npy')
    train_q_len = np.load(DATA_FOLDER + 'train_q_len.npy')
    train_pos_ans_input = np.load(DATA_FOLDER + 'train_a_input.npy')
    train_pos_ans_len = np.load(DATA_FOLDER + 'train_a_len.npy')
    train_neg_ans_input = np.load(DATA_FOLDER + 'train_n_a_input.npy')
    train_neg_ans_len = np.load(DATA_FOLDER + 'train_n_a_len.npy')

    # validation set
    validation_q_input = np.load(DATA_FOLDER + 'valid_q_input.npy')
    validation_q_len = np.load(DATA_FOLDER + 'valid_q_len.npy')
    validation_pos_ans_input = np.load(DATA_FOLDER + 'valid_a_input.npy')
    validation_pos_ans_len = np.load(DATA_FOLDER + 'valid_a_len.npy')
    validation_neg_ans_input = np.load(DATA_FOLDER + 'valid_n_a_input.npy')
    validation_neg_ans_len = np.load(DATA_FOLDER + 'valid_n_a_len.npy')

    with tf.variable_scope('inputs_py'):
        q_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        pos_a_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        neg_a_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        q_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        pos_a_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        neg_a_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        accuracy = tf.placeholder(tf.float32, shape=None)
        log_loss = tf.placeholder(tf.float32, shape=None)

    with tf.variable_scope('embedding'):
        word_embed = tf.constant(embed_mat, dtype=tf.float32, shape=embed_mat.shape, name='word_embed')
        q_embed_lookup = tf.nn.embedding_lookup(word_embed, tf.reshape(q_input, shape=[batch_size * max_input_len]), name='question_embed_lookup')
        pos_a_embed_lookup = tf.nn.embedding_lookup(word_embed, tf.reshape(pos_a_input, shape=[batch_size * max_input_len]), name='pos_ans_embed_lookup')
        neg_a_embed_lookup = tf.nn.embedding_lookup(word_embed, tf.reshape(neg_a_input, shape=[batch_size * max_input_len]), name='neg_ans_embed_lookup')

    initializer=tf.contrib.layers.xavier_initializer()    
        
    with tf.variable_scope('lstm_quetion', initializer=initializer):
        q_seq_inputs= tf.reshape(q_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='question_seq_inputs')
        q_lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True)
        q_lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True)
        stacked_q_lstm = tf.nn.rnn_cell.MultiRNNCell([q_lstm_cell_1, q_lstm_cell_2], state_is_tuple=True)
        q_output, q_final_state = tf.nn.dynamic_rnn( cell=stacked_q_lstm,
                                            inputs=q_seq_inputs,
                                            sequence_length=q_tf_size, # a vector of [batch_size] each tensor instance (question)length
                                            dtype=tf.float32)
    with tf.variable_scope('lstm_answer', initializer=initializer):
        pos_a_seq_inputs = tf.reshape(pos_a_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='pos_answer_seq_inputs')
        a_lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True)
        a_lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True)
        stacked_a_lstm = tf.nn.rnn_cell.MultiRNNCell([a_lstm_cell_1, a_lstm_cell_2], state_is_tuple=True)
        pos_a_output, pos_a_final_state = tf.nn.dynamic_rnn( cell=stacked_a_lstm,
                                            inputs=pos_a_seq_inputs,
                                            sequence_length=pos_a_tf_size, 
                                            dtype=tf.float32)


    with tf.variable_scope('lstm_answer', reuse=True):
        neg_a_seq_inputs = tf.reshape(neg_a_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='neg_answer_seq_inputs')
        neg_a_output, neg_a_final_state = tf.nn.dynamic_rnn(cell=stacked_a_lstm,
                                            inputs= neg_a_seq_inputs,
                                            sequence_length=neg_a_tf_size, 
                                            dtype=tf.float32)

    with tf.variable_scope('l2norm'):
        q_tf_size_r = tf.to_float(tf.reshape(q_tf_size, [-1, 1]))
        pos_a_tf_size_r = tf.to_float(tf.reshape(pos_a_tf_size, [-1, 1]))
        neg_a_tf_size_r = tf.to_float(tf.reshape(neg_a_tf_size, [-1, 1]))
        q_mean_state = tf.mul(tf.reduce_sum(q_output, reduction_indices=1), tf.inv(q_tf_size_r))
        pos_mean_state = tf.mul(tf.reduce_sum(pos_a_output, reduction_indices=1), tf.inv(pos_a_tf_size_r))
        neg_mean_state = tf.mul(tf.reduce_sum(neg_a_output, reduction_indices=1), tf.inv(neg_a_tf_size_r))
        norm_question = tf.nn.l2_normalize(q_mean_state, dim=1, name='norm_question')
        norm_pos_answer = tf.nn.l2_normalize(pos_mean_state,dim=1, name='norm_pos_answer')
        norm_neg_answer = tf.nn.l2_normalize(neg_mean_state,dim=1, name='norm_neg_answer')

    with tf.variable_scope('validation'):
        dif = tf.reduce_sum(tf.mul(norm_question, norm_pos_answer) - tf.mul(norm_question, norm_neg_answer), reduction_indices=1)
        valid_result = tf.shape(tf.where(tf.less(dif,0)))
    
    
    for variable in tf.get_collection(tf.GraphKeys.VARIABLES, scope='lstm_question'):
        tf.add_to_collection('losses', reg * tf.nn.l2_loss(variable))
        
    for variable in tf.get_collection(tf.GraphKeys.VARIABLES, scope='lstm_answer'):
        tf.add_to_collection('losses', reg * tf.nn.l2_loss(variable))
    
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.maximum(margin - tf.reduce_sum(tf.mul(norm_question, norm_pos_answer), reduction_indices=1) + \
        tf.reduce_sum(tf.mul(norm_question, norm_neg_answer), reduction_indices=1), 0))
        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_grads_and_vars)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    train_writer = tf.train.SummaryWriter(TB_FOLDER ,sess.graph) 
    acc_sum = tf.scalar_summary('accuracy', accuracy)
    loss_sum = tf.scalar_summary('log_loss', log_loss)
    lstm_var = tf.all_variables()
    hist_sum = []
    for var in lstm_var:
        hist_sum.append(tf.histogram_summary(var.name, var))
    merged = tf.merge_all_summaries()
    
    
    train_index = np.arange(0, len(train_q_input))
    acc_loss = 0

    for epoch in range(2):
        np.random.shuffle(train_index)
        for itr in range(len(train_q_input) / batch_size):

            loss_value, _ = sess.run([loss, train_op], feed_dict={
                    q_input : train_q_input[train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    pos_a_input : train_pos_ans_input[train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    neg_a_input : train_neg_ans_input[train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    q_tf_size : train_q_len[train_index[itr * batch_size:(itr + 1) * batch_size]],
                    pos_a_tf_size : train_pos_ans_len[train_index[itr * batch_size:(itr + 1) * batch_size]],
                    neg_a_tf_size : train_neg_ans_len[train_index[itr * batch_size:(itr + 1) * batch_size]]
                    })
                    #options=run_options,
                    #run_metadata=run_metadata)
            #train_writer.add_run_metadata(run_metadata, 'step%d' % itr)

            acc_loss += loss_value
            #print '==> itr: %d, loss: %f' % (itr, loss_value)

            if itr != 0 and itr % iter_log == 0:
                log_acc_loss = acc_loss / iter_log
                print '==> itr: %d, loss: %f' % (itr, log_acc_loss)
                acc_loss = 0
                print '==> itr: %d, validation' % (itr)
                count = 0
                for val_itr in tqdm(range(len(validation_q_input) / batch_size)):
                    
                    dif_arr, count_shape = sess.run([dif, valid_result], feed_dict={
                                q_input : validation_q_input[val_itr * batch_size:(val_itr + 1) * batch_size, :],
                                pos_a_input : validation_neg_ans_input[val_itr * batch_size:(val_itr + 1) * batch_size, :], 
                                neg_a_input : validation_pos_ans_input[val_itr * batch_size:(val_itr + 1) * batch_size, :],
                                q_tf_size : validation_q_len[val_itr * batch_size:(val_itr + 1) * batch_size], 
                                pos_a_tf_size : validation_neg_ans_len[val_itr * batch_size:(val_itr + 1) * batch_size],
                                neg_a_tf_size : validation_pos_ans_len[val_itr * batch_size:(val_itr + 1) * batch_size]
                             })
                    count += count_shape[0]
                    #print '==> validation: %d' % val_itr
                print '==> validation accuracy: %f' % (count * 1.0 / len(validation_q_input))
                summary = sess.run(merged, feed_dict={accuracy: count * 1.0 / (len(validation_q_input) / batch_size), log_loss:log_acc_loss})
                train_writer.add_summary(summary, itr)
                acc_loss = 0
            if itr != 0 and itr % 2*iter_log == 0:
                save_path = saver.save(sess, TB_FOLDER + 'model_' + str(itr))
        acc_loss = 0

            
            
if __name__ == '__main__':
    if len(sys.argv) == 6:
        DATA_FOLDER = sys.argv[1]
        TB_FOLDER = sys.argv[2]
        reg = float(sys.argv[3])
        lstm_size= int(sys.argv[4])
        batch_size = int(sys.argv[5])
        iter_log = 100
        if (not os.path.isdir(DATA_FOLDER)):
            print 'not valid data folder...'
        elif (not os.path.isdir(TB_FOLDER)):
            print 'not valid tensorboard folder...'
        else: 
            lstm_multilayer(DATA_FOLDER, TB_FOLDER, reg=reg, lstm_size=lstm_size, batch_size = batch_size, iter_log = iter_log)
    else:
        print 'input data folder...'