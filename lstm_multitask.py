import tensorflow as tf
import random
import numpy as np
import time
from tqdm import tqdm 
import os,sys

max_input_len = 200
embed_dim = 200 
margin = 0.2
#reg = 10e-4
#lstm_size = 100
#batch_size = 1000
#iter_log = 50
'''
_input shape[train_size, max_input_len]
_len shape[train_size]
'''

def lstm_multitask(TASK1_DATA_FOLDER, TASK2_DATA_FOLDER, TB_FOLDER, reg = 10e-4, lstm_size=100, batch_size = 1000, iter_log = 100):
    print 'setting:reg=%f, lstm_size=%d ' % (reg, lstm_size)
    # load Task1 data
    t1_embed_mat = np.load('../multitask/embed_mat.npy')# shape (vocab_size, embedding_dim)
    # training set
    t1_train_q_input = np.load(TASK1_DATA_FOLDER + 'train_q_input.npy')
    t1_train_q_len = np.load(TASK1_DATA_FOLDER + 'train_q_len.npy')
    t1_train_pos_ans_input = np.load(TASK1_DATA_FOLDER + 'train_a_input.npy')
    t1_train_pos_ans_len = np.load(TASK1_DATA_FOLDER + 'train_a_len.npy')
    t1_train_neg_ans_input = np.load(TASK1_DATA_FOLDER + 'train_n_a_input.npy')
    t1_train_neg_ans_len = np.load(TASK1_DATA_FOLDER + 'train_n_a_len.npy')

    # validation set
    t1_validation_q_input = np.load(TASK1_DATA_FOLDER + 'valid_q_input.npy')
    t1_validation_q_len = np.load(TASK1_DATA_FOLDER + 'valid_q_len.npy')
    t1_validation_pos_ans_input = np.load(TASK1_DATA_FOLDER + 'valid_a_input.npy')
    t1_validation_pos_ans_len = np.load(TASK1_DATA_FOLDER + 'valid_a_len.npy')
    t1_validation_neg_ans_input = np.load(TASK1_DATA_FOLDER + 'valid_n_a_input.npy')
    t1_validation_neg_ans_len = np.load(TASK1_DATA_FOLDER + 'valid_n_a_len.npy')

    
    # load Task2 data
    t2_embed_mat = np.load('../multitask/embed_mat.npy')# shape (vocab_size, embedding_dim)
    # training set
    t2_train_q_input = np.load(TASK2_DATA_FOLDER + 'train_q_input.npy')
    t2_train_q_len = np.load(TASK2_DATA_FOLDER + 'train_q_len.npy')
    t2_train_pos_ans_input = np.load(TASK2_DATA_FOLDER + 'train_a_input.npy')
    t2_train_pos_ans_len = np.load(TASK2_DATA_FOLDER + 'train_a_len.npy')
    t2_train_neg_ans_input = np.load(TASK2_DATA_FOLDER + 'train_n_a_input.npy')
    t2_train_neg_ans_len = np.load(TASK2_DATA_FOLDER + 'train_n_a_len.npy')

    # validation set
    t2_validation_q_input = np.load(TASK2_DATA_FOLDER + 'valid_q_input.npy')
    t2_validation_q_len = np.load(TASK2_DATA_FOLDER + 'valid_q_len.npy')
    t2_validation_pos_ans_input = np.load(TASK2_DATA_FOLDER + 'valid_a_input.npy')
    t2_validation_pos_ans_len = np.load(TASK2_DATA_FOLDER + 'valid_a_len.npy')
    t2_validation_neg_ans_input = np.load(TASK2_DATA_FOLDER + 'valid_n_a_input.npy')
    t2_validation_neg_ans_len = np.load(TASK2_DATA_FOLDER + 'valid_n_a_len.npy')

    with tf.variable_scope('inputs_py'):
    
        t1_q_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        t1_pos_a_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        t1_neg_a_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        t1_q_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        t1_pos_a_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        t1_neg_a_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        
        
        t2_q_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        t2_pos_a_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        t2_neg_a_input = tf.placeholder(tf.int32, shape=[batch_size, max_input_len])
        t2_q_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        t2_pos_a_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        t2_neg_a_tf_size = tf.placeholder(tf.int32, shape=[batch_size])
        
        
        accuracy = tf.placeholder(tf.float32, shape=None)
        t1_accuracy = tf.placeholder(tf.float32, shape=None)
        t2_accuracy = tf.placeholder(tf.float32, shape=None)
        log_loss = tf.placeholder(tf.float32, shape=None)
        
    with tf.variable_scope('embedding'):
         
        
        t1_word_embed = tf.constant(t1_embed_mat, dtype=tf.float32, shape=t1_embed_mat.shape, name='t1_word_embed')
        t2_word_embed = tf.constant(t2_embed_mat, dtype=tf.float32, shape=t2_embed_mat.shape, name='t2_word_embed')
        
        t1_q_embed_lookup = tf.nn.embedding_lookup(t1_word_embed, tf.reshape(t1_q_input, shape=[batch_size * max_input_len]), name='t1_question_embed_lookup')
        t1_pos_a_embed_lookup = tf.nn.embedding_lookup(t1_word_embed, tf.reshape(t1_pos_a_input, shape=[batch_size * max_input_len]), name='t1_pos_ans_embed_lookup')
        t1_neg_a_embed_lookup = tf.nn.embedding_lookup(t1_word_embed, tf.reshape(t1_neg_a_input, shape=[batch_size * max_input_len]), name='t1_neg_ans_embed_lookup')
        
        t2_q_embed_lookup = tf.nn.embedding_lookup(t2_word_embed, tf.reshape(t2_q_input, shape=[batch_size * max_input_len]), name='t2_question_embed_lookup')
        t2_pos_a_embed_lookup = tf.nn.embedding_lookup(t2_word_embed, tf.reshape(t2_pos_a_input, shape=[batch_size * max_input_len]), name='t2_pos_ans_embed_lookup')
        t2_neg_a_embed_lookup = tf.nn.embedding_lookup(t2_word_embed, tf.reshape(t2_neg_a_input, shape=[batch_size * max_input_len]), name='t2_neg_ans_embed_lookup')
       
    initializer=tf.contrib.layers.xavier_initializer()
        
    with tf.variable_scope('lstm_quetion', initializer=initializer):
        t1_q_seq_inputs= tf.reshape(t1_q_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='t1_question_seq_inputs')
        shared_q_output, shared_q_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True),
                                            inputs=t1_q_seq_inputs,
                                            sequence_length=t1_q_tf_size, 
                                            dtype=tf.float32, scope='lstm_question_shared', swap_memory=True)
        
        t1_q_output, t1_q_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True),
                                            inputs=shared_q_output,
                                            sequence_length=t1_q_tf_size, # a vector of [batch_size] each tensor instance (question)length
                                            dtype=tf.float32, scope='lstm_question_t1', swap_memory=True)
        
       
        
    with tf.variable_scope('lstm_quetion', reuse=True, initializer=initializer):
          
        t2_q_seq_inputs= tf.reshape(t2_q_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='t2_question_seq_inputs')
        
        shared_q_output, shared_q_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True),
                                            inputs=t2_q_seq_inputs,
                                            sequence_length=t2_q_tf_size, 
                                            dtype=tf.float32, scope='lstm_question_shared', swap_memory=True)
    with tf.variable_scope('lstm_quetion'):
        t2_q_output, t2_q_final_state = tf.nn.dynamic_rnn( 
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size, state_is_tuple=True),
                                            inputs=shared_q_output,
                                            sequence_length=t2_q_tf_size, # a vector of [batch_size] each tensor instance (question)length
                                            dtype=tf.float32, scope='lstm_question_t2', swap_memory=True
                                            )
        
        
    
    with tf.variable_scope('lstm_answer',initializer=initializer):
        
        t1_pos_a_seq_inputs = tf.reshape(t1_pos_a_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='t1_pos_answer_seq_inputs')
        shared_pos_a_output, shared_pos_a_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs=t1_pos_a_seq_inputs,
                                            sequence_length=t1_pos_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_shared', swap_memory=True)
        
        t1_pos_a_output, t1_pos_a_final_state = tf.nn.dynamic_rnn( cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs=shared_pos_a_output,
                                            sequence_length=t1_pos_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_t1', swap_memory=True)


    with tf.variable_scope('lstm_answer', reuse=True, initializer=initializer):
        t1_neg_a_seq_inputs = tf.reshape(t1_neg_a_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='t1_neg_answer_seq_inputs')
        shared_neg_a_output, shared_neg_a_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs=t1_neg_a_seq_inputs,
                                            sequence_length=t1_neg_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_shared', swap_memory=True)

        t1_neg_a_output, t1_neg_a_final_state = tf.nn.dynamic_rnn(cell= tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs= shared_neg_a_output,
                                            sequence_length= t1_neg_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_t1', swap_memory=True)
        
        
    with tf.variable_scope('lstm_answer', reuse=True, initializer=initializer):
        t2_pos_a_seq_inputs = tf.reshape(t2_pos_a_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='t2_pos_answer_seq_inputs')
        
        shared_pos_a_output, shared_pos_a_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs=t2_pos_a_seq_inputs,
                                            sequence_length=t2_pos_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_shared', swap_memory=True)
    with tf.variable_scope('lstm_answer'):
        t2_pos_a_output, t2_pos_a_final_state = tf.nn.dynamic_rnn( cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs=shared_pos_a_output,
                                            sequence_length=t2_pos_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_t2', swap_memory=True)


    with tf.variable_scope('lstm_answer', reuse=True, initializer=initializer):
        t2_neg_a_seq_inputs = tf.reshape(t2_neg_a_embed_lookup, shape=[batch_size, max_input_len, embed_dim], name='t2_neg_answer_seq_inputs')
        
        shared_neg_a_output, shared_neg_a_final_state = tf.nn.dynamic_rnn(
                                            cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs=t2_neg_a_seq_inputs,
                                            sequence_length=t2_neg_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_shared', swap_memory=True)
        
        t2_neg_a_output, t2_neg_a_final_state = tf.nn.dynamic_rnn(cell= tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size),
                                            inputs= shared_neg_a_output,
                                            sequence_length= t2_neg_a_tf_size, 
                                            dtype=tf.float32, scope='lstm_answer_t2', swap_memory=True)

    with tf.variable_scope('l2norm'):
        
        t1_q_tf_size_r = tf.to_float(tf.reshape(t1_q_tf_size, [-1, 1]))
        t1_pos_a_tf_size_r = tf.to_float(tf.reshape(t1_pos_a_tf_size, [-1, 1]))
        t1_neg_a_tf_size_r = tf.to_float(tf.reshape(t1_neg_a_tf_size, [-1, 1]))
        
        t1_q_mean_state = tf.mul(tf.reduce_sum(t1_q_output, reduction_indices=1), tf.inv(t1_q_tf_size_r))
        t1_pos_mean_state = tf.mul(tf.reduce_sum(t1_pos_a_output, reduction_indices=1), tf.inv(t1_pos_a_tf_size_r))
        t1_neg_mean_state = tf.mul(tf.reduce_sum(t1_neg_a_output, reduction_indices=1), tf.inv(t1_neg_a_tf_size_r))
        
        t1_norm_question = tf.nn.l2_normalize(t1_q_mean_state, dim=1, name='t1_norm_question')
        t1_norm_pos_answer = tf.nn.l2_normalize(t1_pos_mean_state,dim=1, name='t1_norm_pos_answer')
        t1_norm_neg_answer = tf.nn.l2_normalize(t1_neg_mean_state,dim=1, name='t1_norm_neg_answer')
        
        t2_q_tf_size_r = tf.to_float(tf.reshape(t2_q_tf_size, [-1, 1]))
        t2_pos_a_tf_size_r = tf.to_float(tf.reshape(t2_pos_a_tf_size, [-1, 1]))
        t2_neg_a_tf_size_r = tf.to_float(tf.reshape(t2_neg_a_tf_size, [-1, 1]))
        
        t2_q_mean_state = tf.mul(tf.reduce_sum(t2_q_output, reduction_indices=1), tf.inv(t2_q_tf_size_r))
        t2_pos_mean_state = tf.mul(tf.reduce_sum(t2_pos_a_output, reduction_indices=1), tf.inv(t2_pos_a_tf_size_r))
        t2_neg_mean_state = tf.mul(tf.reduce_sum(t2_neg_a_output, reduction_indices=1), tf.inv(t2_neg_a_tf_size_r))
        
        t2_norm_question = tf.nn.l2_normalize(t2_q_mean_state, dim=1, name='t2_norm_question')
        t2_norm_pos_answer = tf.nn.l2_normalize(t2_pos_mean_state,dim=1, name='t2_norm_pos_answer')
        t2_norm_neg_answer = tf.nn.l2_normalize(t2_neg_mean_state,dim=1, name='t2_norm_neg_answer')
        
        
    for variable in tf.get_collection(tf.GraphKeys.VARIABLES, scope='lstm_question'):
        tf.add_to_collection('losses', reg * tf.nn.l2_loss(variable))
        
    for variable in tf.get_collection(tf.GraphKeys.VARIABLES, scope='lstm_answer'):
        tf.add_to_collection('losses', reg * tf.nn.l2_loss(variable))

    with tf.variable_scope('loss'):
        t1_loss = tf.reduce_mean(tf.maximum(margin - tf.reduce_sum(tf.mul(t1_norm_question, t1_norm_pos_answer), reduction_indices=1) + \
        tf.reduce_sum(tf.mul(t1_norm_question, t1_norm_neg_answer), reduction_indices=1), 0))
        
        t2_loss = tf.reduce_mean(tf.maximum(margin - tf.reduce_sum(tf.mul(t2_norm_question, t2_norm_pos_answer), reduction_indices=1) + \
        tf.reduce_sum(tf.mul(t2_norm_question, t2_norm_neg_answer), reduction_indices=1), 0))
        
        joint_loss = t1_loss + t2_loss
        tf.add_to_collection('losses', joint_loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        
    with tf.variable_scope('optimizer_shared'):
        t1_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        t1_grads_and_vars = t1_optimizer.compute_gradients(total_loss)
        t1_capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in t1_grads_and_vars]
        train_op = t1_optimizer.apply_gradients(t1_capped_grads_and_vars)
        
        # t2_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # t2_grads_and_vars = t2_optimizer.compute_gradients(loss)
        # t2_capped_grads_and_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in t2_grads_and_vars]
        # t2_optimizer.apply_gradients(t2_capped_grads_and_vars)
        
    with tf.variable_scope('validation'):
        
        t1_dif = tf.reduce_sum(tf.mul(t1_norm_question, t1_norm_pos_answer) - tf.mul(t1_norm_question, t1_norm_neg_answer), reduction_indices=1)
        t1_valid_result = tf.shape(tf.where(tf.greater_equal(t1_dif,0))) 
        
        t2_dif = tf.reduce_sum(tf.mul(t2_norm_question, t2_norm_pos_answer) - tf.mul(t2_norm_question, t2_norm_neg_answer), reduction_indices=1)
        t2_valid_result = tf.shape(tf.where(tf.greater_equal(t2_dif,0))) 
    
    
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())    
    train_writer = tf.train.SummaryWriter(TB_FOLDER,sess.graph) 
    acc_sum = tf.scalar_summary('accuracy', accuracy)
    t1_acc_sum = tf.scalar_summary('t1_accuracy', t1_accuracy)
    t2_acc_sum = tf.scalar_summary('t2_accuracy', t2_accuracy)
    loss_sum = tf.scalar_summary('log_loss', log_loss)
    lstm_var = tf.all_variables()
    hist_sum = []
    for var in lstm_var:
        hist_sum.append(tf.histogram_summary(var.name, var))
    merged = tf.merge_all_summaries()
    
    
    
    t1_train_index = np.arange(0, len(t1_train_q_input))
    t2_train_index = np.arange(0, len(t2_train_q_input))
    acc_loss = 0

    for epoch in range(5):
        np.random.shuffle(t1_train_index)
        np.random.shuffle(t2_train_index)
        train_q_input = min(len(t1_train_q_input), len(t2_train_q_input))
        for itr in range(train_q_input / batch_size):
            loss_value, _ = sess.run([joint_loss, train_op], feed_dict={
                    t1_q_input : t1_train_q_input[t1_train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    t1_pos_a_input : t1_train_pos_ans_input[t1_train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    t1_neg_a_input : t1_train_neg_ans_input[t1_train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    t1_q_tf_size : t1_train_q_len[t1_train_index[itr * batch_size:(itr + 1) * batch_size]],
                    t1_pos_a_tf_size : t1_train_pos_ans_len[t1_train_index[itr * batch_size:(itr + 1) * batch_size]],
                    t1_neg_a_tf_size : t1_train_neg_ans_len[t1_train_index[itr * batch_size:(itr + 1) * batch_size]],
                    
                    t2_q_input : t2_train_q_input[t2_train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    t2_pos_a_input : t2_train_pos_ans_input[t2_train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    t2_neg_a_input : t2_train_neg_ans_input[t2_train_index[itr * batch_size:(itr + 1) * batch_size], :],
                    t2_q_tf_size : t2_train_q_len[t2_train_index[itr * batch_size:(itr + 1) * batch_size]],
                    t2_pos_a_tf_size : t2_train_pos_ans_len[t2_train_index[itr * batch_size:(itr + 1) * batch_size]],
                    t2_neg_a_tf_size : t2_train_neg_ans_len[t2_train_index[itr * batch_size:(itr + 1) * batch_size]]
                    
                })
                    #options=run_options,
                    #run_metadata=run_metadata)
            #train_writer.add_run_metadata(run_metadata, 'step%d' % itr)
            #print loss_value
            acc_loss += loss_value

            # validation     
            if itr != 0 and itr % iter_log == 0:
                log_acc_loss = acc_loss / iter_log
                print '==> itr: %d, loss: %f' % (itr, log_acc_loss)
                print '==> itr: %d, validation' % (itr)
                t1_count = 0
                t2_count = 0
                
                for val_itr in tqdm(range(len(t1_validation_q_input) / batch_size)):
            
                    t1_dif_arr, t1_count_shape = sess.run([t1_dif, t1_valid_result], feed_dict={
                                t1_q_input : t1_validation_q_input[val_itr * batch_size:(val_itr + 1) * batch_size, :],
                                t1_pos_a_input : t1_validation_pos_ans_input[val_itr * batch_size:(val_itr + 1) * batch_size,:], :],
                                t1_neg_a_input : t1_validation_neg_ans_input[val_itr * batch_size:(val_itr + 1) * batch_size,:], :],
                                t1_q_tf_size : t1_validation_q_len[val_itr * batch_size:(val_itr + 1) * batch_size],
                                t1_pos_a_tf_size : t1_validation_pos_ans_len[val_itr * batch_size:(val_itr + 1) * batch_size],
                                t1_neg_a_tf_size : t1_validation_neg_ans_len[val_itr * batch_size:(val_itr + 1) * batch_size]
                             })
                    t1_count += t1_count_shape[0]
                '''if val_itr % 20 == 0:
                        print count
                    print dif_arr
                    break
                break'''
                    #print '==> validation: %d' % val_itr
                
                for val_itr in tqdm(range(len(t2_validation_q_input) / batch_size)):
            
                    t2_dif_arr, t2_count_shape = sess.run([t2_dif, t2_valid_result], feed_dict={
                                t2_q_input : t2_validation_q_input[val_itr * batch_size:(val_itr + 1) * batch_size, :],
                                t2_pos_a_input : t2_validation_pos_ans_input[val_itr * batch_size:(val_itr + 1) * batch_size, :], :],
                                t2_neg_a_input : t2_validation_neg_ans_input[val_itr * batch_size:(val_itr + 1) * batch_size, :], :],
                                t2_q_tf_size : t2_validation_q_len[val_itr * batch_size:(val_itr + 1) * batch_size],
                                t2_pos_a_tf_size : t2_validation_pos_ans_len[val_itr * batch_size:(val_itr + 1) * batch_size],
                                t2_neg_a_tf_size : t2_validation_neg_ans_len[val_itr * batch_size:(val_itr + 1) * batch_size]
                             })
                    t2_count += t2_count_shape[0]
                
                t1_acc = t1_count * 1.0 / len(t1_validation_q_input)
                t2_acc = t2_count * 1.0 / len(t2_validation_q_input)
                print '==> validation accuracy:\n Task 1: %f\n Task 2: %f' % (t1_acc, t2_acc)
                overall_acc = (t1_count + t2_count) * 1.0 / (len(t1_validation_q_input) + len(t2_validation_q_input))
                print '==> overall accuracy: %f' % (overall_acc)
                summary = sess.run(merged, feed_dict={t1_accuracy : t1_acc, t2_accuracy: t2_acc, accuracy: overall_acc, log_loss:log_acc_loss})
                train_writer.add_summary(summary, itr)
                acc_loss = 0
            if itr != 0 and itr % 2*iter_log == 0:
                save_path = saver.save(sess, TB_FOLDER + 'model_' + str(itr))
        acc_loss = 0
            
if __name__ == '__main__':
    if len(sys.argv) == 7:
        TASK1_DATA_FOLDER = sys.argv[1]
        TASK2_DATA_FOLDER = sys.argv[2]
        TB_FOLDER = sys.argv[3]
        reg = float(sys.argv[4])
        lstm_size= int(sys.argv[5])
        batch_size = int(sys.argv[6])
        iter_log = 100
        
        if (not os.path.isdir(TASK1_DATA_FOLDER)) or (not os.path.isdir(TASK2_DATA_FOLDER)):
            print 'not valid data folder...'
        elif (not os.path.isdir(TB_FOLDER)):
            print 'not valid tensorboard folder...'
        else: 
            lstm_multitask(TASK1_DATA_FOLDER, TASK2_DATA_FOLDER, TB_FOLDER, reg=reg, lstm_size=lstm_size, batch_size = batch_size, iter_log = iter_log)
    else:
        print 'input data folder...'