#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rnn_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/27 下午7:45
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import tensorflow as tf

BATCH_SIZE = 32 # the batch size of input data
INPUT_SIZE = 28 # the number in singe time dimension of a single sequence of input data
NUM_UNITS = 128  # hide layer size
TIME_STEPS = 10  # number of sequence size
NUM_LAYERS = 3
NUM_MULTI_UNITS = [64, 128, 256]


def get_rnn_cell(num_units=128, activation='tanh'):

    return tf.nn.rnn_cell.BasicRNNCell(num_units=num_units, activation=activation)

if __name__ == "__main__":


    # RCNNCell, tensorflow 中 RNN 的基本单元, 子类 BasicRNNCell, BasicLSTMCell
    # (output, cur_state) = cell.call(input, pre_state)
    # eg -> (y_1, h_1) = cell.call(x_1, h_0)

    two_step_graph = tf.Graph()
    with two_step_graph.as_default():

        #--------------- test BasicRNNCell-------------
        rnn_cell =  tf.nn.rnn_cell.BasicRNNCell(num_units=128, activation='tanh')
        print(rnn_cell.state_size)  # 128

        # --------------- construct two time RNN net work------------------------------
        # output_1, state_1 = rnn_cell.__call__(inputs=inputs, state=h_0)

        inputs_1 = tf.placeholder(shape=(32, INPUT_SIZE), dtype=tf.float32)

        h_0 = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)  # get initial state h_0 of all zeros

        #----------------- cell calculate step---------------------------------
        # output = new_state = act(W * input + U * state + B)
        # input = (batch_size, input_size)
        # state = (batch_size, num_units)
        # W = (input_size, num_units)
        # U = (num_units, num_units)
        # B = (batch_size, num_units)
        # output_size = tf.matmul(input,  W) + tf.matmul(state, U) + B
        # = (batch_size, num_units) + (batch_size, num_units) + (batch_size, num_units) = (batch_size, num_units)
        # ------------------- real calculate step---------------
        # kernel = np.concat((W, U), axis=0)=>([input_size + num_units, num_units])
        # inputs = tf.concat((input, state), axis=1) => (batch_size, input_size + num_inputs)
        # bias = (batch_size, num_units)
        # output_size = tf.matmul(input, kernel) + bias = (batch_size, num_units) + (batch_size, num_units) => (batch_size, num_units)
        output_1, state_1 = rnn_cell(inputs=inputs_1, state=h_0)  # output = new_state
        print(output_1.shape)
        print(state_1.shape)
        inputs_2 = tf.placeholder(shape=(32, INPUT_SIZE), dtype=tf.float32)

        output_2, state_2 = rnn_cell(inputs=inputs_2, state=state_1)

        print(output_2.shape)
        print(state_2.shape)

        input_step_batch_1 = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
        input_step_batch_2 = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.group(tf.global_variables_initializer(),
                       tf.initialize_local_variables())

    with tf.Session(graph=two_step_graph) as sess:

        sess.run(init_op)

        for var in tf.global_variables():
            print(var.op.name, var.shape)

        input_step_1 = input_step_batch_1.eval()
        input_step_2 = input_step_batch_2.eval()

        output_1, state_1, output_2, state_2 = sess.run([output_1, state_1, output_1, state_2],
                                                        feed_dict={inputs_1: input_step_1, inputs_2:input_step_2})

        assert output_1.all() == state_1.all()
        assert output_2.all() == state_2.all()
        print('Done !')


    # construct multi step graph
    multi_step_graph = tf.Graph()
    with multi_step_graph.as_default():
        inputs = tf.placeholder(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32)
        rnn_cell =  tf.nn.rnn_cell.BasicRNNCell(num_units=NUM_UNITS, activation='tanh')
        initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=inputs, initial_state=initial_state)
        print(outputs.shape) # all steps outputs
        print(states.shape)  # last step hidden layer states

        input_batch = tf.random_normal(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32)
        init_op = tf.group(tf.global_variables_initializer())

    with tf.Session(graph=multi_step_graph) as sess:

        sess.run(init_op)

        for var in tf.global_variables():
            print(var.op.name, var.shape)

        input_data = input_batch.eval()

        outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})
        outputs_last_step  = outputs[:, -1, :]
        assert outputs_last_step.all() == states.all()
        print('Done !')


    # construct multi layer graph
    multi_layer_graph = tf.Graph()
    with multi_layer_graph.as_default():
        inputs = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
        cells = [get_rnn_cell(NUM_MULTI_UNITS[i]) for i in range(NUM_LAYERS)]
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        print(rnn_cells.state_size) # (64, 128, 256)
        initial_state = rnn_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        # ----------------MultiRNNCell calculate step------------------
        # inputs = (batch_size, input_size)
        # states = [(batch_size, num_units_1), (batch_size, num_units_2), (batch_size, num_units_3)]
        # new_states = []
        # step 1 : output_1, h_1 = cell(inputs, states[0])
        #          new_states.append(h_1)
        # step 2 : output_2, h_2 =  cell(output_1, state[1])
        #          new_states.append(h_2)
        # step 3 : output_3, h_3 =  cell(output_2, state[2])
        #          new_states.append(h_3)
        # return output_3, new_states
        outputs, states = rnn_cells(inputs=inputs, state=initial_state)
        print(outputs.shape)  # (32, 256)
        print(states[0].shape, states[1].shape, states[2].shape)  # (32, 64) (32, 128) (32, 256)

        input_batch = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.global_variables_initializer()

    with tf.Session(graph=multi_layer_graph) as sess:
        sess.run(init_op)

        input_data = input_batch.eval()
        outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})
        assert outputs.all() == states[-1].all()
        print('Done !')


    # ---------------construct multi step multi layer RNN---------------
    # construct multi step multi layer graph
    multi_step_multi_layer_graph = tf.Graph()
    with multi_step_multi_layer_graph.as_default():
        inputs = tf.placeholder(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32)
        cells = [get_rnn_cell(NUM_MULTI_UNITS[i]) for i in range(NUM_LAYERS)]

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        print(rnn_cells.state_size)  # (64, 128, 256)
        initial_state = rnn_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, states = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=inputs, initial_state=initial_state)
        print(outputs.shape)  # (32, 256)
        print(states[0].shape, states[1].shape, states[2].shape)  # (32, 64) (32, 128) (32, 256)

        input_batch = tf.random_normal(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.global_variables_initializer()


    with tf.Session(graph=multi_step_multi_layer_graph) as sess:

        sess.run(init_op)

        for var in tf.global_variables():
            print(var.op.name, var.shape)

        input_data = input_batch.eval()

        outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})
        outputs_last_step  = outputs[:, -1, :]
        states_last_layer = states[-1]
        assert outputs_last_step.all() == states_last_layer.all()
        print('Done !')












