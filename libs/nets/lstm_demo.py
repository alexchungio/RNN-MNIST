#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : lstm_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/28 上午11:40
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf

BATCH_SIZE = 32 # the batch size of input data
INPUT_SIZE = 28 # the number in singe time dimension of a single sequence of input data
NUM_UNITS = 128  # hide layer size
TIME_STEPS = 10  # number of sequence size
NUM_LAYERS = 3
NUM_MULTI_UNITS = [64, 128, 256]


def get_lstm_cell(num_units=128, activation='tanh'):

    return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, activation=activation)

if __name__ == "__main__":

    lstm_cell = get_lstm_cell(NUM_UNITS)
    # c => carry state  h => hide state
    print(lstm_cell.state_size) # LSTMStateTuple(c=128, h=128)

    inputs = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_data")

    state_0 =  lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    # ----------------- cell calculate step---------------------------------
    #  w_i => input_gate, w_j => input W_f => forget_gate, w_o => output_gate
    # input = (batch_size, input_size)
    # h_state = (batch_size, num_units)
    # c_states = (batch_size, num_units)
    # state = tuple(h_state, c_state)
    # w_i = w_j= W_f = w_o = (input_size + num_units, num_units)
    # z_i = tf.tf.matmul(tf.concat((input, h_state),), w_i) => (batch_size, num_units)
    # new_input = tf.tf.matmul(tf.concat((input, h_state),), w_j) => (batch_size, num_units)
    # z_f = tf.tf.matmul(tf.concat((input, h_state),), w_f) => (batch_size, num_units)
    # z_0 = tf.tf.matmul(tf.concat((input, h_state),), w_0) => (batch_size, num_units)

    # select step => new_input = tf.multiply(sigmoid(z_i), tanh(new_input))
    # forget_step => z_f = tf.multiply(c_states, sigmoid(z_f))
    # new_c = new_input + z_f
    # new_h = tf.multiply(tanh(new_c), sigmoid(new_O))
    # return new_h, tuple(new_h, new_c)
    # ------------------- real calculate step---------------
    # inputs = tf.concat((input, state), axis=1) => (batch_size, input_size + num_inputs)
    # kernel = (input_size+num_units, 4*num_units)
    # bias = (4 * self._num_units)
    # gate_inputs = tf.matmul(inputs, kernel)
    # z_i, new_input, z_f, z_o = split(value=gate_inputs, num_or_size_splits=4, axis=-1)

    # select step => new_input = tf.multiply(sigmoid(z_i), tanh(new_input))
    # forget_step => z_f = tf.multiply(c_states, sigmoid(z_f))
    # new_c = new_input + z_f
    # new_h = tf.multiply(tanh(new_c), sigmoid(new_O))
    # return new_h, tuple(new_h, new_c)

    outputs, states = lstm_cell(inputs=inputs, state=state_0)  # outputs = states.h
    print(outputs.shape)
    print(states.h.shape)
    print(states.c.shape)


    # ------------------------------construct two step lstm------------------------------
    tow_step_graph = tf.Graph()
    with tow_step_graph.as_default():
        lstm_cell = get_lstm_cell(num_units=NUM_UNITS)

        input_1 = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_1")
        input_2 = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_2")

        state_0 = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

        output_1, state_1 = lstm_cell(inputs=input_1, state=state_0)

        output_2, state_2 = lstm_cell(inputs=input_2, state=state_1)
        print(output_2.shape)
        print(state_2.h.shape)
        print(state_2.c.shape)

        input_step_batch_1 = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
        input_step_batch_2 = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.initialize_local_variables())

    with tf.Session(graph=tow_step_graph) as sess:
        sess.run(init_op)

        for var in tf.global_variables():
            print(var.op.name, var.shape)  # (input_size + num_units, num_units*4)

        input_data_1 = input_step_batch_1.eval()
        input_data_2 = input_step_batch_2.eval()

        output_2, state_2 = sess.run([output_2, state_2], feed_dict={input_1: input_data_1,
                                                           input_2: input_data_2})

        assert (output_2 == state_2.h).all()  # h_2 == state_2.h
        print('Two step test done !')

        # ----------------------construct multi step lstm------------------------------
        # tf.nn.dynamic_rnn
        multi_step_graph = tf.Graph()
        with multi_step_graph.as_default():
            lstm_cell = get_lstm_cell(num_units=NUM_UNITS)

            inputs = tf.placeholder(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32, name="inputs_1")

            state_0 = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

            outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inputs, initial_state=state_0)

            print(outputs.shape)
            print(states.h.shape)
            print(states.c.shape)

            input_step_batch = tf.random_normal(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32)

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.initialize_local_variables())

        with tf.Session(graph=multi_step_graph) as sess:
            sess.run(init_op)
            for var in tf.global_variables():
                print(var.op.name, var.shape)  # # (input_size + num_units, num_units*4)
            input_data = input_step_batch.eval()

            outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})
            outputs_last_step = outputs[:, -1, :]
            assert (outputs_last_step == states.h).all()  # h_2 == state_2.h
            print('Multi step test done !')

    # ----------------------construct multi layer lstm------------------------------
    multi_layer_graph = tf.Graph()
    with multi_layer_graph.as_default():

        inputs = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_1")

        cells = [get_lstm_cell(num_units=NUM_MULTI_UNITS[i]) for i in range(NUM_LAYERS)]

        lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

        state_0 = lstm_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, states = lstm_cells(inputs=inputs, state=state_0)

        print(outputs.shape)
        print(states[-1].h.shape)
        print(states[-1].c.shape)

        input_step_batch = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.initialize_local_variables())

    with tf.Session(graph=multi_layer_graph) as sess:
        sess.run(init_op)
        for var in tf.global_variables():
            print(var.op.name, var.shape)  # # (input_size + num_units, num_units*4)
        input_data = input_step_batch.eval()

        outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})

        assert (outputs == states[-1].h).all()  # h_2 == state_2.h
        print('Multi layer test done !')

    # ----------------------construct multi step multi layer lstm------------------------------
    multi_step_multi_layer_graph = tf.Graph()
    with multi_step_multi_layer_graph.as_default():

        inputs = tf.placeholder(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32, name="inputs_1")

        cells = [get_lstm_cell(num_units=NUM_MULTI_UNITS[i]) for i in range(NUM_LAYERS)]

        lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

        state_0 = lstm_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, states = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=inputs, initial_state=state_0)

        print(outputs.shape)
        print(states[-1].h.shape)
        print(states[-1].c.shape)

        input_step_batch = tf.random_normal(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.initialize_local_variables())

    with tf.Session(graph=multi_step_multi_layer_graph) as sess:
        sess.run(init_op)
        for var in tf.global_variables():
            print(var.op.name, var.shape)  # # (input_size + num_units, num_units*4)
        input_data = input_step_batch.eval()

        outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})

        outputs_last_step = outputs[:, -1, :]
        states_last_layer = states[-1]

        assert (outputs_last_step == states_last_layer.h).all()  # h_2 == state_2.h
        print('Multi step multi layer test done !')





