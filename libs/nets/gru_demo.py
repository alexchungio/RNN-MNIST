#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : gru_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/2 下午4:39
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf

BATCH_SIZE = 32 # the batch size of input data
INPUT_SIZE = 28 # the number in singe time dimension of a single sequence of input data
NUM_UNITS = 128  # hide layer size
TIME_STEPS = 10  # number of sequence size
NUM_LAYERS = 3
NUM_MULTI_UNITS = [64, 128, 256]


def get_gru_cell(num_units=128, activation='tanh'):

    return tf.nn.rnn_cell.GRUCell(num_units=num_units, activation=activation)


if __name__ == "__main__":

    gru_cell = get_gru_cell(NUM_UNITS)
    # c => carry state  h => hide state
    print(gru_cell.state_size) # 128

    inputs = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_data")

    state_0 =  gru_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    # ----------------- cell calculate step---------------------------------
    #  w_r => reset_gate, w_u => update_gate  w_c => candidate
    # input = (batch_size, input_size)
    # state = (batch_size, num_units)
    # w_r = w_u = w_c = (input_size + num_units, num_units)
    # step 1 calculate r and u
    # r = tf.matmul(tf.concat((input, state), axis=1), w_r) => (batch_size, num_units)
    # u = tf.matmul(tf.concat((input, state), axis=1), w_u) => (batch_size, num_units)

    # step 2 calculate c(candidate)
    # c = tf.matmul(tf.concat((input, r), axis=1), w_c) => (batch_size, num_units)
    # new_h = tf.multiply(u, h) + tf.multiply(1-u, c)

    # return new_h, new_h
    # ------------------- real calculate step---------------
    # inputs = tf.concat((input, state), axis=1) => (batch_size, input_size + num_inputs)
    # gate_kernel = (input_size+num_units, num_units * 2)
    # gate__bias = (num_units*2,)
    # candidate_kernel = (input_size+num_units, num_units)
    # candidate_kernel = (num_units,)
    # gate_inputs = tf.matmul(inputs, gate_kernel)
    # r, u = split(value=gate_inputs, num_or_size_splits=2, axis=-1)
    # c = tf.matmul(tf.concat((input, r), axis=1), candidate_kernel)
    # new_h = tf.multiply(u, state) + tf.multiply(1-u, c)
    # return new_h, new_h
    outputs, states = gru_cell(inputs=inputs, state=state_0)  # outputs = states
    print(outputs.shape)
    print(states.shape)

    # ------------------------------construct two step gru------------------------------
    tow_step_graph = tf.Graph()
    with tow_step_graph.as_default():
        gru_cell = get_gru_cell(num_units=NUM_UNITS)

        input_1 = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_1")
        input_2 = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_2")

        state_0 = gru_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

        output_1, state_1 = gru_cell(inputs=input_1, state=state_0)

        output_2, state_2 = gru_cell(inputs=input_2, state=state_1)
        print(output_2.shape)
        print(state_2.shape)


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

        assert (output_2 == state_2).all()  # h_2 == state_2.h
        print('Two step test done !')

        # ----------------------construct multi step gru------------------------------
        # tf.nn.dynamic_rnn
        multi_step_graph = tf.Graph()
        with multi_step_graph.as_default():
            gru_cell = get_gru_cell(num_units=NUM_UNITS)

            inputs = tf.placeholder(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32, name="inputs_1")

            state_0 = gru_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

            outputs, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=inputs, initial_state=state_0)

            print(outputs.shape)
            print(states.shape)

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
            assert (outputs_last_step == states).all()  # h_2 == state_2.h
            print('Multi step test done !')

    # ----------------------construct multi layer gru------------------------------
    multi_layer_graph = tf.Graph()
    with multi_layer_graph.as_default():

        inputs = tf.placeholder(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, name="inputs_1")

        cells = [get_gru_cell(num_units=NUM_MULTI_UNITS[i]) for i in range(NUM_LAYERS)]

        gru_cells = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

        state_0 = gru_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, states = gru_cells(inputs=inputs, state=state_0)

        print(outputs.shape)
        print(states[-1].shape)

        input_step_batch = tf.random_normal(shape=(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.initialize_local_variables())

    with tf.Session(graph=multi_layer_graph) as sess:
        sess.run(init_op)
        for var in tf.global_variables():
            print(var.op.name, var.shape)  # # (input_size + num_units, num_units*4)
        input_data = input_step_batch.eval()

        outputs, states = sess.run([outputs, states], feed_dict={inputs: input_data})

        assert (outputs == states[-1]).all()  # h_2 == state_2.h
        print('Multi layer test done !')

    # ----------------------construct multi step multi layer gru------------------------------
    multi_step_multi_layer_graph = tf.Graph()
    with multi_step_multi_layer_graph.as_default():

        inputs = tf.placeholder(shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32, name="inputs_1")

        cells = [get_gru_cell(num_units=NUM_MULTI_UNITS[i]) for i in range(NUM_LAYERS)]

        gru_cells = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

        state_0 = gru_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, states = tf.nn.dynamic_rnn(cell=gru_cells, inputs=inputs, initial_state=state_0)

        print(outputs.shape)
        print(states[-1].shape)


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

        assert (outputs_last_step == states_last_layer).all()  # h_2 == state_2.h
        print('Multi step multi layer test done !')

