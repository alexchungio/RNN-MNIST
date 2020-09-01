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

    h_0 =  lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
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

    outputs, states = lstm_cell(inputs=inputs, state=h_0)  # outputs = states.h
    print(outputs.shape)
    print(states.h.shape)
    print(states.c.shape)

