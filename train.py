#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/27 下午3:46
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from libs.configs import cfgs
from libs.nets.model import RNN


def main(argv):

    mnist = input_data.read_data_sets("data/mnist", one_hot=True)
    train_images, train_labels = mnist.train.images, mnist.train.labels
    test_images, test_labels = mnist.test.images, mnist.test.labels

    model = RNN(input_size=cfgs.INPUT_SIZE, time_steps=cfgs.TIME_STEPS, num_layers=cfgs.NUM_LAYERS,
                num_units=cfgs.NUM_UNITS)

    saver = tf.train.Saver(max_to_keep=30)

    # get computer graph
    graph = tf.get_default_graph()

    write = tf.summary.FileWriter(logdir=cfgs.SUMMARY_PATH, graph=graph)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.op.name, var.shape)
        # get and add histogram to summary protocol buffer

        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()

        train_step_per_epoch = mnist.train.num_examples // cfgs.BATCH_SIZE
        test_step_pre_epoch = mnist.test.num_examples // cfgs.BATCH_SIZE

        for epoch in range(1, cfgs.NUM_EPOCH+1):
            train_bar = tqdm(range(1, train_step_per_epoch+1))
            for step in train_bar:
                x_train, y_train = mnist.train.next_batch(cfgs.BATCH_SIZE)
                x_train = x_train.reshape(cfgs.BATCH_SIZE, cfgs.TIME_STEPS, cfgs.INPUT_SIZE)
                feed_dict = model.fill_feed_dict(x_train, y_train, keep_prob=cfgs.KEPP_PROB)
                summary, global_step, train_loss, train_acc, _ = sess.run([summary_op, model.global_step, model.loss, model.acc, model.train],
                                                                          feed_dict=feed_dict)
                if step % cfgs.SMRY_ITER == 0:
                    write.add_summary(summary=summary, global_step=global_step)
                    write.flush()

                train_bar.set_description("Epoch {0} : Step {1} => Train Loss: {2:.4f} | Train ACC: {3:.4f}".
                                          format(epoch, step, train_loss, train_acc))
            test_loss_list = []
            test_acc_list = []
            for step in range(test_step_pre_epoch):
                x_test, y_test = mnist.test.next_batch(cfgs.BATCH_SIZE)
                x_test = x_test.reshape(cfgs.BATCH_SIZE, cfgs.TIME_STEPS, cfgs.INPUT_SIZE)
                feed_dict = model.fill_feed_dict(x_test, y_test, keep_prob=1.0)

                test_loss, test_acc, _ = sess.run([model.loss, model.acc, model.train], feed_dict=feed_dict)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            test_loss = sum(test_loss_list) / len(test_loss_list)
            test_acc = sum(test_acc_list) / len(test_acc_list)
            print("Epoch {0} : Step {1} => Val Loss: {2:.4f} | Val ACC: {3:.4f} ".format(epoch, step,
                                                                                             test_loss, test_acc))
            ckpt_file = os.path.join(cfgs.TRAINED_CKPT, 'model_loss={0:4f}.ckpt'.format(test_loss))
            saver.save(sess=sess, save_path=ckpt_file, global_step=global_step)
    sess.close()
    print('model training has complete')


if __name__ == "__main__":

    tf.app.run()