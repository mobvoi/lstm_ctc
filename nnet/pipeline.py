# Copyright 2018 Mobvoi Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.


# Copyright 2018 Mobvoi Inc. All Rights Reserved.
# Author: cfyeh@mobvoi.com (Ching-Feng Yeh)

#!/usr/bin/python2

import math
import tensorflow as tf


def create_pipeline_sequence_batch(dataset,
                                   input_dim,
                                   batch_size=64,
                                   batch_threads=8,
                                   num_epochs=1):
    ''' create_pipeline_sequence_batch() is for recurrent models such as
        RNN / LSTM / ..., and mostly for training / validation.
        The shape of the returned pipeline['nnet_input'] is
        (batch_size, max_seq_len, dim)
    '''

    padded_shapes = dict()
    padded_shapes['nnet_input'] = [None, input_dim]
    padded_shapes['nnet_target'] = [None]
    padded_shapes['sequence_length'] = []
    padded_shapes['target_length']=[]
    padding_values = dict()
    padding_values['nnet_input'] = tf.constant(0, dtype=tf.float32)
    padding_values['nnet_target'] = tf.constant(-1, dtype=tf.int64)
    padding_values['sequence_length'] = tf.constant(-1, dtype=tf.int32)
    padding_values['target_length'] = tf.constant(-1, dtype=tf.int32)

    dataset = \
        dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
        )
    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()

    initializer = iterator.initializer
    
    pipeline = dict()
    pipeline['nnet_input'] = batch['nnet_input']
    pipeline['sequence_length'] = tf.cast(batch['sequence_length'], tf.int32)
    pipeline['nnet_target'] = batch['nnet_target']
    pipeline['target_length'] = tf.cast(batch['target_length'], tf.int32)

    return initializer, pipeline


def create_pipeline_sequential(filename,
                               tfrecord,
                               num_epochs=1):
    ''' create_pipeline_sequential() is for inference.
        The shape of the returned pipeline['nnet_input'] is
        (num_frames, dim)
    '''

    dataset = tf.data.Dataset.zip((filename, tfrecord))
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    filename, tfrecord = iterator.get_next()

    initializer = iterator.initializer

    pipeline = dict()
    pipeline['filename'] = filename
    pipeline['nnet_input'] = tfrecord['nnet_input']
    pipeline['sequence_length'] = tfrecord['sequence_length']

    return initializer, pipeline
