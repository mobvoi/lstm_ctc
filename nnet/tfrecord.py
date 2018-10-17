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
import random
import sys
import time
import tensorflow as tf
from operator import itemgetter


def _splice(nnet_input, left_context, right_context):
    res = []
    num_rows = tf.shape(nnet_input)[0]
    first_frame = tf.slice(nnet_input, [0, 0], [1, -1])
    last_frame = tf.slice(nnet_input, [num_rows - 1, 0], [1, -1])
    left_padding = tf.tile(first_frame, [left_context, 1])
    right_padding = tf.tile(last_frame, [right_context, 1])
    padded_input = tf.concat([left_padding, nnet_input, right_padding], 0)
    for i in xrange(left_context + right_context + 1):
        frame = tf.slice(padded_input, [i, 0], [num_rows, -1])
        res.append(frame)

    return tf.concat(res, 1)


def _subsample(nnet_input, factor):
    indices = tf.range(tf.shape(nnet_input)[0] / factor) * factor
    subsampled_input = \
        tf.gather(
            params=nnet_input,
            indices=indices,
            axis=0,
        )
    return subsampled_input


def dataset_from_tfrecords(tfrecords_scp,
                           left_context = 0,
                           right_context = 0,
                           subsample = 0,
                           shuffle = False,
                           seed = None,
                           num_parallel_calls = 32):
    tfrecord_list = []
    input_dim = None
    has_label = None
    for line in open(tfrecords_scp, 'r'):
        token = line.rstrip().split()
        fid_ = token[0]
        num_rows_ = int(token[1])
        num_cols_ = int(token[2])
        has_label_ = int(token[3])
        tfrecord_ = token[4]
        tfrecord_list.append(tfrecord_)
        if input_dim is None:
            input_dim = num_cols_
        if has_label is None:
            has_label = has_label_
        if input_dim != num_cols_:
            log = 'inconsistent nnet_input dimension in tfrecords:' + \
                  ' %d vs. %d' % (input_dim, num_cols_)
            tf.logging.fatal(log)
            sys.exit(1)
        if has_label != has_label_:
            log = 'inconsistent has_label in tfrecords:' + \
                  ' %d vs. %d' % (has_label, has_label_)
            tf.logging.fatal(log)
            sys.exit(1)

    if shuffle:
        if seed is None:
            seed = time.time()
        random.seed(seed)
        random.shuffle(tfrecord_list)

    
    def _parse(example_proto):
        sequence_features = dict()

        nnet_input = tf.FixedLenSequenceFeature(shape=[input_dim], dtype=tf.float32)
        sequence_features['nnet_input'] = nnet_input

        if has_label:
            sequence_features['nnet_target'] = \
                tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)

        _, sequence = tf.parse_single_sequence_example(
                          example_proto, sequence_features=sequence_features
                      )

        if left_context or right_context:
            sequence['nnet_input'] = _splice(sequence['nnet_input'], left_context, right_context)
            sequence['nnet_input'].set_shape([None, input_dim * (1 + left_context + right_context)])

        if subsample:
            sequence['nnet_input'] = _subsample(sequence['nnet_input'], subsample)

        sequence['sequence_length'] = tf.shape(sequence['nnet_input'])[0]
        if has_label:
            sequence['target_length'] = tf.shape(sequence['nnet_target'])[0]

        return sequence

    filename = tf.data.Dataset.from_tensor_slices(tfrecord_list)
    tfrecord = tf.data.TFRecordDataset(tfrecord_list).map(
                              _parse, num_parallel_calls=num_parallel_calls)
    input_dim *= (1 + left_context + right_context)
    return filename, tfrecord, input_dim


def write_tfrecord(filename, nnet_input, nnet_target=None):
    num_rows = nnet_input.shape[0]
    num_cols = nnet_input.shape[1]

    writer = tf.python_io.TFRecordWriter(filename)

    feature_list = dict()

    feature = [
        tf.train.Feature(float_list=tf.train.FloatList(value=row))
        for row in nnet_input
    ]
    feature_list['nnet_input'] = \
        tf.train.FeatureList(feature=feature)

    if nnet_target is not None:
        feature = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))
            for val in nnet_target
        ]
        feature_list['nnet_target'] = tf.train.FeatureList(feature=feature)
        

    example = tf.train.SequenceExample(
                  feature_lists=tf.train.FeatureLists(feature_list=feature_list)
              )

    writer.write(example.SerializeToString())
    writer.close()


if __name__=="__main__":
  tfrecord_scp = sys.argv[1]
  output=dataset_from_tfrecords(tfrecord_scp)



