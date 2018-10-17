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


#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
File: moe.py
Author: Yangyang Shi
Email: yyshi@mobvoi.com
Description: 
  mixture of exports are added on top of LSTM
"""

import math
import tensorflow as tf

def create_moe(lstm_output, output_dim,  num_targets, 
               num_experts, moe_temperature,  dropout_rate):
    
    # Feed-forward for the prior
    stddev = 1.0 / math.sqrt(float(output_dim))
    W_prior = tf.Variable(
            tf.truncated_normal(
                [output_dim, num_experts],
                stddev=stddev
            )
        )
    b_prior = tf.Variable(
            tf.zeros([num_experts])
        )
    y_prior = tf.nn.xw_plus_b(lstm_output, W_prior, b_prior)
    y_prior = tf.expand_dims(y_prior, 2)
    y_prior = tf.nn.softmax(y_prior, axis=1, name="prior")
    y_prior = tf.nn.dropout(y_prior, dropout_rate)

    # Feed-forward for the last layer
    stddev = 1.0 / math.sqrt(float(output_dim))
    W = tf.Variable(
            tf.truncated_normal(
                [output_dim, num_targets*num_experts],
                stddev=stddev
            )
        )
    b = tf.Variable(
            tf.zeros([num_targets*num_experts])
        )
    y_decoder = moe_temperature*tf.tanh(tf.nn.xw_plus_b(lstm_output, W, b))
    y_decoder = tf.reshape(y_decoder, [ -1,  num_experts, num_targets])
    y_decoder = tf.nn.dropout(y_decoder, dropout_rate)

    # average over batch_size and sequence length
    #   y_base = tf.reduce_mean(y_decoder, axis=0)
    #   bbt = tf.matmul(y_base, tf.transpose(y_base))
    #   bbt_norm = tf.norm(bbt, ord=2)+1e-6

    #   y_dist = tf.truediv(bbt, bbt_norm) \
    #       - tf.eye(num_experts)
    #   y_dist = tf.norm(y_dist, ord=2)
    y = tf.reduce_sum(tf.multiply(y_prior, y_decoder), 1)
    return y


