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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: utils.py
Author: Yangyang Shi
Email: yyshi@mobvoi.com
Description: 
"""

import tensorflow as tf


def combine_label_nbest(logits, nnet_target,
                        sequence_length, 
                        beam_width, top_paths):
    # label max length 
    shapes=[nnet_target.dense_shape[1]]

    # get nbest path
    nbest_decoded, log_probs = tf.nn.ctc_beam_search_decoder(
      inputs=logits,
      sequence_length=sequence_length,
      beam_width=beam_width,
      top_paths=top_paths,
      merge_repeated=True
    )

    # get distance for nbest
    nbest_dist_list=[]
    for i in range(0, top_paths):
        shapes.append(nbest_decoded[i].dense_shape[1])
        nbest_dist = tf.edit_distance(
          tf.cast(nbest_decoded[i], tf.int64),
          tf.cast(nnet_target, tf.int64),
          normalize=False
        )
        nbest_dist_list.append(nbest_dist)
 
    # get max length
    max_length = tf.reduce_max(
      tf.stack(shapes,axis=0)
    )
    width = nnet_target.dense_shape[0]

    # convert nbest path to dense tensor
    nbest_tensor_list=[]
    for i in range(0, top_paths):
        nbest_tensor_list.append(
          tf.sparse_to_dense(
            nbest_decoded[i].indices,
            tf.stack([width,max_length]),
            nbest_decoded[i].values,
            default_value=-1
          )
        )

    # combine all the dense tensor to one batch
    nbest_combine = tf.concat(nbest_tensor_list, 0)
    nbest_sequence_length=tf.reduce_sum(
      tf.sign(nbest_combine+1), 
      axis =1
    )
    nbest_weight=tf.sign(
      tf.concat(nbest_dist_list, 0)
    )

    # convert nnet_target to the same shape as nbest
    nnet_target_dense = tf.sparse_to_dense(
      nnet_target.indices,
      tf.stack([width, max_length]),
      nnet_target.values,
      default_value=-1
    )

    nnet_target_length=tf.reduce_sum(
      tf.sign(nnet_target_dense+1),
      axis=1
    )
    
    # nnet target weight should always be 1 
    nnet_target_weight = tf.sign(nnet_target_length)

    # combine nnet_target with nbest
    nbest_combine = tf.cast(nbest_combine, tf.int32)
    nnet_target_weight = tf.cast(
      nnet_target_weight, tf.float32)
    nbest_sequence_length = tf.cast(
      nbest_sequence_length, tf.int32)
    result = tf.concat([nnet_target_dense, nbest_combine],0)
    
    result_weight=tf.stop_gradient(
      tf.concat([nnet_target_weight, nbest_weight],0)
    )
    result_length=tf.stop_gradient(
      tf.concat([nnet_target_length, \
                 nbest_sequence_length],\
                0)
    )

    return result, result_weight, result_length



def fill_blank_path(nbest_tensor_list):
    new_tensor_list=[]
    for i in range(0, len(nbest_tensor_list)):
        nbest_tensor = nbest_tensor_list[i]
        nbest_length = tf.reduce_sum(
          tf.sign(nbest_tensor + 1),
          axis = 1
        )

        # compensation weight for length ==0 -> 1
        # for length > 0 -> 0
        # default empty value is -1 
        # so we need add extra 1 to make label id
        # correct
        comp = -tf.sign(tf.sign(nbest_length) - 1)
        comp = tf.expand_dims(comp, -1)
        j = (i+1)%len(nbest_tensor_list)
        new_tensor = nbest_tensor + \
            tf.multiply(comp, (nbest_tensor_list[j]+1))
        new_tensor_list.append(new_tensor)

    return new_tensor_list

