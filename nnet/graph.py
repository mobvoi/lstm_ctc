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


#!/usr/bin/python2

import lstm
import bilstm
import utils
import math
import sys
import tensorflow as tf

def get_create_logits(string):
    if not string:
        return None
    elif string == 'blstm':
        return lstm.create_logits_blstm
    elif string == 'cudnnlstm':
        return lstm.create_logits_cudnnlstm
    elif string == 'lstm':
        return lstm.create_logits_lstm
    else:
        return None


def get_optimizer(string, learning_rate, momentum=0.9):
    if not string:
        return None
    elif string == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif string == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif string == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                                          momentum=momentum)
    else:
        return None


def create_graph_for_validation_ctc(pipeline,
                                    nnet_config):
    graph = dict()

    nnet_input = pipeline['nnet_input']
    graph['nnet_input'] = nnet_input

    sequence_length = pipeline['sequence_length']
    graph['sequence_length'] = sequence_length

    nnet_type = nnet_config.get('nnet_type')
    create_logits = get_create_logits(nnet_type)
    logits, encoder, reg_loss = create_logits(
                 nnet_input=nnet_input,
                 sequence_length=sequence_length,
                 nnet_config=nnet_config,
             )
    graph['logits'] = logits

    
    # Convert from [batch, time, target] to [time, batch, target]
    logits = tf.transpose(logits, (1, 0, 2))

    nnet_target = pipeline['nnet_target']
    graph['raw_target'] = nnet_target
    sparse_indices = \
        tf.where(
            tf.not_equal(
                nnet_target, tf.constant(-1, dtype=tf.int64)
            )
        )
    sparse_values = \
        tf.gather_nd(
            params=nnet_target,
            indices=sparse_indices,
        )
    dense_shape = \
        tf.cast(
            x=tf.shape(nnet_target),
            dtype=tf.int64,
        )
    sparse = \
        tf.SparseTensor(
            indices=sparse_indices,
            values=sparse_values,
            dense_shape=dense_shape,
        )
    sparse = \
        tf.cast(
            x=sparse,
            dtype=tf.int32,
        )
    nnet_target = sparse
    graph['nnet_target'] = nnet_target
    batch_size=tf.shape(sparse_values)[0]
    graph['size']=batch_size
       

    loss = tf.nn.ctc_loss(
             labels=nnet_target,
             inputs=logits,
             sequence_length=sequence_length,
             ignore_longer_outputs_than_inputs = True
           )

    loss = tf.reduce_sum(loss)
    tf.summary.scalar('loss', loss)
    graph['eval_loss'] = loss

    other_weights=0
    other_loss=None
    for item in reg_loss:
        if item[0] is not None and item[1] is not None \
           and item[1]>0:
            other_weights += item[1]
            if other_loss == None:
                other_loss =item[0] 
            else:
                other_loss += item[0]

    if other_loss is not None and other_weights != 0:
        #loss  = (1-other_weights)* loss + other_loss
        loss = loss + other_loss

        
    graph['loss'] = loss  # keep track of the total loss

    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
                                  inputs=logits,
                                  sequence_length=sequence_length,
                                  merge_repeated=True
                              )
    dist = tf.reduce_sum(
               tf.edit_distance(
                   tf.cast(decoded[0], tf.int64),
                   tf.cast(nnet_target, tf.int64),
                   normalize=False
               ),
           )
    graph['eval'] = dist

    global_step = tf.train.get_or_create_global_step()
    global_step = tf.assign(global_step, global_step + 1, name='global_step')
    graph['global_step'] = global_step

    summary = tf.summary.merge_all()
    graph['summary'] = summary

    for key, val in graph.iteritems():
        tf.add_to_collection(key, val)

    return graph


def create_graph_for_training_ctc(pipeline,
                                  nnet_config,
                                  learn_rate,
                                  clip_norm=5.0,
                                  optimizer='sgd',
                                  l2_decay_weight=1e-5):

    graph = create_graph_for_validation_ctc(
                pipeline=pipeline,
                nnet_config=nnet_config,
            )

    lrate = tf.constant(
                learn_rate,
                name='lrate'
            )

    loss = graph['loss']
    tvars = tf.trainable_variables()
    l2_loss = tf.add_n(
          [ tf.nn.l2_loss(v) \
           for v in tvars if 'bias' not in v.name ]) \
            * l2_decay_weight

    loss = loss + l2_loss
    grads, _ = tf.clip_by_global_norm(
          tf.gradients(loss, tvars),clip_norm
        )

    update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tf.logging.info(update)
    with tf.control_dependencies(update):
        optimizer = get_optimizer(optimizer, lrate)
        train = optimizer.apply_gradients(
          zip(grads, tvars)
        )
        #train = optimizer.minimize(loss, name='train')

    graph['lrate'] = lrate
    graph['train'] = train

    for key, val in graph.iteritems():
        tf.add_to_collection(key, val)

    return graph


def create_graph_for_inference(pipeline,
                               nnet_config,
                              smooth_factor=1.0):
    graph = dict()

    filename = pipeline['filename']
    graph['filename'] = filename
    nnet_input = pipeline['nnet_input']
    graph['nnet_input'] = nnet_input
    sequence_length = pipeline['sequence_length']
    graph['sequence_length'] = sequence_length

    nnet_type = nnet_config.get('nnet_type')
    create_logits = get_create_logits(nnet_type)
    if nnet_type == 'blstm' or nnet_type == 'cudnnlstm' or nnet_type == 'lstm':
        nnet_input = tf.expand_dims(nnet_input, 0)
        sequence_length = tf.expand_dims(sequence_length, 0)
        logits, _, _ = create_logits(
                     nnet_input=nnet_input,
                     sequence_length=sequence_length,
                     nnet_config=nnet_config,
                 )
        logits = tf.squeeze(logits, 0)
    graph['logits'] = logits
    graph['nnet_output'] = tf.nn.softmax(smooth_factor * logits)

    for key, val in graph.iteritems():
        tf.add_to_collection(key, val)

    return graph


def create_graph_for_decoding(pipeline,
                              nnet_config):
    graph = dict()

    filename = pipeline['filename']
    graph['filename'] = filename
    nnet_input = pipeline['nnet_input']
    graph['nnet_input'] = nnet_input
    sequence_length = pipeline['sequence_length']
    graph['sequence_length'] = sequence_length

    nnet_type = nnet_config.get('nnet_type')
    create_logits = get_create_logits(nnet_type)
    if nnet_type == 'blstm' or nnet_type == 'lstm':
        nnet_input = tf.expand_dims(nnet_input, 0)
        sequence_length = tf.expand_dims(sequence_length, 0)
        logits = create_logits(
                     nnet_input=nnet_input,
                     sequence_length=sequence_length,
                     nnet_config=nnet_config,
                 )
        # Convert from [batch, time, target] to [time, batch, target]
        logits = tf.transpose(logits, (1, 0, 2))
        decoded, log_probabilities = \
            tf.nn.ctc_beam_search_decoder(
                inputs=logits,
                sequence_length=sequence_length,
                merge_repeated=True
            )
        logits = tf.squeeze(logits, 0)
    graph['logits'] = logits
    graph['nnet_output'] = tf.nn.softmax(logits)
    graph['decoded'] = decoded[0].values

    for key, val in graph.iteritems():
        tf.add_to_collection(key, val)

    return graph
