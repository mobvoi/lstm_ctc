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


import math
import tensorflow as tf
from moe import create_moe
from class_prior import get_class_prior


def create_logits_blstm(nnet_input, sequence_length, nnet_config):
    """Create logits for a simple bidirectional-lstm model
    1. naive batch normalization doesn't work. In naive batch normalization, bn 
    is applied after the each bilstm layer. Maybe other way of doing BN works


    Args:
        nnet_input: nnet input (tf tensor)
        sequence_length : input sequence length (tf tensor)
        nnet_config: nnet config (dict)

    Return:
        logits: logits (tf tensor)
    """
    input_dim = nnet_config.get('input_dim')
    log = 'create_logits_blstm(): input_dim = %d' % input_dim
    tf.logging.info(log)
    left_context = nnet_config.get('left_context')
    log = 'create_logits_blstm(): left_context = %d' % left_context
    tf.logging.info(log)
    right_context = nnet_config.get('right_context')
    log = 'create_logits_blstm(): right_context = %d' % right_context
    tf.logging.info(log)
    input_dim *= (1 + left_context + right_context)
    log = 'create_logits_blstm(): input_dim (including context) = %d' % input_dim
    tf.logging.info(log)
    num_layers = nnet_config.get('num_layers')
    log = 'create_logits_blstm(): num_layers = %d' % num_layers
    tf.logging.info(log)
    num_neurons = nnet_config.get('num_neurons')
    log = 'create_logits_blstm(): num_neurons = %d' % num_neurons
    tf.logging.info(log)
    num_targets = nnet_config.get('num_targets')
    log = 'create_logits_blstm(): num_targets = %d' % num_targets
    tf.logging.info(log)
    num_projects = nnet_config.get('num_projects')
    if num_projects is not None:
        log = 'create_logits_blstm(): num_projects = %d' % num_projects
        tf.logging.info(log)
    use_peepholes = nnet_config.get('use_peepholes')
    if use_peepholes is None:
        use_peepholes = False
    log = 'create_logits_blstm(): use_peepholes = %s' % use_peepholes
    tf.logging.info(log)
    num_experts = nnet_config.get('num_experts')
    if num_experts is not None:
        log = 'create_logits_blstm(): num_experts = %d' % num_experts
        tf.logging.info(log)
    moe_temp = nnet_config.get('moe_temp')
    if moe_temp is None:
        moe_temp = 10.0
    log = 'create_logits_blstm(): moe_temp = %f' % moe_temp
    tf.logging.info(log)
    dropout_rate = nnet_config.get('dropout_rate')
    log = 'create_logits_blstm(): dropout_rate = %f' % dropout_rate
    tf.logging.info(log)
    uniform_label_sm = nnet_config.get('uniform_label_sm')
    if uniform_label_sm is not None:
        log = 'create_logits_blstm(): uniform label sm weight = %f' % uniform_label_sm
        tf.logging.info(log)
    prior_label_sm = nnet_config.get('prior_label_sm')
    if prior_label_sm is not None:
        log = 'create_logits_blstm(): prior_label_sm = %f' % prior_label_sm
        tf.logging.info(log)
    prior_label_path = nnet_config.get('prior_label_path')
    if prior_label_path is not None:
        log = 'create_logits_blstm(): prior_label_path = %s' % prior_label_path 
        tf.logging.info(log)
    is_training = nnet_config.get('is_training')
    if is_training is None:
        is_training = True
    log = 'create_logits_blstm(): is_training = %s' % is_training
    tf.logging.info(log)
    if is_training == False:
        dropout_rate = 1.0
        log = 'create_logits_blstm(): is not in training, turn off dropout'
        tf.logging.info(log)
    

    seed = None
    reg_loss =[]

    nnet_input_shape = tf.shape(nnet_input)
    batch_size = nnet_input_shape[0]
    lstm_input_dim = input_dim

    # reverse over sequence length dimension
    back_nnet_input=tf.reverse_sequence(nnet_input, sequence_length, seq_axis=1, batch_axis=0)

    # append sil at the beginning of each nnet input for ornn regularization
    #if weight_ornn is not None and weight_ornn !=0:
    #    sil = tf.zeros([batch_size,1, lstm_input_dim ], dtype=tf.float32) 
    #    poutput = tf.concat([nnet_input, sil], 1)
    #    nnet_input = tf.concat([sil, nnet_input], 1)
    #    back_poutput = tf.concat([back_nnet_input, sil], 1)
    #    back_nnet_input = tf.concat([sil, back_nnet_input], 1)
    #    sequence_length = sequence_length + 1

    
    # Building BLSTMs
    with tf.variable_scope("frnn"):
        # forward 
        forward_cells = \
            [ tf.contrib.rnn.DropoutWrapper(
              tf.contrib.rnn.LSTMCell(
                  num_units=num_neurons,
                  num_proj=num_projects,
                  use_peepholes=use_peepholes,
                  forget_bias=5.0,
                  state_is_tuple=True,
                  name="frnn"+str(i)
              ),
              output_keep_prob=dropout_rate) for i in xrange(num_layers) ]

                
        forward_initial_states = \
            [ cell.zero_state(
                  batch_size=batch_size,
                  dtype=tf.float32,
              ) for cell in forward_cells ]

                # backward
    with tf.variable_scope("brnn"): 
        backward_cells = \
            [ tf.contrib.rnn.DropoutWrapper( 
              tf.contrib.rnn.LSTMCell(
                  num_units=num_neurons,
                  num_proj=num_projects,
                  use_peepholes=use_peepholes,
                  forget_bias=5.0,
                  state_is_tuple=True,
                  name="brnn"+str(i)
              ),
              output_keep_prob=dropout_rate ) for i in xrange(num_layers) ]
        

        backward_initial_states = \
            [ cell.zero_state(
                  batch_size=batch_size,
                  dtype=tf.float32,
              ) for cell in backward_cells ]
        
    finput=nnet_input
    binput=back_nnet_input

    for i in xrange(num_layers):
        forward_output, fw_state = \
            tf.nn.dynamic_rnn(
                cell=forward_cells[i],
                inputs=finput,
                sequence_length=sequence_length,
                initial_state=forward_initial_states[i],
                dtype=tf.float32,
                scope="fd"+str(i)
            )
        backward_output, bw_state = \
            tf.nn.dynamic_rnn(
                cell=backward_cells[i],
                inputs=binput,
                sequence_length=sequence_length,
                initial_state=backward_initial_states[i],
                dtype=tf.float32,
                scope="bd"+str(i)
            )
        
        reverse_backward_output = tf.reverse_sequence(backward_output,sequence_length, seq_axis=1, batch_axis=0)
        ## here we  concatenation. we tried addition. According to several
        ## epoches of training. The addition performs worse than concatenation
        ## finput= tf.concat([forward_output, reverse_backward_output], 1)
        ## finput = forward_output + reverse_backward_output
        
        ## To keep the size of bidirectional lstm as unidirectional lstm, we don't 
        ## do residual in the first Bidirectional LSTM. Take input size 360 and num_projects
        ## 180 as example. 
        if i==0 and input_dim == 2 * num_projects :
            finput=finput + tf.concat([forward_output, reverse_backward_output], 2)
        else:
            finput = tf.concat([forward_output, reverse_backward_output], 2)
        binput=tf.reverse_sequence(finput,sequence_length, seq_axis=1, batch_axis=0)

    ## encoder
    fw_enc=tf.concat(fw_state, 1)
    bw_enc=tf.concat(bw_state,1)
    encoder = tf.concat([fw_enc, bw_enc],1)

    ## reverse back back_output
    output = finput
    output_dim = num_neurons if num_projects is None else num_projects
    output_dim = output_dim * 2
    log = 'create_logits_blstm(): output_dim = %d' % output_dim
    tf.logging.info(log)

    ploss = None
    #if weight_ornn is not None and \
    #   weight_ornn != 0:
    #   if lstm_input_dim != num_projects:
    #       print("fpro_size should be equal to num_projects")
    #       exit(1)
    #   ploss = tf.losses.mean_squared_error(forward_output, poutput)\
    #         + tf.losses.mean_squared_error(backward_output, back_poutput)
    #   ploss = weight_ornn * ploss

    output = tf.reshape(output, [-1, output_dim])

    if num_experts is not None and num_experts > 0:
        y = create_moe(output, \
                      output_dim, \
                      num_targets, \
                      num_experts,
                      moe_temp,
                      dropout_rate
                     ) 
    else:
        # Feed-forward for the last layer
        stddev = 1.0 / math.sqrt(float(num_neurons))
        W = tf.Variable(
                tf.truncated_normal(
                    [output_dim, num_targets],
                    stddev=stddev
                )
            )
        b = tf.Variable(
                tf.zeros([num_targets])
            )
        y = tf.nn.xw_plus_b(output, W, b)
    y = tf.reshape(y, [batch_size, -1, num_targets])

    logits = y

    # label smooth part
    if uniform_label_sm is not None and uniform_label_sm >0:
        uni_prob = tf.constant([1.0/num_targets]*num_targets, tf.float32)
        uni_prob = tf.expand_dims(tf.expand_dims(uni_prob,0), 0)
        pred_prob = tf.nn.softmax(logits)
        kl_dis=tf.multiply((tf.log(pred_prob) -tf.log(uni_prob)),pred_prob)
        sm_loss = tf.reduce_sum(kl_dis) * uniform_label_sm
        reg_loss.append((sm_loss, uniform_label_sm))
    elif prior_label_sm is not None and prior_label_sm >0 and \
        prior_label_path is not None:
        class_prior = get_class_prior( prior_label_path)
        prior_prob = tf.constant(class_prior, tf.float32)
        pred_prob = tf.nn.softmax(logits)
        kl_dis = tf.multiply((tf.log(pred_prob) - prior_prob), pred_prob)
        sm_loss = tf.reduce_sum(kl_dis) * prior_label_sm
        reg_loss.append((sm_loss, prior_label_sm))


    
    return logits, encoder, reg_loss
