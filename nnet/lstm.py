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
from ornn import create_ornn, create_ornn_next
from moe import create_moe
from fpro import feature_project
from bilstm import create_logits_blstm


def create_logits_cudnnlstm(nnet_input, sequence_length, nnet_config):
    """Create logits for a simple unidirectional-lstm model

    Args:
        nnet_input: nnet input (tf tensor)
        nnet_config: nnet config (dict)

    Return:
        logits: logits (tf tensor)
    """
    input_dim = nnet_config.get('input_dim')
    log = 'create_logits_cudnnlstm(): input_dim = %d' % input_dim
    tf.logging.info(log)
    left_context = nnet_config.get('left_context')
    log = 'create_logits_cudnnlstm(): left_context = %d' % left_context
    tf.logging.info(log)
    right_context = nnet_config.get('right_context')
    log = 'create_logits_cudnnlstm(): right_context = %d' % right_context
    tf.logging.info(log)
    input_dim *= (1 + left_context + right_context)
    log = 'create_logits_cudnnlstm(): input_dim (including context) = %d' % input_dim
    tf.logging.info(log)
    num_layers = nnet_config.get('num_layers')
    log = 'create_logits_cudnnlstm(): num_layers = %d' % num_layers
    tf.logging.info(log)
    num_neurons = nnet_config.get('num_neurons')
    log = 'create_logits_cudnnlstm(): num_neurons = %d' % num_neurons
    tf.logging.info(log)
    num_targets = nnet_config.get('num_targets')
    log = 'create_logits_cudnnlstm(): num_targets = %d' % num_targets
    tf.logging.info(log)
    num_projects = nnet_config.get('num_projects')
    if num_projects is not None:
        log = 'create_logits_cudnnlstm(): num_projects = %d' % num_projects
        tf.logging.info(log)
    use_peepholes = nnet_config.get('use_peepholes')
    if use_peepholes is None:
        use_peepholes = False
    log = 'create_logits_cudnnlstm(): use_peepholes = %s' % use_peepholes
    tf.logging.info(log)
    seed = None

    nnet_input_shape = tf.shape(nnet_input)
    batch_size = nnet_input_shape[0]

    # Building LSTMs
    cells = \
        [ tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
              num_units=num_neurons,
          ) for i in xrange(num_layers) ]
    #cells = \
    #    [ tf.contrib.rnn.LSTMCell(
    #          num_units=num_neurons,
    #          num_proj=num_projects,
    #          use_peepholes=True,
    #          state_is_tuple=True,
    #      ) for i in xrange(num_layers) ]
    cells = \
        tf.contrib.rnn.MultiRNNCell(
            cells=cells,
            state_is_tuple=True,
        )
    initial_state = \
        cells.zero_state(
            batch_size=batch_size,
            dtype=tf.float32,
        )
    output, _ = \
        tf.nn.dynamic_rnn(
            cell=cells,
            inputs=nnet_input,
            sequence_length=sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
        )
    output_dim = num_neurons if num_projects is None else num_projects
    log = 'create_logits_cudnnlstm(): output_dim = %d' % output_dim
    tf.logging.info(log)
    output = tf.reshape(output, [-1, output_dim])

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

    return logits


def create_logits_lstm(nnet_input, sequence_length, nnet_config):
    """Create logits for a simple unidirectional-lstm model

    Args:
        nnet_input: nnet input (tf tensor)
        nnet_config: nnet config (dict)

    Return:
        logits: logits (tf tensor)
    """
    input_dim = nnet_config.get('input_dim')
    log = 'create_logits_lstm(): input_dim = %d' % input_dim
    tf.logging.info(log)
    left_context = nnet_config.get('left_context')
    log = 'create_logits_lstm(): left_context = %d' % left_context
    tf.logging.info(log)
    right_context = nnet_config.get('right_context')
    log = 'create_logits_lstm(): right_context = %d' % right_context
    tf.logging.info(log)
    input_dim *= (1 + left_context + right_context)
    log = 'create_logits_lstm(): input_dim (including context) = %d' % input_dim
    tf.logging.info(log)
    num_layers = nnet_config.get('num_layers')
    log = 'create_logits_lstm(): num_layers = %d' % num_layers
    tf.logging.info(log)
    num_neurons = nnet_config.get('num_neurons')
    log = 'create_logits_lstm(): num_neurons = %d' % num_neurons
    tf.logging.info(log)
    num_targets = nnet_config.get('num_targets')
    log = 'create_logits_lstm(): num_targets = %d' % num_targets
    tf.logging.info(log)
    num_projects = nnet_config.get('num_projects')
    if num_projects is not None:
        log = 'create_logits_lstm(): num_projects = %d' % num_projects
        tf.logging.info(log)
    use_peepholes = nnet_config.get('use_peepholes')
    if use_peepholes is None:
        use_peepholes = False
    log = 'create_logits_lstm(): use_peepholes = %s' % use_peepholes
    dropout_rate = nnet_config.get('dropout_rate')
    log = 'create_logits_lstm(): dropout_rate = %f' % dropout_rate
    tf.logging.info(log)
    use_bn = nnet_config.get('use_bn')
    if use_bn is None:
        use_bn=False
    log = 'create_logits_lstm(): use_bn = %s' % use_bn
    tf.logging.info(log)
    num_experts = nnet_config.get('num_experts')
    if num_experts is not None:
        log = 'create_logits_lstm(): num_experts = %d' % num_experts
        tf.logging.info(log)
    weight_orth = nnet_config.get('weight_orth')
    log = 'create_logits_lstm(): weight_orth = %f' % weight_orth
    tf.logging.info(log)
    weight_ornn = nnet_config.get('weight_ornn')
    if weight_ornn is not None:
        log = 'create_logits_lstm(): weight_ornn = %f' % weight_ornn
        tf.logging.info(log)
    weight_ornn_next = nnet_config.get('weight_ornn_next')
    if weight_ornn_next is not None:
        log = 'create_logits_lstm(): weight_ornn_next = %f' % weight_ornn_next
        tf.logging.info(log)
    fpro_size = nnet_config.get('fpro_size')
    if fpro_size is not None:
        log = 'create_logits_lstm(): fpro_size  = %d' % fpro_size
        tf.logging.info(log)
    weight_fpro = nnet_config.get('weight_fpro')
    if weight_fpro is not None:
        log = 'create_logits_lstm(): weight_fpro = %f' % weight_fpro
        tf.logging.info(log)
    is_training = nnet_config.get('is_training')
    if is_training is None:
        is_training = True
    log = 'create_logits_lstm(): is_training = %s' % is_training
    tf.logging.info(log)
    if is_training == False:
        dropout_rate = 1.0
        log = 'create_logits_lstm(): is not in training, turn off dropout'
        tf.logging.info(log)

    seed = None

    nnet_input_shape = tf.shape(nnet_input)
    batch_size = nnet_input_shape[0]

    # Input feature projection
    lstm_input_dim = input_dim
    fp_loss = None 
    if fpro_size is not None and fpro_size !=0:
        nnet_input, fp_loss = feature_project(
            nnet_input,
            input_dim, 
            sequence_length,
            fpro_size)
        if weight_fpro != 0:
            fp_loss = fp_loss * weight_fpro
        else:
            fp_loss = None
        lstm_input_dim = fpro_size 

    # append sil at the beginning of each nnet input for ornn regularization
    #if weight_ornn is not None and weight_ornn !=0 and \
    #   fpro_size is not None and fpro_size !=0
    #    sil = tf.zeros([batch_size,1, lstm_input_dim ], dtype=tf.float32) 
    #    poutput = tf.concat([nnet_input, sil], 1)
    #    nnet_input = tf.concat([sil, nnet_input], 1)
    #    sequence_length = sequence_length + 1

    # Building LSTMs
    with tf.variable_scope("irnn"):
        cells=[]
        for i in xrange(num_layers):
            if i==0 and input_dim != num_projects:
                 cells.append(
                   tf.contrib.rnn.DropoutWrapper(
                     tf.contrib.rnn.LSTMCell(
                      num_units=num_neurons,
                      num_proj=num_projects,
                      use_peepholes=True,
                      state_is_tuple=True,
                      ),
                    output_keep_prob=dropout_rate)
                 )
            else:
                cells.append(
                   tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.ResidualWrapper(
                    tf.contrib.rnn.LSTMCell(
                        num_units=num_neurons,
                        num_proj=num_projects,
                        use_peepholes=True,
                        state_is_tuple=True,
                    )
                    ),
                    output_keep_prob=dropout_rate)
                )

        initial_states = \
            [ cells[i].zero_state(
                batch_size=batch_size,
                dtype=tf.float32,
            ) for i in xrange(num_layers) ] 


    drnn_input=nnet_input
    for i in xrange(num_layers):
        if i==0 and use_bn:
            drnn_input = \
                tf.layers.batch_normalization(
                    drnn_input,
                    training=is_training,
                    name="drnn_bn_0_"+str(i)
                )

        output, _ = \
            tf.nn.dynamic_rnn(
                cell=cells[i],
                inputs=drnn_input,
                sequence_length=sequence_length,
                initial_state=initial_states[i],
                dtype=tf.float32,
                scope="drnn"+str(i)
            )
        if use_bn:
            output = \
                tf.layers.batch_normalization(
                    output,
                    training=is_training,
                    name="drnn_bn"+str(i)
                )
        drnn_input = output

    output_dim = num_neurons if num_projects is None else num_projects
    log = 'create_logits_lstm(): output_dim = %d' % output_dim
    tf.logging.info(log)
    
    ploss = None
    if weight_ornn is not None and \
       weight_ornn != 0:
       if lstm_input_dim != num_projects:
           print("fpro_size should be equal to num_projects")
           exit(1)
       ploss = create_ornn(output,batch_size, \
                           output_dim,sequence_length) 
       #ploss = tf.losses.mean_squared_error(output, poutput)
       ploss = weight_ornn * ploss

    ploss2 = None
    if weight_ornn_next is not None and \
       weight_ornn_next != 0 and fpro_size != 0:
       ploss2 = create_ornn_next(output, nnet_input, \
                             output_dim, lstm_input_dim, \
                             batch_size)
       ploss2 = weight_ornn_next * ploss2

    output = tf.reshape(output, [-1, output_dim])

    y_dist = None
    if num_experts is not None and num_experts!=0:
        y, y_dist = create_moe(output, \
                      output_dim, \
                      num_targets, \
                      num_experts
                     ) 
        y_dist  = weight_orth * y_dist
    else:
        # Feed-forward for the last layer
        stddev = 1.0 / math.sqrt(float(output_dim))
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

    reg_loss = ploss

    if y_dist is not None:
        if reg_loss is not None:
            reg_loss += y_dist
        else:
            reg_loss = y_dist

    if fp_loss is not None:
        if reg_loss is not None:
            reg_loss += fp_loss
        else:
            reg_loss = fp_loss

    if ploss2 is not None:
        if reg_loss is not None:
            reg_loss += ploss2
        else:
            reg_loss = ploss2

    return logits, reg_loss



