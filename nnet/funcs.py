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

import math
import numpy
import sys
import tensorflow as tf


def train(sess, graph, evaluate = False, report_interval = None):
    step = 0
    processed = 0
    loss = 0.0
    nodes = { 'size' : graph['size'],
              'train' : graph['train'],
              'summary' : graph['summary'],
              'loss' : graph['loss'],
              'eval_loss' : graph['eval_loss'],
              'dsm_loss' : graph['dsm_loss'],
              'sequence_length' : graph['sequence_length']
            }

    if evaluate:  # if evaluation is required, add additional nodes.
        acc = 0.0
        nodes['eval'] = graph['eval']

    try:
        while True:
            values = sess.run(nodes)
            batch_size = values['size']
            batch_loss = values['eval_loss']

            if evaluate:
                batch_eval = values['eval']

            if batch_size > 0:
                processed += batch_size
                batch_loss /= batch_size
                loss += (batch_loss - loss) * batch_size / processed
                if evaluate:
                    batch_eval /= batch_size
                    acc += (batch_eval - acc) * batch_size / processed

            step += 1
            if report_interval and step % report_interval == 0:
                log = 'step = %d, batch_size = %d, loss = %f' % \
                       (step, batch_size, loss)
                if evaluate:
                    log += ', eval = %f' % acc
                tf.logging.info(log)

            if math.isnan(loss):
                raise ValueError

    except tf.errors.OutOfRangeError:
        log = 'done'
        tf.logging.info(log)

    except KeyboardInterrupt:
        log = 'interrupted by user'
        tf.logging.fatal(log)
        sys.exit(1)

    except ValueError:
        log = 'tr_loss = %f' % loss
        tf.logging.info(log)
        log = 'nan loss detected'
        tf.logging.fatal(log)
        sys.exit(1)

    log = 'tr_loss = %f' % loss
    tf.logging.info(log)

    return True


def validate(sess, graph, evaluate = False, report_interval = None):
    step = 0
    processed = 0
    loss = 0.0
    nodes = {
        'size' : graph['size'],
        'loss' : graph['loss'],
        'eval_loss' : graph['eval_loss']
    }
    if evaluate:  # if evaluation is required, add additional nodes.
        acc = 0.0
        nodes['eval'] = graph['eval']

    try:
        while True:
            values = sess.run(nodes)
            batch_size = values['size']
            batch_loss = values['eval_loss']

            if evaluate:
                batch_eval = values['eval']

            if batch_size > 0:
                processed += batch_size
                batch_loss /= batch_size
                loss += (batch_loss - loss) * batch_size / processed
                if evaluate:
                    batch_eval /= batch_size
                    acc += (batch_eval - acc) * batch_size / processed

            step += 1
            if report_interval and step % report_interval == 0:
                log = 'step = %d, batch_size = %d, loss = %f' % \
                       (step, batch_size, loss)
                if evaluate:
                    log += ', eval = %f' % acc
                tf.logging.info(log)

            if math.isnan(loss):
                raise ValueError

    except tf.errors.OutOfRangeError:
        log = 'done'
        tf.logging.info(log)

    except KeyboardInterrupt:
        log = 'interrupted by user'
        tf.logging.fatal(log)
        sys.exit(1)

    except ValueError:
        log = 'cv_loss = %f' % loss
        tf.logging.info(log)
        log = 'nan loss detected'
        tf.logging.fatal(log)
        sys.exit(1)

    log = 'cv_loss = %f' % loss
    tf.logging.info(log)
    if evaluate:
        log = 'cv_eval = %f' % acc
        tf.logging.info(log)

    return True
