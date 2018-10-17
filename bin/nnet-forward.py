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

import argparse
import nnet
import numpy
import os
import pyKaldiIO
import sys
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)



def main(_):
    nnet_output_writer = \
        pyKaldiIO.BaseFloatMatrixWriter(args.nnet_output)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Alway use minimum memory.
    sess = tf.Session(config = config)

    nnet_config = nnet.parse_config(args.nnet_config)
    left_context = nnet_config.get('left_context')
    right_context = nnet_config.get('right_context')
    subsample = nnet_config.get('subsample')
    nnet_config['is_training'] = False
    if args.apply_log:
        args.apply_softmax=True

    class_prior = None if args.class_prior is None else \
                  nnet.get_class_prior(args.class_prior)

    filename, tfrecord, _ = \
        nnet.dataset_from_tfrecords(
            tfrecords_scp=args.tfrecords_scp,
            left_context=left_context,
            right_context=right_context,
            subsample=subsample,
            shuffle=False,
        )

    pipeline_initializer, pipeline = \
        nnet.create_pipeline_sequential(
            filename=filename,
            tfrecord=tfrecord,
        )

    graph = \
        nnet.create_graph_for_inference(
            pipeline=pipeline,
            nnet_config=nnet_config,
            smooth_factor=args.smooth_factor
        )

    sess.run(pipeline_initializer)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, args.nnet_in)

    nodes = { 'filename' : graph['filename'] }
    nodes['nnet_output'] = graph['nnet_output'] \
                           if args.apply_softmax else graph['logits']

    try:
        processed = 0
        while True:
            values = sess.run(nodes)
            filename = values['filename']
            nnet_output = values['nnet_output']
            if args.apply_log:
                nnet_output = numpy.log(nnet_output)

            if class_prior is not None:
                nnet_output = nnet_output - class_prior

            key = os.path.basename(filename)
            key, _ = os.path.splitext(key)

            nnet_output_writer.Write(key, nnet_output)

            processed += 1
            if args.report_interval and \
               processed % args.report_interval == 0:
                log = 'processed = %d' % (processed)
                tf.logging.info(log)
    
    except tf.errors.OutOfRangeError:
        log = 'done'
        tf.logging.info(log)

    except KeyboardInterrupt:
        log = 'interrupted by user'
        tf.logging.fatal(log)
        sys.exit(1)

    nnet_output_writer.Close()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # positional args.
    parser.add_argument('tfrecords_scp', metavar = '<tfrecords-scp>',
                        type = str, help = 'tfrecords-scp.')
    parser.add_argument('nnet_config', metavar = '<nnet-config>',
                        type = str, help = 'nnet-config.')
    parser.add_argument('nnet_in', metavar = '<nnet-in>',
                        type = str, help = 'nnet-in.')
    parser.add_argument('nnet_output', metavar = '<nnet-output-wspecifier>',
                        type = str, help='wspecifier for nnet-output.')
        # switches
    parser.add_argument('--apply-softmax', metavar = 'apply-softmax',
                        help='whether to apply softmax.',
                        type = str2bool, default = 'true')
    parser.add_argument('--apply-log', metavar = 'apply-log',
                        help='whether to apply log on top of softmax',
                        type = str2bool, default = 'true')
    parser.add_argument('--report-interval', metavar = 'report-interval',
                        type = int, help='progress report interval.', default = 100)
    parser.add_argument('--class-prior', metavar = 'class-prior',
                        type = str, help='class prior to scale the softmax output',
                        default = None)
    parser.add_argument('--smooth-factor', metavar ='smooth factor',
                        type = float, help='smooth factor for softmax', 
                        default = 1.0)


    args = parser.parse_args()

    log = ' '.join(sys.argv)
    tf.logging.info(log)

    tf.app.run(main=main, argv=[sys.argv[0]])
