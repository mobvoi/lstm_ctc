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
    output_writer = \
        pyKaldiIO.Int32VectorWriter(args.output)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Alway use minimum memory.
    sess = tf.Session(config = config)

    nnet_config = nnet.parse_config(args.nnet_config)
    nnet_config['is_training'] = False

    filename, tfrecord, _ = \
        nnet.dataset_from_tfrecords(
            tfrecords_scp=args.tfrecords_scp,
            left_context=nnet_config.get('left_context'),
            right_context=nnet_config.get('right_context'),
            shuffle=False,
        )

    pipeline_initializer, pipeline = \
        nnet.create_pipeline_sequential(
            filename=filename,
            tfrecord=tfrecord,
        )

    graph = nnet.create_graph_for_decoding(
                pipeline=pipeline,
                nnet_config=nnet_config,
            )

    sess.run(pipeline_initializer)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, args.nnet_in)

    nodes = { 'filename' : graph['filename'],
              'decoded' :  graph['decoded'] }

    try:
        processed = 0
        while True:
            values = sess.run(nodes)
            filename = values['filename']
            decoded = values['decoded']

            key = os.path.basename(filename)
            key, _ = os.path.splitext(key)

            output_writer.Write(key, decoded)

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

    output_writer.Close()


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
    parser.add_argument('output', metavar = '<output-wspecifier>',
                        type = str, help='wspecifier for output.')

    # switches
    parser.add_argument('--report-interval', metavar = 'report-interval',
                        type = int, help='progress report interval.', default = 100)

    args = parser.parse_args()

    log = ' '.join(sys.argv)
    tf.logging.info(log)

    tf.app.run(main=main, argv=[sys.argv[0]])
