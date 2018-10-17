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
import math
import nnet
import sys
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Alway use minimum memory.
        sess = tf.Session(config = config)
    
        nnet_config = nnet.parse_config(args.nnet_config)
        nnet_config['is_training'] = False

        nnet_type = nnet_config.get('nnet_type')
        left_context = nnet_config.get('left_context')
        right_context = nnet_config.get('right_context')
        subsample = nnet_config.get('subsample')

        filename, tfrecord, input_dim = \
            nnet.dataset_from_tfrecords(
                tfrecords_scp=args.tfrecords_scp,
                left_context=left_context,
                right_context=right_context,
                subsample=subsample,
                shuffle=False,
            )

        if args.objective == 'ctc':
            if nnet_type == 'blstm' or nnet_type == 'cudnnlstm' or nnet_type == 'lstm':
                pipeline_initializer, pipeline = \
                    nnet.create_pipeline_sequence_batch(
                        dataset=tfrecord,
                        input_dim=input_dim,
                        batch_size=args.batch_size,
                        batch_threads=args.batch_threads,
                        num_epochs=1,
                    )
                graph = \
                    nnet.create_graph_for_validation_ctc(
                        pipeline=pipeline,
                        nnet_config=nnet_config,
                    )
            else:
                log = 'unsupported nnet_type: %s' % nnet_type
                tf.logging.fatal(log)
                sys.exit(1)
        else:
            log = 'unsupported objective: %s' % args.objective
            tf.logging.fatal(log)
            sys.exit(1)

        sess.run(pipeline_initializer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
    
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, args.nnet_in)
    
        success = \
            nnet.validate(
                sess=sess,
                graph=graph,
                evaluate=args.evaluate,
                report_interval=args.report_interval
            )

    except KeyboardInterrupt:
        log = 'interrupted by user'
        tf.logging.fatal(log)
        sys.exit(1)


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
    parser.add_argument('tfrecords_scp', metavar = '<tfrecords.scp>',
                        type = str, help = 'tfrecords.scp.')
    parser.add_argument('nnet_config', metavar = '<nnet-config>',
                        type = str, help = 'nnet-config.')
    parser.add_argument('nnet_in', metavar = '<nnet-in>',
                        type = str, help = 'nnet-in.')

    # switches
    parser.add_argument('--objective', metavar = 'objective',
                        help='objective function.',
                        type = str, default = 'xent')
    parser.add_argument('--evaluate', metavar = 'evaluate',
                        help='whether to evaluate the model in addition to loss.',
                        type = str2bool, default = 'false')
    parser.add_argument('--batch-size', metavar = 'batch-size',
                        type = int, help='batch size.', default = 256)
    parser.add_argument('--batch-threads', metavar = 'batch-threads',
                        type = int, help='batch threads.', default = 8)
    parser.add_argument('--num-parallel-calls', metavar = 'num-parallel-calls',
                        type = int, help='num-parallel-calls.', default = 32)
    parser.add_argument('--report-interval', metavar = 'report-interval',
                        type = int, help='progress report interval.', default = 100)

    args = parser.parse_args()

    log = ' '.join(sys.argv)
    tf.logging.info(log)

    tf.app.run(main=main, argv=[sys.argv[0]])
