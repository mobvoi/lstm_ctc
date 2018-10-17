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
import pyKaldiIO
import sys
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    nnet_input_reader = \
        pyKaldiIO.SequentialBaseFloatMatrixReader(args.nnet_input)

    nnet_target_reader = \
        pyKaldiIO.RandomAccessInt32VectorReader(args.nnet_target) \
        if args.nnet_target is not None else None

    with open(args.tfrecords_scp, 'w') as scp:
        processed = 0

        while not nnet_input_reader.Done():
            key = nnet_input_reader.Key()
            skip = False

            if nnet_target_reader is not None and \
               not nnet_target_reader.HasKey(key):
                log = 'missing nnet targets for \"%s\" in %s' % \
                      (key, args.nnet_target)
                tf.logging.info(log)
                skip = True

            if skip:
                nnet_input_reader.Next()
                continue

            nnet_target = nnet_target_reader.Value(key) \
                          if nnet_target_reader is not None else None

            if nnet_target is not None and \
               nnet_target.shape[0] == 0:
                log = 'length of nnet targets for \"%s\" is 0 in %s' % \
                      (key, args.nnet_target)
                tf.logging.info(log)
                skip = True
            
            if skip:
                nnet_input_reader.Next()
                continue

            filename = args.tfrecords_dir + '/' + key + '.tfrecords'

            nnet_input = nnet_input_reader.Value()
            # Training with alignments, need to check if the lengths of
            # (feature, label) are consistent.
            if args.check_length and nnet_target is not None:
                if nnet_input.shape[0] != nnet_target.shape[0]:
                    log = 'mismatched sizes between nnet_input and nnet_target:' + \
                          '%s vs. %s' % (str(nnet_input.shape), str(nnet_target.shape))
                    tf.logging.fatal(log)
                    sys.exit(1)

            if nnet_target is not None and \
               nnet_target.shape[0] >= nnet_input.shape[0]:
                log = 'nnet_input.shape = %s nnet_target.shape = %s for \"%s\" in %s' % \
                      (str(nnet_input.shape), str(nnet_target.shape), key, args.nnet_target)
                tf.logging.info(log)
                skip = True

            if nnet_target is not None and \
               nnet_target.shape[0] <= args.target_length_cutoff:
                log = 'nnet_target shape = %s for \"%s \" in %s is too short' % \
                    (str(nnet_target.shape), key, args.nnet_target)
                tf.logging.info(log)
                skip = True

            
            if skip:
                nnet_input_reader.Next()
                continue

            #tf.logging.info('key = %s nnet_target.shape = %s' % (key, str(nnet_target.shape)))
            nnet.write_tfrecord(
                filename=filename,
                nnet_input=nnet_input,
                nnet_target=nnet_target,
            )

            num_rows = nnet_input.shape[0]
            num_cols = nnet_input.shape[1]
            has_label = 1 if args.nnet_target else 0
            scp.write('%s %d %d %d %s\n' %
                      (key, num_rows, num_cols,
                       has_label, filename))

            processed += 1
            if args.report_interval and \
               processed % args.report_interval == 0:
                log = 'processed = %d' % (processed)
                tf.logging.info(log)

            nnet_input_reader.Next()

    nnet_input_reader.Close()
    if nnet_target_reader is not None:
        nnet_target_reader.Close()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # positional args.
    parser.add_argument('nnet_input', metavar = '<nnet-input-rspecifier>',
                        type = str, help = 'nnet-input rspecifier.')
    parser.add_argument('tfrecords_dir', metavar = '<tfrecords-dir>',
                        type = str, help = 'directory for storing tfrecords.')
    parser.add_argument('tfrecords_scp', metavar = '<tfrecords.scp>',
                        type = str, help = 'tfrecords.scp for converted tfrecords.')

    # switches
    parser.add_argument('--nnet-target', metavar = '<nnet-target-rspecifiers>',
                        help = 'nnet-target rspecifiers.',
                        type = str, default = None)
    parser.add_argument('--target-length-cutoff', metavar = 'target length cut off',
                        help = 'filter short length audio',
                        type = int, default=1)
    parser.add_argument('--check-length', metavar = 'check-length',
                        help = 'whether to check the consistensy of lengths (should be false for CTC).',
                        type = str2bool, default = 'true')
    parser.add_argument('--report-interval', metavar = 'report-interval',
                        help='progress report interval.',
                        type = int, default = 100)

    args = parser.parse_args()

    tf.app.run(main=main, argv=[sys.argv[0]])
