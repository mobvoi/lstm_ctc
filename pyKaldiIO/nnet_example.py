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

from io_funcs import ExpectToken
from io_funcs import LogError
from io_funcs import LogWarning
from io_funcs import ReadToken
from io_funcs import ReadInt32
from kaldi_matrix import FloatMatrix
from kaldi_matrix import SparseMatrix
from nnet_common import ReadIndexVector

class NnetIo(object):
    def __init__(self, name = None, t_begin = None, feats = None):
        self.name = None  # the name of the input in the neural net; in simple
                          # setups it will just be "input".
        self.indexes = None  # "indexes" stream a vector the same length as
                             # features.NumRows(), explaining the meaning of each
                             # row of the "features" matrix.
                             # Note: the "n" values in the indexes will always be
                             # zero in individual examples, but in general nonzero
                             # after we aggregate the examples into the minibatch
                             # level.
        self.features = None  # The features or labels.

    def Read(self, stream, binary):
        ExpectToken(stream, binary, '<NnetIo>')
        self.name = ReadToken(stream, binary)
        self.indexes = ReadIndexVector(stream, binary)
        c = stream.Peek(1)
        if binary:
            # TODO(cfyeh): implement double matrix.
            if c == 'F' or c == 'C':
                self.features = FloatMatrix()
            elif c == 'S':
                self.features = SparseMatrix()
            else:
                LogError('Unrecognized identifier \"%s\"' % c)
        else:
            # TODO(cfyeh): implement text mode.
            LogError('Text mode not implemented yet.')
        self.features.Read(stream, binary)
        ExpectToken(stream, binary, '</NnetIo>')
        return True


class NnetExample(object):
    def __init__(self):
        self.io = None

    def Read(self, stream, binary):
        ExpectToken(stream, binary, "<Nnet3Eg>")
        ExpectToken(stream, binary, "<NumIo>")
        size = ReadInt32(stream, binary)
        if size <= 0 or  size > 1000000:
            LogError('Invalid size %d' % size)
        self.io = []
        for i in xrange(size):
           self.io.append(NnetIo())
        for io in self.io:
            io.Read(stream, binary)
        ExpectToken(stream, binary, "</Nnet3Eg>")
        return True

    def GetFeature(self, name = 'input'):
        for io in self.io:
            if io.name == 'input':
                return io.features.value
        LogWarning('No feature found for specified name \"%s\"' % name)
        return None

    def GetLabel(self, name = 'output'):
        for io in self.io:
            if io.name == 'output':
                res = []
                for row in io.features.rows:
                    for pair in row.pair:
                        res.append(pair[0])
                return res
        LogWarning('No label found for specified name \"%s\"' % name)
        return None
