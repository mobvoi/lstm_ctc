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

import numpy
import random
from io_funcs import BasicType
from io_funcs import LogError

class NnetDataRandomizerOptions(object):
    def __init__(self, rsize = None, rseed = None, msize = None):
        self.randomizer_size = 32768
        self.randomizer_seed = 777
        self.minibatch_size = 256
        if rsize is not None and rseed is not None and msize is not None:
            self.randomizer_size = rsize
            self.randomizer_seed = rseed
            self.minibatch_size = msize

class RandomizerMask(object):
    def __init__(self, conf = None):
        if conf is not None:
            random.seed(conf.randomizer_seed)

    def Generate(self, mask_size):
        res = [ i for i in xrange(mask_size) ]
        random.shuffle(res)
        return res


class MatrixRandomizer(object):
    def __init__(self, conf = None):
        self.data_begin = 0
        self.data_end = 0
        self.conf = NnetDataRandomizerOptions()
        self.data = None
        self.data_aux = None
        self.minibatch = None
        if conf is not None:
            self.conf = conf

    def AddData(self, mat):
        m_rows = mat.shape[0]
        m_cols = mat.shape[1]
        if self.data is None:
            rows = self.conf.randomizer_size
            cols = m_cols
            self.data = numpy.empty([rows, cols], dtype=float, order='C')
        if self.data_begin > 0:
            if self.data_begin > self.data_end:  # Sanity check.
                LogError('Sanity check failed: self.data_begin \"%d\" > '
                         'self.data_end \"%d\"'
                         % (self.data_begin, self.data_end))
            left_over = self.data_end - self.data_begin
            if left_over >= self.data_begin:  # No overlap.
                LogError('Unexpected data overlap: left_over \"%d\" >= '
                         'self.data_begin \"%d\"'
                         % (left_over, self.data_begin))
            if left_over > 0:
                self.data[0:left_over, :] = \
                    self.data[self.data_begin:self.data_begin+left_over, :]
            self.data_begin = 0
            self.data_end = left_over
        if self.data.shape[0] < self.data_end + m_rows:
            self.data_aux = numpy.empty_like(self.data)
            self.data_aux[:] = self.data
            rows = self.data_end + m_rows + 1000 # Add extra 1000 rows, so we
                                                 # don't reallocate soon.
            cols = self.data.shape[1]
            self.data = numpy.empty([rows, cols], dtype=float, order='C')
            self.data[0:self.data_aux.shape[0], :] = self.data_aux
        self.data[self.data_end:self.data_end + m_rows, :] = mat
        self.data_end += m_rows

    def IsFull(self):
        return self.data_begin == 0 and \
               self.data_end > self.conf.randomizer_size

    def NumFrames(self):
        return self.data_end

    def Randomize(self, mask):
        if self.data_begin != 0:
            LogError('Unexpected self.data_begin %d vs. 0' % self.data_begin)
        if self.data_end <= 0:
            LogError('Unexpected self.data_end %d' % self.data_begin)
        if self.data_end != len(mask):
            LogError('Mismatched self.data_end \"%d\" vs. mask size \"%d\"'
                     % (self.data_end, len(mask)))
        self.data = self.data[mask]

    def Done(self):
        return (self.data_end - self.data_begin) < \
               self.conf.minibatch_size

    def Next(self):
        self.data_begin += self.conf.minibatch_size

    def Value(self):
        if self.data_end - self.data_begin < self.conf.minibatch_size:
            LogError('Insufficient data for mini batch: %d vs. %d'
                     % (self.data_end - self.data_begin, self.conf.minibatch_size))
        self.minibatch = \
            self.data[self.data_begin:self.data_begin+self.conf.minibatch_size, :]
        return self.minibatch


class BasicVectorRandomizer(object):
    """Randomizer for basic type vectors such as int32/float/...
    """
    def __init__(self, basic_type, conf = None):
        self.data_begin = 0
        self.data_end = 0
        self.conf = NnetDataRandomizerOptions()
        self.data = None
        self.data_aux = None
        self.minibatch = None
        if conf is not None:
            self.conf = conf
        if basic_type == BasicType.cint32:
            self.data = numpy.empty(self.conf.randomizer_size, numpy.int32)
        elif basic_type == BasicType.cfloat:
            self.data = numpy.empty(self.conf.randomizer_size, numpy.float32)
        else:
            LogError('BasicType \"%s\" not implemented yet.' % basic_type)

    def AddData(self, data):
        if self.data_begin > 0:
            if self.data_begin > self.data_end:  # Sanity check.
                LogError('Sanity check failed: self.data_begin \"%d\" > '
                         'self.data_end \"%d\"'
                         % (self.data_begin, self.data_end))
            left_over = self.data_end - self.data_begin
            if left_over >= self.data_begin:  # No overlap.
                LogError('Unexpected data overlap: left_over \"%d\" >= '
                         'self.data_begin \"%d\"'
                         % (left_over, self.data_begin))
            if left_over > 0:
                self.data[0:left_over] = \
                    self.data[self.data_begin:self.data_begin+left_over]
            self.data_begin = 0
            self.data_end = left_over
        if len(self.data) < self.data_end + len(data):
            self.data_aux = numpy.empty_like(self.data)
            self.data_aux[:] = self.data
            size = self.data_end + data.shape[0] + 1000 # Add extra 1000 elements, so we
                                                        # don't reallocate soon.
            self.data = numpy.empty(size, self.data.dtype)
            self.data[0:self.data_aux.shape[0]] = self.data_aux
        self.data[self.data_end:self.data_end+data.shape[0]] = data
        self.data_end += data.shape[0]

    def IsFull(self):
        return self.data_begin == 0 and \
               self.data_end > self.conf.randomizer_size

    def NumFrames(self):
        return self.data_end

    def Randomize(self, mask):
        if self.data_begin != 0:
            LogError('Unexpected self.data_begin %d vs. 0' % self.data_begin)
        if self.data_end <= 0:
            LogError('Unexpected self.data_end %d' % self.data_begin)
        if self.data_end != len(mask):
            LogError('Mismatched self.data_end \"%d\" vs. mask size \"%d\"'
                     % (self.data_end, len(mask)))
        self.data = self.data[mask]

    def Done(self):
        return (self.data_end - self.data_begin) < \
               self.conf.minibatch_size

    def Next(self):
        self.data_begin += self.conf.minibatch_size

    def Value(self):
        if self.data_end - self.data_begin < self.conf.minibatch_size:
            LogError('Insufficient data for mini batch: %d vs. %d'
                     % (self.data_end - self.data_begin, self.conf.minibatch_size))
        self.minibatch = \
            self.data[self.data_begin:self.data_begin+self.conf.minibatch_size]
        return self.minibatch


class Int32VectorRandomizer(BasicVectorRandomizer):
    """A wrapper for BasicVectorRandomizer(BasicType.cint32).
    """
    def __init__(self, conf = None):
        super(Int32VectorRandomizer,
              self).__init__(BasicType.cint32, conf)


class FloatVectorRandomizer(BasicVectorRandomizer):
    """A wrapper for BasicVectorRandomizer(BasicType.cfloat).
    """
    def __init__(self, conf = None):
        super(FloatVectorRandomizer,
              self).__init__(BasicType.cfloat, conf)
