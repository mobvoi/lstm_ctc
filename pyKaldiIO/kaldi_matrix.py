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

import logging
import numpy
import struct
from enum import Enum
from io_funcs import BasicType
from io_funcs import ExpectToken
from io_funcs import InitKaldiOutputStream
from io_funcs import LogError
from io_funcs import ReadBasicType
from io_funcs import ReadFloat
from io_funcs import ReadInt32
from io_funcs import ReadToken
from io_funcs import WriteBasicType

class GlobalHeader(object):
    def __init__(self):
        self.format = None  # int32
        self.min_value = None  # float
        self.range = None  # float
        self.num_rows = None  # int32
        self.num_cols = None  # int32


class PerColHeader(object):
    def __init__(self):
        self.percentile_0 = None  # uint16
        self.percentile_25 = None  # uint16
        self.percentile_75 = None  # uint16
        self.percentile_100 = None  # uint16


def CharToFloat(p0, p25, p75, p100, value):
    if value <= 64:
        return p0 + (p25 - p0) * value * (1/64.0)
    elif value <= 192:
        return p25 + (p75 - p25) * (value - 64) * (1/128.0)
    else:
        return p75 + (p100 - p75) * (value - 192) * (1/63.0)


def Uint16ToFloat(global_header, value):
  # the constant 1.52590218966964e-05 is 1/65535.
  return global_header.min_value + \
      global_header.range * 1.52590218966964e-05 * value


class CompressedMatrix(object):
    def __init__(self):
        self.global_header = None
        self.percol_header = None
        self.data = None  # a 1-D array [ [col1] [col2] [col3] ... ]
                          # mat(i, j) = data[j * rows + i]

    def Read(self, stream, binary):
        if binary:
            self.global_header = GlobalHeader()
            token = ReadToken(stream, binary, False)
            stream.Read(1)  # comsume the space
            if token == 'CM':
                self.global_header.format = 1
            elif token == 'CM2':
                self.global_header.format = 2
            else:
                LogError('Unexpected token \"%s\", expecting CM or CM2.'
                         % token)
            data = struct.unpack('f'*2, stream.Read(4*2))
            self.global_header.min_value = data[0]
            self.global_header.range = data[1]
            data = struct.unpack('i'*2, stream.Read(4*2))
            self.global_header.num_rows = data[0]
            self.global_header.num_cols = data[1]
            rows = self.global_header.num_rows
            cols = self.global_header.num_cols
            if self.global_header.format == 1:  # num_rows > 8, in CM1 format.
                self.percol_header = []
                data = numpy.frombuffer(bytearray(stream.Read(2*4*cols)),
                                        numpy.uint16).reshape(cols, 4)
                for c in xrange(cols):
                    percol_header = PerColHeader()
                    percol_header.percentile_0 = data[c, 0]
                    percol_header.percentile_25 = data[c, 1]
                    percol_header.percentile_75 = data[c, 2]
                    percol_header.percentile_100 = data[c, 3]
                    self.percol_header.append(percol_header)
                self.data = numpy.frombuffer(bytearray(stream.Read(rows*cols)),
                                             numpy.uint8).reshape(cols, rows).transpose()
            elif self.global_header.format == 2:  # num_rows <= 8, in CM2 format.
                self.data = numpy.frombuffer(bytearray(stream.Read(2*rows*cols)),
                                             numpy.uint16).reshape(rows, cols)
            else:
                LogError('Unrecognized format = %s' % self.global_header.format)
        else:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'copy-feats\' in kaldi '
                     'to convert it to binary.')

    def GetNumpyMatrix(self):
        rows = self.global_header.num_rows
        cols = self.global_header.num_cols
        mat = numpy.empty([rows, cols], dtype=float, order='C')
        if self.global_header.format == 1:
            for c in xrange(cols):
                p0 = Uint16ToFloat(self.global_header,
                                   self.percol_header[c].percentile_0)
                p25 = Uint16ToFloat(self.global_header,
                                    self.percol_header[c].percentile_25)
                p75 = Uint16ToFloat(self.global_header,
                                    self.percol_header[c].percentile_75)
                p100 = Uint16ToFloat(self.global_header,
                                     self.percol_header[c].percentile_100)
                for r in xrange(rows):
                    mat[r, c] = CharToFloat(p0, p25, p75, p100, self.data[r, c])
        elif self.global_header.format == 2:
            for r in xrange(rows):
                for c in xrange(cols):
                    mat[r, c] = Uint16ToFloat(self.global_header, self.data[r, c])
        else:
            LogError('Unrecognized format = %s' % self.global_header.format)
        return mat


class FloatMatrix(object):
    """A wrapper of numpy matrix for I/O in Kaldi format.
    """
    def __init__(self):
        self.value = None

    def Read(self, stream, binary):
        """Read a matrix from the given stream.

        Args:
            stream: An opened KaldiInputStream.
            binary: If the input stream is in binary.

        Returns:
            An boolean variable indicating if the operation is successful.
        """
        # TODO(cfyeh): implement double matrix.
        if binary:
            peekval = stream.Peek(1)
            if peekval == 'C':
                cmat = CompressedMatrix()
                cmat.Read(stream, binary)
                self.value = cmat.GetNumpyMatrix()
                return True
            elif peekval == 'D':
                LogError('Double matrix not implemented yet.')
            elif peekval == 'F':
                expect_token = 'FM'
                token = ReadToken(stream, binary)
                if token != expect_token:
                    LogError('Expect token \"%s\", got \"%s\"'
                             % (expect_token, token))
                rows = ReadInt32(stream, binary)
                cols = ReadInt32(stream, binary)
                data = stream.Read(4*(rows*cols))
                self.value = numpy.frombuffer(bytearray(data), numpy.float32).reshape(rows, cols)
            else:
                LogError('Unrecognized flag \"%s\"' % peekval)
        else:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'copy-feats\' in kaldi '
                     'to convert it to binary.')
        return True

    def Value(self):
        """Return the stored numpy matrix.
        """
        return self.value

    def Clear(self):
        """Clear the object.
        """
        del self.value
        self.value = None
        return True


class FloatVector(object):
    """A wrapper of numpy vector for I/O in Kaldi format.
    """
    def __init__(self):
        self.value = None

    def Read(self, stream, binary):
        """Read a vector from the given stream.

        Args:
            stream: An opened KaldiInputStream.
            binary: If the input stream is in binary.

        Returns:
            An boolean variable indicating if the operation is successful.
        """
        # TODO(cfyeh): implement double vector.
        if binary:
            peekval = stream.Peek(1)
            if peekval == 'C':
                cmat = CompressedMatrix()
                cmat.Read(stream, binary)
                self.value = cmat.GetNumpyMatrix()
                return True
            elif peekval == 'D':
                LogError('Double vector not implemented yet.')
            elif peekval == 'F':
                expect_token = 'FV'
                token = ReadToken(stream, binary)
                if token != expect_token:
                    LogError('Expect token \"%s\", got \"%s\"'
                             % (expect_token, token))
                size = ReadInt32(stream, binary)
                data = stream.Read(4*size)
                self.value = numpy.frombuffer(bytearray(data), numpy.float32)
            else:
                LogError('Unrecognized flag \"%s\"' % peekval)
        else:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'copy-feats\' in kaldi '
                     'to convert it to binary.')
        return True

    def Value(self):
        """Return the stored numpy vector.
        """
        return self.value

    def Clear(self):
        """Clear the object.
        """
        del self.value
        self.value = None
        return True


class SparseVector(object):
    def __init__(self):
        self.dim = 0
        self.pair = []  # (row-index, value)

    def Read(self, stream, binary):
        if binary:
            ExpectToken(stream, binary, "SV")
            self.dim = ReadInt32(stream, binary)
            if self.dim < 0:
               LogError('Unexpected dimension \"%s\"' % self.dim)
            num_elems = ReadInt32(stream, binary)
            if num_elems < 0 or num_elems > self.dim:
               LogError('Unexpected number of elements %s vs. %s'
                        % (num_elems, self.dim))
            self.pair = []
            for i in xrange(num_elems):
                idx = ReadInt32(stream, binary)
                val = ReadFloat(stream, binary)
                self.pair.append([idx, val])
        else:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'copy-feats\' in kaldi '
                     'to convert it to binary.')


def WriteFloatMatrixToStream(stream, binary, value):
    InitKaldiOutputStream(stream, binary)
    if binary:
        my_token = 'FM'
        stream.Write('%s ' % my_token)
        WriteBasicType(stream, binary, BasicType.cint32, value.shape[0])
        WriteBasicType(stream, binary, BasicType.cint32, value.shape[1])
        for row in value.tolist():
            stream.Write(struct.pack('%sf' % len(row), *row))
    else:
        if not value.shape[0] or not value.shape[1]:
            stream.Write(' []\n')
        else:
            stream.Write(' [')
            for r in xrange(value.shape[0]):
                stream.Write('\n  ')
                for c in xrange(value.shape[1]):
                    stream.Write('%f ' % value[r][c])
            stream.Write(']\n')
    return True


def WriteFloatVectorToStream(stream, binary, value):
    InitKaldiOutputStream(stream, binary)
    if binary:
        my_token = 'FV'
        stream.Write('%s ' % my_token)
        WriteBasicType(stream, binary, BasicType.cint32, value.shape[0])
        val = value.tolist()
        stream.Write(struct.pack('%sf' % len(val), *val))
    else:
        if not value.shape[0]:
            stream.Write(' []\n')
        else:
            stream.Write(' [ ')
            for i in xrange(value.shape[0]):
                stream.Write('%f ' % value[i])
            stream.Write(']\n')
    return True


class SparseMatrix(object):
    def __init__(self):
        self.rows = []

    def Read(self, stream, binary):
        if binary:
            ExpectToken(stream, binary, 'SM')
            num_rows = ReadInt32(stream, binary)
            if num_rows < 0 or num_rows > 10000000:
                LogError('Unexpected number of rows %s' % num_rows)
            self.rows = []
            for r in xrange(num_rows):
                self.rows.append(SparseVector())
                self.rows[r].Read(stream, binary)
        else:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'copy-feats\' in kaldi '
                     'to convert it to binary.')
