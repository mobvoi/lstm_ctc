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
from io_funcs import InitKaldiOutputStream
from io_funcs import LogError
from io_funcs import ReadBasicType
from io_funcs import ReadFloat
from io_funcs import ReadInt32
from io_funcs import ReadToken
from io_funcs import WriteBasicType
from kaldi_matrix import FloatMatrix
from kaldi_matrix import FloatVector
from kaldi_matrix import WriteFloatMatrixToStream
from kaldi_matrix import WriteFloatVectorToStream
from nnet_example import NnetExample


class HolderType(Enum):
    """Enumerations for Kaldi holder types.
    """
    kNoHolder = 0
    kFloatMatrixHolder = 1
    kFloatVectorHolder = 2
    kPosteriorHolder = 3
    kInt32VectorHolder = 4
    kNnetExampleHolder = 5


class FloatMatrixHolder(object):
    """A wrapper of numpy matrix for I/O in Kaldi format.
    """
    def __init__(self):
        self.inst = FloatMatrix()

    def Read(self, stream, binary):
        return self.inst.Read(stream, binary)

    def Value(self):
        """Return the stored numpy matrix.
        """
        return self.inst.Value()

    def Clear(self):
        return self.inst.Clear()

    def IsReadInBinary(self):
        return True


class FloatVectorHolder(object):
    """A wrapper of numpy vector for I/O in Kaldi format.
    """
    def __init__(self):
        self.inst = FloatVector()

    def Read(self, stream, binary):
        return self.inst.Read(stream, binary)

    def Value(self):
        """Return the stored numpy matrix.
        """
        return self.inst.Value()

    def Clear(self):
        return self.inst.Clear()

    def IsReadInBinary(self):
        return True


class PosteriorHolder(object):
    """A wrapper to store posteriorgrams in frames and for I/O in Kaldi format.
    e.g. std::vector<std::vector<std::pair<int32, BaseFloat> > > in Kaldi.
    """
    def __init__(self):
        self.value = None

    def Read(self, stream, binary):
        """Read posteriorgrams from the given stream.

        Args:
            stream: An opened KaldiInputStream.
            binary: If the input stream is in binary.

        Returns:
            An boolean variable indicating if the operation is successful.
        """
        if binary:
            sz = ReadInt32(stream, binary)
            if sz < 0 or sz > 10000000:
                LogError('Got negative or improbably large size \"%d\"' % sz)
            self.value = []
            for i in xrange(sz):
                self.value.append([])
                sz2 = ReadInt32(stream, binary)
                if sz2 < 0:
                    LogError('Got negative size \"%d\"' % sz2)
                for j in xrange(sz2):
                    lab = ReadInt32(stream, binary)
                    val = ReadFloat(stream, binary)
                    self.value[i].append((lab, val))
        else:
            line = stream.Readline()
            tokens = line.rstrip().split()
            i = 0
            self.value = []
            while True:
                if i == len(tokens):
                    break
                if tokens[i] != '[':
                    LogError('Expecting \"[\", got \"%s\" instead.'
                             % tokens[i])
                self.value.append([])
                while True:
                    i += 1
                    if tokens[i] == ']':
                        break;
                    lab = int(tokens[i])
                    i += 1
                    val = float(tokens[i])
                    self.value[-1].append((lab, val))
                i += 1
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

    def IsReadInBinary(self):
        return True


class BasicVectorHolder(object):
    """A wrapper to store basic type vectors and for I/O in Kaldi format.
    e.g. std::vector<BaseType> in C/C++, BasicType can be int32/float/...
    """
    def __init__(self, basic_type):
        self.type = basic_type
        self.value = None

    def Read(self, stream, binary):
        """Read a BasicType vector from the given stream.

        Args:
            stream: An opened KaldiInputStream.
            binary: If the input stream is in binary.

        Returns:
            An boolean variable indicating if the operation is successful.
        """
        if binary:
            sz = ReadInt32(stream, binary)
            self.value = numpy.empty(sz, numpy.int32)
            for i in xrange(sz):
                val = ReadBasicType(stream, binary, self.type)
                self.value[i] = val
        else:
            line = stream.Readline()
            if self.type == BasicType.cint32:
                self.value = numpy.asarray([ int(x) for x in line.rstrip().split() ], numpy.int32)
            else:
                LogError('BasicType \"%s\" not implemented yet.' % self.type)
        return True

    def Value(self):
        """Return the stored vector.
        """
        return self.value

    def Clear(self):
        """Clear the object.
        """
        del self.value
        self.value = None
        return True

    def IsReadInBinary(self):
        return True


class NnetExampleHolder(object):
    """A wrapper of nnet3 egs for I/O in Kaldi format.
    """
    def __init__(self):
        self.inst = NnetExample()

    def Read(self, stream, binary):
        return self.inst.Read(stream, binary)

    def Value(self):
        return self.inst

    def Clear(self):
        del self.inst
        self.inst = NnetExample()
        return True

    def IsReadInBinary(self):
        return True


def NewHolderByType(holder_type):
    if holder_type == HolderType.kNoHolder:
        LogError('No holder type is specified.')
    elif holder_type == HolderType.kFloatMatrixHolder:
        return FloatMatrixHolder()
    elif holder_type == HolderType.kFloatVectorHolder:
        return FloatVectorHolder()
    elif holder_type == HolderType.kPosteriorHolder:
        return PosteriorHolder()
    elif holder_type == HolderType.kInt32VectorHolder:
        return BasicVectorHolder(BasicType.cint32)
    elif holder_type == HolderType.kNnetExampleHolder:
        return NnetExampleHolder()
    else:
        LogError('Unrecognized holder type \"%s\"' % holder_type)


def WriteInt32VectorToStream(stream, binary, value):
    InitKaldiOutputStream(stream, binary)
    if binary:
        LogError('binary version not implemented yet.')
        #WriteBasicType(stream, binary, BasicType.cint32, value.shape[0])
        #val = value.tolist()
        #stream.Write(struct.pack('%sd' % len(val), *val))
    else:
        for i in xrange(value.shape[0]):
            WriteBasicType(stream, binary, BasicType.cint32, value[i])
        stream.Write('\n')
    return True


def WriteHolderValueToStream(stream, holder_type, binary, value):
    if holder_type == HolderType.kNoHolder:
        LogError('No holder type is specified.')
    elif holder_type == HolderType.kFloatMatrixHolder:
        WriteFloatMatrixToStream(stream, binary, value)
    elif holder_type == HolderType.kFloatVectorHolder:
        WriteFloatVectorToStream(stream, binary, value)
    elif holder_type == HolderType.kPosteriorHolder:
        WritePosteriorToStream(stream, binary, value)
    elif holder_type == HolderType.kInt32VectorHolder:
        WriteInt32VectorToStream(stream, binary, value)
    else:
        LogError('Unrecognized holder type \"%s\"' % holder_type)
    return True
