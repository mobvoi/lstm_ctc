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
from io_funcs import ExpectToken
from io_funcs import LogError
from io_funcs import ReadFloat
from io_funcs import ReadInt32
from io_funcs import ReadToken
from kaldi_io import Input
from kaldi_matrix import FloatMatrix
from kaldi_matrix import FloatVector


class Component(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim


class Sigmoid(Component):
    def __init__(self, input_dim, output_dim):
        super(Sigmoid, self).__init__(input_dim, output_dim)

    def Read(self, stream, binary):
        if not binary:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'nnet-copy\' in kaldi '
                     'to convert it to binary.')

    def Dump(self):
        return 'Sigmoid', []


class Softmax(Component):
    def __init__(self, input_dim, output_dim):
        super(Softmax, self).__init__(input_dim, output_dim)

    def Read(self, stream, binary):
        if not binary:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'nnet-copy\' in kaldi '
                     'to convert it to binary.')

    def Dump(self):
        return 'Softmax', []


class AffineTransform(Component):
    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__(input_dim, output_dim)
        self.linearity = numpy.empty([input_dim, output_dim], dtype=numpy.float32, order='C')
        self.bias = numpy.empty([output_dim], dtype=numpy.float32, order='C')

    def Read(self, stream, binary):
        if not binary:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'nnet-copy\' in kaldi '
                     'to convert it to binary.')
        # Read all the '<Tokens>' in arbitrary order,
        # TODO(cfyeh): figure out whether to store these values.
        while stream.Peek(1) == '<':
            if stream.Peek(2) == '<L':
                ExpectToken(stream, binary, '<LearnRateCoef>')
                learn_rate_coef = ReadFloat(stream, binary)
            elif stream.Peek(2) == '<B':
                ExpectToken(stream, binary, '<BiasLearnRateCoef>')
                bias_learn_rate_coef = ReadFloat(stream, binary)
            elif stream.Peek(2) == '<M':
                ExpectToken(stream, binary, '<MaxNorm>')
                max_norm = ReadFloat(stream, binary)
            else:
                token = ReadToken(stream, binary)
                LogError('Unknown token \"%s\"' % token)

        matrix = FloatMatrix()
        matrix.Read(stream, binary)
        self.linearity = matrix.Value()
        vector = FloatVector()
        vector.Read(stream, binary)
        self.bias = vector.Value()
        return True

    def Dump(self):
        return 'AffineTransform', [self.linearity, self.bias]


class Nnet(object):
    """Kaldi nnet1 model."""
    def __init__(self, rxfilename = None):
        self.components = []
        if rxfilename is not None:
            istream = Input(rxfilename)
            self.Read(istream.Stream(), istream.IsBinary())
            istream.Close()

    def Read(self, stream, binary):
        if not binary:
            LogError('We do not supoort text mode Read(), as it could be '
                     'quite slow using python. If you would really like to '
                     'use text format, please use \'nnet-copy\' in kaldi '
                     'to convert it to binary.')
        while True:
            component = self.ReadComponent(stream, binary)
            if component is None:
                break
            self.components.append(component)
        return True

    def ReadComponent(self, stream, binary):
        token = ReadToken(stream, binary)
        if token == '<Nnet>':  # Skip the optional initial token.
            token = ReadToken(stream, binary)
        if token == "</Nnet>":  # Network ends after terminal token appears.
            return None

        input_dim = ReadInt32(stream, binary)
        output_dim= ReadInt32(stream, binary)

        component = None
        if token == '<AffineTransform>':
            component = AffineTransform(input_dim, output_dim)
        elif token == '<Sigmoid>':
            component = Sigmoid(input_dim, output_dim)
        elif token == '<Softmax>':
            component = Softmax(input_dim, output_dim)
        else:
            LogError('Unrecognized or not yet supported component \"%s\"' % token)

        component.Read(stream, binary)
        if stream.Peek(2) == '<!':
            ExpectToken(stream, binary, '<!EndOfComponent>')

        return component

    def NumComponents(self):
        return len(self.components)

    def DumpComponent(self, idx):
        return self.components[idx].Dump()
