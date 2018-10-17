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
from io_funcs import ReadInt32

class Index(object):
    """Index is intended to represent the various indexes by which we number
    the rows of the matrices that the Components process: mainly 'n', the index
    of the member of the minibatch, 't', used for the frame index in speech
    recognition, and 'x', which is a catch-all extra index which we might use in
    convolutional setups or for other reasons.  It is possible to extend this by
    adding new indexes if needed.
    """
    def __init__(self, n = 0, t = 0, x = 0):
        self.n = n
        self.t = t
        self.x = x

    def Read(self, stream, binary):
        ExpectToken(stream, binary, '<I1>')
        self.n = ReadInt32(stream, binary)
        self.t = ReadInt32(stream, binary)
        self.x = ReadInt32(stream, binary)


def ReadIndexVectorElementBinary(stream, i, vec):
    """Read a Index object in a list of Index.

    Args:
        stream: An opened KaldiInputStream.
        i: The index in the target vector to read.
        vec: The target index vector.
    """
    binary = True
    p = stream.Read(1)
    c = ord(p)
    if abs(c) < 125:
        last_n = 0
        last_t = 0
        last_x = 0
        if i > 0:
            last_index = vec[i-1]
            last_n = last_index.n
            last_t = last_index.t
            last_x = last_index.x
        vec[i].n = last_n
        vec[i].t = last_t + c
        vec[i].x = last_x
    else:
        if c != 127:
            LogError('Unexpected character \"%s\" encountered while '
                     'reading Index vector.' % p)
        vec[i].n = ReadInt32(stream, binary)
        vec[i].t = ReadInt32(stream, binary)
        vec[i].x = ReadInt32(stream, binary)

def ReadIndexVector(stream, binary):
    """Read a list of Index from stream.

    Args:
        stream: An opened KaldiInputStream.
        i: The index in the target vector to read.
        vec: The target index vector.
    """
    ExpectToken(stream, binary, "<I1V>")
    size = ReadInt32(stream, binary)
    if size < 0:
        LogError('Error reading Index vector: size = %d' % size)
    vec = []
    for i in xrange(size):
       vec.append(Index())
    if not binary:
        for i in xrange(size):
            vec[i].Read(stream, binary)
    else:
        for i in xrange(size):
            ReadIndexVectorElementBinary(stream, i, vec)
    return vec
