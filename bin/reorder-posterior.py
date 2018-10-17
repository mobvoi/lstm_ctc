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

import sys

if __name__ == '__main__':
    train = sys.argv[1]
    decode = sys.argv[2]
    train_phone_to_index = dict()
    decode_to_train = []

    for line in open(train, 'r'):
        line = line.strip()
        phone = line[:line.find(' ')]
        index = int(line[line.rfind(' ')+1:])
        train_phone_to_index[phone] = index

    for line in open(decode, 'r'):
        line = line.strip()
        phone = line[:line.find(' ')]
        index = int(line[line.rfind(' ')+1:])
        while len(decode_to_train) <= index:
            decode_to_train.append(None)
        decode_to_train[index] = train_phone_to_index[phone]

    # manually add mapping for <eps>
    decode_to_train[0] = train_phone_to_index['<blank>']

    res = ','.join([ '%d' % p for p in decode_to_train ])
    print res
