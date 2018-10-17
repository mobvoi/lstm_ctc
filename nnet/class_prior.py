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

import os
import math
import numpy as np

PRIOR_CUTOFF=1e-10

def read_label_counts(label_counts):
    with open(label_counts) as fi:
        for line in fi:
            strs = line.strip().lstrip('[').rstrip(']').strip().split()
            return [float(k) for k in strs]


def get_class_prior(label_counts):
    a = read_label_counts(label_counts)
    dis = np.asarray(a, dtype=np.float32)
    dis = dis/np.sum(dis)
    log_dis = np.log(dis)
    
    # remove cutoff, make it zero probability
    for i in range(len(dis)):
        if dis[i] < PRIOR_CUTOFF:
            log_dis[i] = -1e10

    # move the blank to the end
    tmp = log_dis[0]
    for i in range(1,len(dis)):
        log_dis[i-1] = log_dis[i]
    log_dis[len(log_dis)-1] = tmp

    return log_dis
