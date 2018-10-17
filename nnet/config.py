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


def str2int(var):
    try:
        res = int(var)
        return res
    except:
        return None

def str2flt(var):
    try:
        res = float(var)
        return res
    except:
        return None

def str2bool(var):
    v = var.lower()
    if v in ['true']:
      return True
    if v in ['false']:
      return False
    return None

def parse_config(fn):
    config = dict()
    for line in open(fn, 'r'):
        line = line.strip()
        if line.startswith('#'):
            continue
        tokens = [ t for t in line.split() if not t.startswith('#')]
        key = tokens[0]
        val = tokens[-1]
        val_int = str2int(val)
        if val_int is not None:
            config[key] = val_int
            continue
        val_flt = str2flt(val)
        if val_flt is not None:
            config[key] = val_flt
            continue
        val_bool = str2bool(val)
        if val_bool is not None:
            config[key] = val_bool
            continue
        config[key] = val

    return config
