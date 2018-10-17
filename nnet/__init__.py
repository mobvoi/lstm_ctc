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


from config import parse_config
from funcs import train
from funcs import validate
from graph import create_graph_for_decoding
from graph import create_graph_for_inference
from graph import create_graph_for_training_ctc
from graph import create_graph_for_validation_ctc
from pipeline import create_pipeline_sequence_batch
from pipeline import create_pipeline_sequential
from tfrecord import dataset_from_tfrecords
from tfrecord import write_tfrecord
from class_prior import get_class_prior
