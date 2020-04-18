# This file tabules the frequency of entities within the different splits
import os
import numpy
import torch
import kge.model

# Location of all experiment files
base_path = os.path.join('.', 'local', 'best')

# Change these variables to select the model and dataset
dataset_name = 'fb15k-237'
model_name = 'rescal'
dataset_and_model = dataset_name + '-' + model_name
dataset_and_model_file = dataset_and_model + '.pt'

# Reconstruct model
full_path = os.path.join(base_path, model_name, dataset_and_model_file)
model = kge.model.KgeModel.load_from_checkpoint(full_path)

s_train = model.dataset.split('train').select(1, 0)
s_test = model.dataset.split('test').select(1, 0)
s_valid = model.dataset.split('valid').select(1, 0)

o_train = model.dataset.split('train').select(1, 2)
o_test = model.dataset.split('test').select(1, 2)
o_valid = model.dataset.split('valid').select(1, 2)

number_of_total_entities = model.dataset.num_entities()

s_count_in_train = numpy.zeros(number_of_total_entities)
s_count_in_test = numpy.zeros(number_of_total_entities)
s_count_in_valid = numpy.zeros(number_of_total_entities)

o_count_in_train = numpy.zeros(number_of_total_entities)
o_count_in_test = numpy.zeros(number_of_total_entities)
o_count_in_valid = numpy.zeros(number_of_total_entities)

for (j, entity) in enumerate(s_train):
    i = int(entity.item())
    s_count_in_train[i] += 1

for (j, entity) in enumerate(s_test):
    i = int(entity.item())
    s_count_in_test[i] += 1

for (j, entity) in enumerate(s_valid):
    i = int(entity.item())
    s_count_in_valid[i] += 1

for (j, entity) in enumerate(o_train):
    i = int(entity.item())
    o_count_in_train[i] += 1

for (j, entity) in enumerate(o_test):
    i = int(entity.item())
    o_count_in_test[i] += 1

for (j, entity) in enumerate(o_valid):
    i = int(entity.item())
    o_count_in_valid[i] += 1

with open(dataset_name + '-subj-count-in-train.txt', 'w') as f:
    for c in s_count_in_train:
        f.write("%d\n" % c)

with open(dataset_name + '-subj-count-in-test.txt', 'w') as f:
    for c in s_count_in_test:
        f.write("%d\n" % c)

with open(dataset_name + '-subj-count-in-valid.txt', 'w') as f:
    for c in s_count_in_valid:
        f.write("%d\n" % c)

with open(dataset_name + '-obj-count-in-train.txt', 'w') as f:
    for c in o_count_in_train:
        f.write("%d\n" % c)

with open(dataset_name + '-obj-count-in-test.txt', 'w') as f:
    for c in o_count_in_test:
        f.write("%d\n" % c)

with open(dataset_name + '-obj-count-in-valid.txt', 'w') as f:
    for c in o_count_in_valid:
        f.write("%d\n" % c)
