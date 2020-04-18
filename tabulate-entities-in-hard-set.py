# This file accepts a hard set mask for validation triples. It outputs, for each
# entity, the number of times it appears in the hard or easy set.
import os
import numpy
import torch
import kge.model

def read_hard_set_mask(full_path):
    # Get each line
    f = open(full_path, 'r')
    raw_lines = f.readlines()
    f.close()

    # Convert each line to an integer array of ranks
    hard_set_mask = [int(line) for line in raw_lines]

    return hard_set_mask

dataset_name = 'fb15k-237'
full_file_name = dataset_name + '-hard-set.txt'
file_path = ''
full_file_path = os.path.join(file_path, full_file_name)

hard_set_mask = read_hard_set_mask(full_file_path)

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
valid_split = model.dataset.split('valid')

subject_by_triple = valid_split.select(1, 0)
object_by_triple = valid_split.select(1, 2)

number_of_total_entities = model.dataset.num_entities()

s_count_in_hard_set = numpy.zeros(number_of_total_entities)
o_count_in_hard_set = numpy.zeros(number_of_total_entities)
s_count_in_easy_set = numpy.zeros(number_of_total_entities)
o_count_in_easy_set = numpy.zeros(number_of_total_entities)

for (j, entity) in enumerate(subject_by_triple):
    i = int(entity.item())
    s_count_in_hard_set[i] += hard_set_mask[j]

with open(dataset_name + '-subj-count-hard-set.txt', 'w') as f:
    for c in s_count_in_hard_set:
        f.write("%d\n" % c)

for (j, entity) in enumerate(object_by_triple):
    i = int(entity.item())
    o_count_in_hard_set[i] += hard_set_mask[j]

with open(dataset_name + '-subj-count-in-easy-set.txt', 'w') as f:
    for c in s_count_in_easy_set:
        f.write("%d\n" % c)

for (j, entity) in enumerate(subject_by_triple):
    i = int(entity.item())
    s_count_in_easy_set[i] += (1 - hard_set_mask[j])

with open(dataset_name + '-obj-count-hard-set.txt', 'w') as f:
    for c in o_count_in_hard_set:
        f.write("%d\n" % c)

for (j, entity) in enumerate(object_by_triple):
    i = int(entity.item())
    o_count_in_easy_set[i] += (1 - hard_set_mask[j])

with open(dataset_name + '-obj-count-in-easy-set.txt', 'w') as f:
    for c in o_count_in_easy_set:
        f.write("%d\n" % c)
