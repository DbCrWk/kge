# This file accepts a hard set mask for validation triples. It outputs, for each
# relation, the number of times it appears in the hard or easy set.
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

relation_by_triple = valid_split.select(1, 1)
number_of_total_relations = model.dataset.num_relations()

count_in_hard_set_base = numpy.zeros(number_of_total_relations)
count_in_easy_set_base = numpy.zeros(number_of_total_relations)

count_in_hard_set = [
    sum([
        hard_set_mask[j] if int(relation.item()) == i else 0
        for (j, relation)
        in enumerate(relation_by_triple)
    ])
    for (i, _)
    in enumerate(count_in_hard_set_base)
]

count_in_easy_set = [
    sum([
        (1 - hard_set_mask[j]) if int(relation.item()) == i else 0
        for (j, relation)
        in enumerate(relation_by_triple)
    ])
    for (i, _)
    in enumerate(count_in_hard_set_base)
]

with open(dataset_name + '-count-in-hard-set.txt', 'w') as f:
    for hard_count in count_in_hard_set:
        f.write("%d\n" % hard_count)

with open(dataset_name + '-count-in-easy-set.txt', 'w') as f:
    for easy_count in count_in_easy_set:
        f.write("%d\n" % easy_count)
