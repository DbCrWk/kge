# This file tabules the frequency of relations within the different splits
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

p_train = model.dataset.split('train').select(1, 1)
p_test = model.dataset.split('test').select(1, 1)
p_valid = model.dataset.split('valid').select(1, 1)

number_of_total_relations = model.dataset.num_relations()

count_in_train_base = numpy.zeros(number_of_total_relations)
count_in_test_base = numpy.zeros(number_of_total_relations)
count_in_valid_base = numpy.zeros(number_of_total_relations)

count_in_train = [
    sum([
        1 if int(relation.item()) == i else 0
        for (j, relation)
        in enumerate(p_train)
    ])
    for (i, _)
    in enumerate(count_in_train_base)
]

count_in_test = [
    sum([
        1 if int(relation.item()) == i else 0
        for (j, relation)
        in enumerate(p_test)
    ])
    for (i, _)
    in enumerate(count_in_test_base)
]

count_in_valid = [
    sum([
        1 if int(relation.item()) == i else 0
        for (j, relation)
        in enumerate(p_valid)
    ])
    for (i, _)
    in enumerate(count_in_valid_base)
]

with open(dataset_name + '-count-in-train.txt', 'w') as f:
    for c in count_in_train:
        f.write("%d\n" % c)

with open(dataset_name + '-count-in-test.txt', 'w') as f:
    for c in count_in_test:
        f.write("%d\n" % c)

with open(dataset_name + '-count-in-valid.txt', 'w') as f:
    for c in count_in_valid:
        f.write("%d\n" % c)
