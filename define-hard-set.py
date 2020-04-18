# This module inspects a collection of validation sets and can produce a mask
# for validation examples in a "hard" set
import os
import numpy as np

def read_validation_set(full_path):
    # Get each line
    f = open(full_path, 'r')
    raw_lines = f.readlines()
    f.close()

    # Convert each line to an integer array of ranks
    validation_set_ranks = [int(line) for line in raw_lines]

    return validation_set_ranks

# Setup for model names
models = [
   'complex',
   'conve',
   'distmult',
   'rescal',
   'transe',
]
dataset = 'fb15k-237'
base_path = os.path.join('.', 'local', 'best')

# Parameters for defining the hard set
min_rank = 10               # A model "captures" this validation example if its predicted rank is at least this high
model_tolerance = 0         # The number of models a validation example can miss for it to not count as "hard"

def hard_set_filter(ranks_across_models):
    number_of_models = len(ranks_across_models)

    capturing_models = [model_rank
                                    < min_rank for model_rank in ranks_across_models]
    number_of_capturing_models = sum(capturing_models)
    return (number_of_capturing_models < number_of_models - model_tolerance)

paths = [os.path.join(base_path, model, dataset + '-' + model
                      + '-valid-rank.txt') for model in models]

validation_sets = [read_validation_set(path) for path in paths]
validation_sets_by_triple = np.transpose(validation_sets)
hard_set_mask = [hard_set_filter(ranks_across_models)
                 for ranks_across_models in validation_sets_by_triple]

# Save to a file
with open(dataset + '-hard-set.txt', 'w') as f:
    for mask in hard_set_mask:
        f.write("%d\n" % mask)
