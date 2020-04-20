"""This module can accept performance sets for various models and will produce a mask for a hard set.

Returns:
    Array<int> -- For each example in a split of the dataset, this array contains a 1 (True) or 0 (False) if that example is in the hard set.
"""
import os
import numpy as np

from Intermediate import write_num_array_to_file, read_file_to_num_array, get_eda_path

# Parameters for defining the hard set
# A model "captures" this validation example if its predicted rank is at least this high
min_rank = 10
# The number of models a validation example can miss for it to not count as "hard"
model_tolerance = 0

def define_hard_set(dataset_name, split, models):
    def hard_set_filter(ranks_across_models):
        number_of_models = len(ranks_across_models)

        capturing_models = [model_rank < min_rank for model_rank in ranks_across_models]
        number_of_capturing_models = sum(capturing_models)
        return (number_of_capturing_models < number_of_models - model_tolerance)

    def get_full_path_for_model(model):
        file_name = dataset_name + '-' + model + '-' + split + '-rank.txt'
        return get_eda_path(file_name)

    paths = [get_full_path_for_model(model) for model in models]
    performance_sets = [read_file_to_num_array(path) for path in paths]
    performance_sets_by_triple = np.transpose(performance_sets)
    hard_set_mask = [
        1 if hard_set_filter(ranks_across_models) else 0
        for ranks_across_models
        in performance_sets_by_triple
    ]

    # Save to a file
    filename = dataset_name + '-' + split + '-hard-set.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, hard_set_mask)

    return hard_set_mask

if __name__ == "__main__":
    dataset_name = 'fb15k-237'
    split = 'valid'
    models = [
        'complex',
        'conve',
        'distmult',
        'rescal',
        'transe',
    ]

    define_hard_set(dataset_name, split, models)
    
