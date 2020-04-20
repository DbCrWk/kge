"""This module tabulates a hard set by relation.

Returns:
    Array<int> -- Gives the number of hard set examples per relation
"""
import os
import numpy
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file

def tabulate_hard_set_by_relation(dataset_name, split):
    # To generate a dataset instance, it's easier to load an existing model against the data
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)
    relation_by_triple = model.dataset.split(split).select(1, 1)

    # Get hard set mask data
    hard_set_file = dataset_name + '-' + split + '-hard-set.txt'
    hard_set_full_path = get_eda_path(hard_set_file)
    hard_set_mask = read_file_to_num_array(hard_set_full_path)

    # Initialize and populate counts for the hard set
    count_in_hard_set = numpy.zeros(model.dataset.num_relations())
    for (j, rel) in enumerate(relation_by_triple):
        count_in_hard_set[rel] += hard_set_mask[j]

    # Write to file
    filename = dataset_name + '-' + split + '-hard-set-by-relation.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, count_in_hard_set)

    return count_in_hard_set

if __name__ == "__main__":
    tabulate_hard_set_by_relation('fb15k-237', 'valid')
