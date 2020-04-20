"""This module tabulates a split per relation

Returns:
    Array<int> -- Gives the number of examples in the split per relation
"""
import os
import numpy
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file


def tabulate_split_by_relation(dataset_name, split):
    # To generate a dataset instance, it's easier to load an existing model against the data
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)
    relation_by_triple = model.dataset.split(split).select(1, 1)

    # Initialize and populate counts for the split
    count_by_relation = numpy.zeros(model.dataset.num_relations())
    for rel in relation_by_triple:
        count_by_relation[rel] += 1

    # Write to file
    filename = dataset_name + '-' + split + '-by-relation.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, count_by_relation)

    return count_by_relation


if __name__ == "__main__":
    tabulate_split_by_relation('fb15k-237', 'test')
    tabulate_split_by_relation('fb15k-237', 'train')
    tabulate_split_by_relation('fb15k-237', 'valid')
