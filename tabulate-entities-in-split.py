"""This module tabulates a split per entity

Returns:
    Array<int> -- Gives the number of examples in the split per entity
"""
import os
import numpy
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file


def tabulate_split_by_entity(dataset_name, split, entity_position):
    # To generate a dataset instance, it's easier to load an existing model against the data
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)
    index_for_entity_position = 0 if entity_position == 'sbj' else 2
    entity_by_triple = model.dataset.split(split).select(1, index_for_entity_position)

    # Initialize and populate counts for the split
    count_by_entity = numpy.zeros(model.dataset.num_entities())
    for ent in entity_by_triple:
        count_by_entity[ent] += 1

    # Write to file
    filename = dataset_name + '-' + split + '-by-' + entity_position + '.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, count_by_entity)

    return count_by_entity


if __name__ == "__main__":
    tabulate_split_by_entity('fb15k-237', 'test', 'sbj')
    tabulate_split_by_entity('fb15k-237', 'test', 'obj')
    tabulate_split_by_entity('fb15k-237', 'train', 'sbj')
    tabulate_split_by_entity('fb15k-237', 'train', 'obj')
    tabulate_split_by_entity('fb15k-237', 'valid', 'sbj')
    tabulate_split_by_entity('fb15k-237', 'valid', 'obj')
