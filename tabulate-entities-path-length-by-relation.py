"""This module tabulates the average path length for entities in a relation

Returns:
    Array<int> -- Gives the average path length for a relation
"""
import os
import numpy
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_float_array_to_file, get_eda_path, get_model_file


def tabulate_entities_path_length_by_relation(dataset_name, union_graph_splits, path_length_split):
    # To generate a dataset instance, it's easier to load an existing model against the data
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)
    relation_by_triple = model.dataset.split(path_length_split).select(1, 1)

    # Get hard set mask data
    path_length_filename = dataset_name + \
        '-from-' + '-'.join(union_graph_splits) + '-shortest-path-on-' + \
        path_length_split + '.txt'
    path_length_full_path = get_eda_path(path_length_filename)
    path_lengths = read_file_to_num_array(path_length_full_path)

    # Initialize and populate counts for the hard set
    total_path_lengths = numpy.zeros(model.dataset.num_relations())
    total_entities = numpy.zeros(model.dataset.num_relations())
    for (j, rel) in enumerate(relation_by_triple):
        total_path_lengths[rel.item()] += path_lengths[j]
        total_entities[rel] += 1

    average_path_length = [
        float(total_path_length) / float(total_entities[k]) if total_entities[k] != 0 else -1
        for (k, total_path_length) in enumerate(total_path_lengths)
    ]

    # Write to file
    filename = dataset_name + '-relation-average-shortest-path-from-' + '-'.join(union_graph_splits) + '-shortest-path-on-' + path_length_split + '.txt'
    full_path = get_eda_path(filename)
    write_float_array_to_file(full_path, average_path_length)

    return average_path_length


if __name__ == "__main__":
    tabulate_entities_path_length_by_relation('fb15k-237', ['train', 'valid'], 'test')
