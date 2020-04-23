"""
This module computes the number of connected components of the relation graph

!Note: this module defines connected components in a precise way by:
1). Converting directed graphs into undirected graphs for the purposes of this computation
2). Ignores any vertices that have no edges in the graph (depending on the
definition, this would contribute to the number of connected components)

Returns:
    Array<int> -- Gives the number of connected components for each relation graph
"""
import os
import networkx as nx
import numpy as np
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file


def compute_num_connected_components_from_graph(G):
    print('computing num_connected_components', len(G.edges))
    if len(G.edges) == 0:
        return 0

    SG = G.to_undirected()
    num_connected_components = nx.number_connected_components(SG)

    return num_connected_components


def create_graph_from_edge_set(edge_set):
    G = nx.DiGraph()
    G.add_edges_from(edge_set)

    return G


def extract_split_edge_set(model, split):
    data = model.dataset.split(split)
    s = data.select(1, 0)
    p = data.select(1, 1)
    o = data.select(1, 2)

    number_of_total_relations = model.dataset.num_relations()
    edge_sets = [[] for i in range(number_of_total_relations)]

    for (i, rel) in enumerate(p):
        edge_sets[rel.item()].append((s[i].item(), o[i].item()))

    return edge_sets


def compute_num_connected_components_of_relation_graph(dataset_name, splits):
    print('-'.join(splits))
    # To generate a dataset instance, it's easier to load an existing model against the data
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)

    edge_sets_per_relation = [
        extract_split_edge_set(model, split)
        for split in splits
    ]

    number_of_total_relations = model.dataset.num_relations()
    edge_sets = [[] for i in range(number_of_total_relations)]

    for edge_set_for_relation in edge_sets_per_relation:
        for (j, edge_set) in enumerate(edge_set_for_relation):
            edge_sets[j] += edge_set

    nums_connected_components = [
        compute_num_connected_components_from_graph(
            create_graph_from_edge_set(edge_set)
        )
        for edge_set in edge_sets
    ]

    # Write to file
    filename = dataset_name + '-' + \
        '-'.join(splits) + '-num-connected-components-by-relation.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, nums_connected_components)

    return nums_connected_components


if __name__ == "__main__":
    compute_num_connected_components_of_relation_graph('fb15k-237', ['train'])
    compute_num_connected_components_of_relation_graph('fb15k-237', ['train', 'valid'])
    compute_num_connected_components_of_relation_graph('fb15k-237', ['train', 'valid', 'test'])
    compute_num_connected_components_of_relation_graph('fb15k-237', ['valid'])
    compute_num_connected_components_of_relation_graph('fb15k-237', ['test'])
