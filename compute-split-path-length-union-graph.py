"""
This modules computes the average path length for every example in a split
conditioned on a particular union graph

!Note: this module defines path length based on the undirected union graph
"""
import os
import networkx as nx
import numpy as np
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file

def compute_shortest_path(G, s, o):
    try:
        return nx.shortest_path_length(G, source=s, target=o)
    except nx.NodeNotFound:
        return -1
    except nx.NetworkXNoPath:
        return -2

def compute_split_path_length_union_graph(dataset_name, union_graph_splits, path_length_split):
    print('Union graph of: ' + '-'.join(union_graph_splits))
    print('Computing path lengths of: ' + path_length_split)

    # To generate a dataset instance, it's easier to load an existing model against the data
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)

    union_graph_edge_set = []
    for split in union_graph_splits:
        data = model.dataset.split(split)
        s = data.select(1, 0)
        o = data.select(1, 2)

        for (i, sbj) in enumerate(s): union_graph_edge_set.append(( sbj.item(), o[i].item() ))

    G = nx.Graph()
    G.add_edges_from(union_graph_edge_set)

    split_data = model.dataset.split(path_length_split)
    s = split_data.select(1, 0)
    o = split_data.select(1, 2)

    shortest_paths = [
        compute_shortest_path(G, sbj.item(), o[i].item())
        for (i, sbj) in enumerate(s)
    ]

    # Write to file
    filename = dataset_name + \
        '-from-' + '-'.join(union_graph_splits) + '-shortest-path-on-' + \
        path_length_split + '.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, shortest_paths)

    return shortest_paths


if __name__ == "__main__":
    compute_split_path_length_union_graph('fb15k-237', ['train', 'valid'], 'test')
