"""This module computes the rank of the relation graph

!Note: this module exploits the fact that for a matrix, eliminating rows and
columns that are identically 0 does not change the rank of a matrix. Therefore,
for each relation, the adjacency matrix of the graph is actually expressed as
the submatrix that has rows and columns that correspond to entities that have
edges for the particular relation.

Returns:
    Array<int> -- Gives the rank for each relation graph
"""
import os
import networkx as nx
import numpy as np
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file

def compute_rank_from_graph(G):
    print('computing rank', len(G.edges))
    if len(G.edges) == 0: return 0

    adjacency_matrix_raw = nx.adjacency_matrix(G)
    adjacency_matrix = adjacency_matrix_raw.asfptype()
    est_rank = np.linalg.matrix_rank(adjacency_matrix.todense())

    return est_rank

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

def compute_rank_of_relation_graph(dataset_name, splits):
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

    ranks = [
        compute_rank_from_graph(
            create_graph_from_edge_set(edge_set)
        )
        for edge_set in edge_sets
    ]

    # Write to file
    filename = dataset_name + '-' + '-'.join(splits) + '-rank-by-relation.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, ranks)

    return ranks


if __name__ == "__main__":
    compute_rank_of_relation_graph('fb15k-237', ['train'])
    compute_rank_of_relation_graph('fb15k-237', ['train', 'valid'])
    compute_rank_of_relation_graph('fb15k-237', ['train', 'valid', 'test'])
    compute_rank_of_relation_graph('fb15k-237', ['valid'])
    compute_rank_of_relation_graph('fb15k-237', ['test'])
