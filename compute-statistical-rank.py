"""This module produces a statistics-based score and rank

Returns:
    Array<int> -- For each example in a split of the dataset, this array contains the final rank (with ties) assigned by the statistical model.
"""
import os
import networkx as nx
import numpy as np
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_num_array_to_file, get_eda_path, get_model_file

# Accept splits for path length and freq analysis
# Compute relation and object frequency

def compute_statistical_rank(dataset_name, union_graph_splits, eval_split):
    print('Union graph of: ' + '-'.join(union_graph_splits))
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)
    num_relations = model.dataset.num_relations()
    num_entities = model.dataset.num_entities()

    union_graph_edge_set = []
    for split in union_graph_splits:
        data = model.dataset.split(split)
        s = data.select(1, 0)
        o = data.select(1, 2)

        for (i, sbj) in enumerate(s):
            union_graph_edge_set.append((sbj.item(), o[i].item(), 1))

    SUG = nx.Graph()
    SUG.add_weighted_edges_from(union_graph_edge_set)

    print('Computing shortest paths')
    shortest_paths = dict(nx.johnson(SUG))

    object_freq = np.zeros((num_relations, num_entities))
    print('Computing object frequency')
    for split in union_graph_splits:
        data = model.dataset.split(split)
        p = data.select(1, 1)
        o = data.select(1, 2)

        for (i, rel) in enumerate(p):
            obj = o[i]
            object_freq[rel][obj] += 1

    

    print('Done!')

if __name__ == "__main__":
    compute_statistical_rank('fb15k-237', ['train'], 'valid')
