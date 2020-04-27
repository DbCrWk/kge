"""This module accepts a particular relation id and elucidates examples for that particular relation.
"""
import os
import torch
import kge.model
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from Intermediate import read_file_to_num_array, write_float_array_to_file, get_eda_path, get_model_file

def elucidate_relation_examples(dataset_name, split, relation_id):
    dummy_model_name = 'rescal'
    model_file = get_model_file(dataset_name, dummy_model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)

    data = model.dataset.split(split)
    s = data.select(1, 0)
    p = data.select(1, 1)
    o = data.select(1, 2)

    print('relation selected: ' + model.dataset.relation_strings(relation_id))
    examples = []
    edge_set = []
    for (i, rel) in enumerate(p):
        if rel == relation_id:
            examples.append((model.dataset.entity_strings(
                s[i].item()), model.dataset.entity_strings(o[i].item())))
            edge_set.append(( s[i].item(), o[i].item() ))

    for example in examples:
        print_example(example)

    G = nx.DiGraph()
    G.add_edges_from(edge_set)
    nodelist = list(G.nodes)

    A = nx.adjacency_matrix(G, nodelist=nodelist).todense()
    row_sum = np.sum(A, 1).flatten().tolist()[0]
    ind = np.argsort(row_sum).tolist()

    def print_entity(i):
        index_in_node_list = ind[i]
        entity_id = nodelist[index_in_node_list]
        entity_string = model.dataset.entity_strings(entity_id)
        print('Top Subjects : ' + str(i) + ' place : ' + str(row_sum[index_in_node_list]) + ' freq : ' + entity_string)

    for (i, _) in enumerate(ind):
        print_entity(i)

    A_sorted = A[ind, :]

    plt.matshow(A_sorted)
    plt.show()

    return examples

def print_example(example):
    s, o = example
    print(str(s) + ' -> ' + str(o))

if __name__ == "__main__":
    elucidate_relation_examples('fb15k-237', 'train', 233)
    
