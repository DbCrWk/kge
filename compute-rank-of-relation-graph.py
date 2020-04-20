# This file construct graph representations for each relation

import os
import numpy
import scipy
import networkx as nx
import torch
import kge.model

# Change these variables to select the model and dataset
dataset_name = 'fb15k-237'
model_name = 'rescal'
dataset_and_model = dataset_name + '-' + model_name
dataset_and_model_file = dataset_and_model + '.pt'

# Location of all experiment files
base_path = os.path.join('.', 'local', 'best')

# Reconstruct model
full_path = os.path.join(base_path, model_name, dataset_and_model_file)
model = kge.model.KgeModel.load_from_checkpoint(full_path)

split = model.dataset.split('train')
s = split.select(1, 0)
p = split.select(1, 1)
o = split.select(1, 2)

number_of_total_relations = model.dataset.num_relations()
number_of_total_entities = model.dataset.num_entities()
edge_sets = [
    []
    for i in range(number_of_total_relations)
]
entity_set = range(number_of_total_entities)

for (i, rel) in enumerate(p):
    edge_sets[rel.item()].append((s[i].item(), o[i].item()))

graph_sets = []
ranks = []
for (j, edge_set) in enumerate(edge_sets):
    print('[computing] : ' + str(j))
    G = nx.DiGraph()
    # G.add_nodes_from(entity_set)
    G.add_edges_from(edge_set)
    graph_sets.append(G)

    adjacency_matrix_raw = nx.adjacency_matrix(G)
    adjacency_matrix = adjacency_matrix_raw.asfptype()
    est_rank = numpy.linalg.matrix_rank(adjacency_matrix.todense())

    ranks.append(est_rank)

    with open(dataset_name + '-train-relation-ranks.txt', 'a') as f:
        f.write("%d\n" % est_rank)

