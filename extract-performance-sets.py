"""This module produces the performance set for a particular model against a particular split of a dataset.

Returns:
    Array<int> -- For each example in a split of the dataset, this array contains the final rank (with ties) assigned by the model.
"""
import os
import torch
import kge.model

from Intermediate import write_num_array_to_file, get_exp_path, get_eda_path, get_model_file

def extract_performance_set(dataset_name, model_name, split):
    # Reconstruct model
    model_file = get_model_file(dataset_name, model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)

    # Grab subjects, predicates, and objects
    s = model.dataset.split(split).select(1, 0)
    p = model.dataset.split(split).select(1, 1)
    o = model.dataset.split(split).select(1, 2)

    # Compute scores
    scores = model.score_sp(s, p)
    raw_ranks = [torch.sum(score_array > score_array[o[i]], dtype=torch.long) for (i, score_array) in enumerate(scores)]
    num_ties = [torch.sum(score_array == score_array[o[i]], dtype=torch.long) for (i, score_array) in enumerate(scores)]
    final_ranks = [(raw_rank + (num_ties[j] // 2)).item() for (j, raw_rank) in enumerate(raw_ranks)]

    # Write to file - optional
    filename = dataset_name + '-' + model_name + '-' + split + '-rank.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, final_ranks)

    return final_ranks

if __name__ == "__main__":
    models = [
        'complex',
        'conve',
        'distmult',
        'rescal',
        'transe',
    ]
    [
        extract_performance_set('fb15k-237', model, 'valid')
        for model in models
    ]
