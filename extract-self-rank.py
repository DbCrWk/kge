"""
This module produces gives the average self-rank for each relation, i.e. the average rank for (s, p, s)

Returns:
    Array<int> -- For each relation p, the average rank for (s, p, s) over all s in E
"""
import os
import torch
import kge.model

from Intermediate import write_num_array_to_file, get_exp_path, get_eda_path, get_model_file


def extract_performance_set(dataset_name, model_name):
    # Reconstruct model
    print('dataset, model', dataset_name, model_name)
    model_file = get_model_file(dataset_name, model_name)
    model = kge.model.KgeModel.load_from_checkpoint(model_file)
    number_of_relations = model.dataset.num_relations()
    number_of_entities = model.dataset.num_entities()

    average_rank = torch.zeros(number_of_relations)
    for (k, relation) in enumerate(range(number_of_relations)):
        print('relation k = ' + str(k))
        subject_array = torch.tensor(range(number_of_entities))
        relation_array = torch.zeros(number_of_entities).fill_(relation)

        scores = model.score_sp(subject_array, relation_array)
        raw_ranks = [torch.sum(score_array > score_array[i], dtype=torch.long)
                     for (i, score_array) in enumerate(scores)]
        num_ties = [torch.sum(score_array == score_array[i], dtype=torch.long)
                    for (i, score_array) in enumerate(scores)]
        final_ranks = [(raw_rank + (num_ties[j] // 2)).item()
                       for (j, raw_rank) in enumerate(raw_ranks)]

        average_rank[k] = sum(final_ranks) / number_of_entities

    # Write to file - optional
    filename = dataset_name + '-' + model_name + '-self-rank.txt'
    full_path = get_eda_path(filename)
    write_num_array_to_file(full_path, average_rank)

    return average_rank


if __name__ == "__main__":
    models = [
        'complex',
        'conve',
        'distmult',
        'rescal',
        'transe',
    ]
    [
        extract_performance_set('fb15k-237', model)
        for model in models
    ]
