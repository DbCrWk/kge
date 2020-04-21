"""This module tabulates the hardness score for a ration

Returns:
    Array<int> -- List of hardness scores; indexed by relation
"""
import os
import numpy
import torch
import kge.model

from Intermediate import read_file_to_num_array, write_float_array_to_file, get_eda_path, get_model_file

def tabulate_relation_hardness_score(dataset_name, split):
    # Get hard set per relation data
    hard_set_file = dataset_name + '-' + split + '-hard-set-by-relation.txt'
    hard_set_full_path = get_eda_path(hard_set_file)
    hard_set = read_file_to_num_array(hard_set_full_path)

    # Get total count per relation data
    count_file = dataset_name + '-' + split + '-by-relation.txt'
    count_full_path = get_eda_path(count_file)
    count_in_split = read_file_to_num_array(count_full_path)

    hard_ratio = [
        -1 if count == 0 else float(hard_set[i]) / float(count)
        for (i, count) in enumerate(count_in_split)
    ]

    # Write to file
    filename = dataset_name + '-' + split + '-hardness-ratio-by-relation.txt'
    full_path = get_eda_path(filename)
    write_float_array_to_file(full_path, hard_ratio)

    return hard_ratio

if __name__ == "__main__":
    tabulate_relation_hardness_score('fb15k-237', 'test')
    tabulate_relation_hardness_score('fb15k-237', 'train')
    tabulate_relation_hardness_score('fb15k-237', 'valid')
