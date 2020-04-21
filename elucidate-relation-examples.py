"""This module accepts a particular relation id and elucidates examples for that particular relation.
"""
import os
import torch
import kge.model

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
    for (i, rel) in enumerate(p):
        if rel == relation_id:
            examples.append((model.dataset.entity_strings(
                s[i].item()), model.dataset.entity_strings(o[i].item())))

    return examples

def print_example(example):
    s, o = example
    print(str(s) + ' -> ' + str(o))

if __name__ == "__main__":
    examples = elucidate_relation_examples('fb15k-237', 'train', 233)
    for example in examples: print_example(example)
