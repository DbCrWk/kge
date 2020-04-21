"""This module contains utilities for writing intermediate files and results.
"""
import os

FILE_MODE_APPEND = 'a'
FILE_MODE_OVERWRITE = 'w'
FILE_MODE_READ = 'r'

base_exp_path = os.path.join('.', 'local', 'best')
base_eda_path = os.path.join('.', 'eda')

def write_num_array_to_file(full_file_path, array):
    with open(full_file_path, FILE_MODE_OVERWRITE) as f:
        for i in array:
            f.write("%d\n" % i)

def write_float_array_to_file(full_file_path, array):
    with open(full_file_path, FILE_MODE_OVERWRITE) as f:
        for i in array:
            f.write("%8.8f\n" % i)

def read_file_to_num_array(full_file_path):
    f = open(full_file_path, FILE_MODE_READ)
    raw_lines = f.readlines()
    f.close()

    return [int(line) for line in raw_lines]

def base_path_factory(base_path):
    def get_full_path(relative_path):
        return os.path.join(base_path, relative_path)
    
    return get_full_path

get_exp_path = base_path_factory(base_exp_path)
get_eda_path = base_path_factory(base_eda_path)

def get_model_file(dataset_name, model_name):
    dataset_and_model = dataset_name + '-' + model_name
    dataset_and_model_file = dataset_and_model + '.pt'
    return get_exp_path(os.path.join(model_name, dataset_and_model_file))
