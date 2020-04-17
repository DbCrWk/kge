import os
import torch
import kge.model

# Location of all experiment files
base_path = os.path.join('.', 'local', 'best')

# Change these variables to select the model and dataset
dataset_name = 'fb15k-237'
model_name = 'rescal'
dataset_and_model = dataset_name + '-' + model_name
dataset_and_model_file = dataset_and_model + '.pt'

# Reconstruct model
full_path = os.path.join(base_path, model_name, dataset_and_model_file)
model = kge.model.KgeModel.load_from_checkpoint(full_path)

# Grab subjects, predicates, and objects
s = model.dataset.split('valid').select(1, 0)
p = model.dataset.split('valid').select(1, 1)
o = model.dataset.split('valid').select(1, 2)

# Compute scores
scores = model.score_sp(s, p)
raw_ranks = [torch.sum(score_array > score_array[o[i]], dtype=torch.long) for (i, score_array) in enumerate(scores)]
num_ties = [torch.sum(score_array == score_array[o[i]], dtype=torch.long) for (i, score_array) in enumerate(scores)]
final_ranks = [(raw_rank + (num_ties[j] // 2)) for (j, raw_rank) in enumerate(raw_ranks)]

# Save to a file
with open(dataset_and_model + '-valid-rank.txt', 'a') as f:
    for item in final_ranks:
        f.write("%s\n" % item.item())
