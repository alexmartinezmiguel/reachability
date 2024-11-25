import torch
import numpy as np 
import argparse
import random
import pickle
import time
import tqdm
import time

from utils import compute_F, compute_mtt_abs, torch_delete, \
     chunks, select_permitted_rewirings, compute_mean_expected_path, list_similarity


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("newspaper", type=str)
parser.add_argument("start_date", type=str)
parser.add_argument("end_date", type=str)
parser.add_argument("out_degree", type=int)
args = parser.parse_args()

# set device (CPU or GPU) and precision for the computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cpu':
    print('Could not use GPU, using CPU instead')
precision = torch.float32
device = torch.device('cpu')

# load similarity matrix and transition probabilities of given dataset 
dataset = 'nelagt'
#newspaper = 'theguardian'
newspaper = args.newspaper
start_date = args.start_date
end_date = args.end_date
k = args.out_degree


def baseline_diversify(probs,similarities,threshold):
    random.seed(0)
    permitted_rewirings = select_permitted_rewirings(probs,similarities,threshold=threshold)
    diversify_ratio = list()
    for rewiring in permitted_rewirings:
        source_node = rewiring[0]
        original_target = rewiring[1]
        new_target = rewiring[2]

        original_list = list(np.where(probs[source_node]!=0.0)[0])
        # original diversify
        div_original = list_similarity(original_list, similarities)
        # Find the index of the element to delete
        index_to_delete = original_list.index(original_target)
        # Delete the element at the specified index
        del original_list[index_to_delete]

        # Insert the new element at the same index
        original_list.insert(index_to_delete, new_target)

        # compute new diversify
        div_final = list_similarity(original_list, similarities)

        # track the change in list similarity
        diversify_ratio.append(div_final/div_original)
    ordered_diversify_ratio_index = np.argsort(diversify_ratio)
    ordered_diversify_rewirings = [permitted_rewirings[i] for i in ordered_diversify_ratio_index]
    
    return ordered_diversify_rewirings





# load similarity matrix
with open(f'data/{dataset}/similarity_matrices/similarity_matrix_{newspaper}_from_{start_date}_to_{end_date}','rb') as f:
    similarity_matrix = torch.tensor(pickle.load(f), dtype=precision, device=device)

with open(f'data/{dataset}/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}','rb') as f:
    trans_prob = torch.tensor(pickle.load(f), dtype=precision, device=device)
# useful objects for computations
all_items = set(range(len(trans_prob)))
n_items = len(trans_prob)

# compute nodes to rewire (lowest diversity)
nodes_to_rewire = list()
# inital diversity
similarities = dict()    
for source_node in range(len(trans_prob)):
    targets = np.argsort(-trans_prob[source_node])[:k]
    similarities.setdefault(source_node,list_similarity(targets, similarity_matrix))
# take k nodes to rewire
nodes_to_rewire = list({k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])}.keys())[:len(similarity_matrix)]

accepted_diversify_rewirings = baseline_diversify(trans_prob, similarity_matrix, threshold=0.95)
# save
#with open(f'results/B3_diversify/rewirings/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_B3_diversify', 'wb') as f:
#    pickle.dump(accepted_diversify_rewirings, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/B3_diversify/rewirings_v2/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_B3_diversify', 'wb') as f:
    pickle.dump(accepted_diversify_rewirings, f, protocol=pickle.HIGHEST_PROTOCOL)
