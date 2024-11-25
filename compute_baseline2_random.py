import torch
import numpy as np 
import argparse
import random
import pickle
import time
import tqdm
import time

from utils import compute_F, compute_mtt_abs, torch_delete, \
     chunks, select_permitted_rewirings, compute_mean_expected_path


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
threshold = 0.95

def baseline_random(probs, similarities, n_rewirings):
    random.seed(0)
    permitted_rewirings = select_permitted_rewirings(probs,similarities,threshold=threshold)
    random.shuffle(permitted_rewirings)
    return random.sample(permitted_rewirings, n_rewirings)


# load similarity matrix
with open(f'data/{dataset}/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}','rb') as f:
    trans_prob = torch.tensor(pickle.load(f), dtype=precision, device=device)
with open(f'data/{dataset}/similarity_matrices/similarity_matrix_{newspaper}_from_{start_date}_to_{end_date}','rb') as f:
    similarity_matrix = torch.tensor(pickle.load(f), dtype=precision, device=device)
# useful objects for computations
all_items = set(range(len(trans_prob)))
n_items = len(trans_prob)


accepted_random_rewirings = baseline_random(trans_prob, similarity_matrix, 10000)
# save
#with open(f'results/B2_random/rewirings/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_B2_random', 'wb') as f:
#    pickle.dump(accepted_random_rewirings, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'results/B2_random/rewirings_v2/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_B2_random', 'wb') as f:
    pickle.dump(accepted_random_rewirings, f, protocol=pickle.HIGHEST_PROTOCOL)
