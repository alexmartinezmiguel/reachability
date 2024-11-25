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

# load similarity matrix and transition probabilities of given dataset 
dataset = 'nelagt'
newspaper = args.newspaper
start_date = args.start_date
end_date = args.end_date
k = args.out_degree


# load similarity matrix
with open(f'data/{dataset}/similarity_matrices/similarity_matrix_{newspaper}_from_{start_date}_to_{end_date}','rb') as f:
    similarity_matrix = torch.tensor(pickle.load(f), dtype=precision, device=device)
# useful objects for computations
all_items = set(range(len(similarity_matrix)))
n_items = len(similarity_matrix)
I = torch.ones((n_items-1,1), dtype=precision, device=device)
I_T = I.T

print(f'Running Baseline 1 - One Time Greedy Search for {dataset}-{newspaper}-from-{start_date}-to-{end_date} dataset for k={k}') 
accepted_rewirings_threshold = dict()

for threshold in [0.95]:
    t_i = time.time()
    with open(f'data/{dataset}/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}','rb') as f:
        trans_prob = torch.tensor(pickle.load(f), dtype=precision, device=device)

    # compute all possible edge rewirings (according to some threshold)
    permitted_rewirings = select_permitted_rewirings(trans_prob.cpu(),similarity_matrix.cpu(),threshold=threshold)
    print(f'Threshold = {threshold}, number of permitted rewirings = {len(permitted_rewirings)}')

    # damping factor (we allow random walks with restarts)
    damping_factor = 0.015
    if damping_factor:
        updated_trans_prob = (1 - damping_factor) * trans_prob + damping_factor / trans_prob.shape[0]

    F = torch.zeros((n_items,n_items-1,n_items-1), dtype=precision, device=device)
    for abs_node_id in all_items:
        M_tt = compute_mtt_abs(updated_trans_prob, abs_node_id)
        F_ = compute_F(M_tt,precision,device)
        F[abs_node_id] = F_

    accepted_rewirings_threshold.setdefault(threshold, list())
    # exploration of best rewiring from the possible rewirings

    rewiring_increment = list()
    j = 0
    for rewiring in permitted_rewirings:
        source_node = rewiring[0]
        original_target = rewiring[1]
        new_target = rewiring[2]

        e_ = np.zeros(n_items).reshape(-1,1)
        e_[source_node][0] = updated_trans_prob[source_node][original_target].item()-updated_trans_prob[source_node][new_target].item()
        g_ = np.zeros(n_items).reshape(-1,1)

        g_[original_target][0] = -1
        g_[new_target][0] = 1

        # we now create the whole tensor
        e = np.zeros((n_items, n_items-1, 1))
        g = np.zeros((n_items, n_items-1, 1))

        for abs_node_id in all_items:
            e[abs_node_id] = np.delete(e_,abs_node_id,0)
            g[abs_node_id] = np.delete(g_,abs_node_id,0)

        e = torch.tensor(e, dtype=precision,device=device)
        g = torch.tensor(g, dtype=precision,device=device)
        g_t = torch.transpose(g,1,2)

        summations_1 = torch.matmul(torch.matmul(I_T, F), e)
        summations_2 = torch.matmul(torch.matmul(g_t, F), I)
        summations_3 = 1.0-torch.matmul(torch.matmul(g_t, F), e)
        rewiring_increment.append(torch.sum(summations_1*summations_2/summations_3).item())
        j += 1
        if (j/len(permitted_rewirings)) in np.arange(0,1,0.1):
            print(f'{100*(j/len(permitted_rewirings))}% completed')

    # exploration of rewirings has finished --> I will save a sorted list of rewirings (from the most optimal to least)
    # when plotting I will just take the "l" best rewirings
    delta = np.array(rewiring_increment)/n_items**2
    sorted_rewiring_indices = np.argsort(delta)
    sorted_rewirings_to_implement = np.array(permitted_rewirings)[sorted_rewiring_indices]
    accepted_rewirings_threshold[threshold].append(list(sorted_rewirings_to_implement))

    t_f = time.time() 
    print(f'It took {t_f-t_i} seconds')
    # save
    with open(f'results/B1_greedy/rewirings/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_B1_greedy_budget_threshold_{threshold}_damping_factor_{damping_factor}', 'wb') as f:
        pickle.dump(accepted_rewirings_threshold, f, protocol=pickle.HIGHEST_PROTOCOL)
