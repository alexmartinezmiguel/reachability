import torch
import numpy as np 
import argparse
import random
import pickle
import time
import tqdm
import time

from utils import compute_F, compute_mtt_abs, torch_delete, \
     chunks, select_permitted_rewirings, compute_mean_expected_path, get_index, get_row_diff


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("newspaper", type=str)
parser.add_argument("start_date", type=str)
parser.add_argument("end_date", type=str)
parser.add_argument("out_degree", type=int)
parser.add_argument("batch_size", type=int)
args = parser.parse_args()

# set device (CPU or GPU) and precision for the computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cpu':
    print('Could not use GPU, using CPU instead')
precision = torch.float32

# load similarity matrix and transition probabilities of given dataset 
dataset = 'nelagt'
#newspaper = 'theguardian'
newspaper = args.newspaper
start_date = args.start_date
end_date = args.end_date
k = args.out_degree
batch_size = args.batch_size


# load similarity matrix
with open(f'data/{dataset}/similarity_matrices/similarity_matrix_{newspaper}_from_{start_date}_to_{end_date}','rb') as f:
    similarity_matrix = torch.tensor(pickle.load(f), dtype=precision, device=torch.device('cpu'))
# useful objects for computations
all_items = set(range(len(similarity_matrix)))
n_items = len(similarity_matrix)
I = torch.ones((n_items-1,1), dtype=precision, device=device)
I_T = I.T

print(f'Running Batch Greedy Search for {dataset}-{newspaper}-from-{start_date}-to-{end_date} dataset for k={k}') 
accepted_rewirings_threshold = dict()

for threshold in [0.95]:
    with open(f'data/{dataset}/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}','rb') as f:
        trans_prob = torch.tensor(pickle.load(f), dtype=precision, device=torch.device('cpu'))

    # compute all possible edge rewirings (according to some threshold)
    permitted_rewirings = select_permitted_rewirings(trans_prob,similarity_matrix,threshold=threshold)
    print(f'Threshold = {threshold}, number of permitted rewirings = {len(permitted_rewirings)}')
    del similarity_matrix

    # damping factor (we allow random walks with restarts)
    damping_factor = 0.015
    if damping_factor:
        updated_trans_prob = (1 - damping_factor) * trans_prob + damping_factor / trans_prob.shape[0]
    # free space
    trans_prob.cpu().detach()
    del trans_prob
    F = torch.zeros((n_items,n_items-1,n_items-1), dtype=precision, device=device)
    for abs_node_id in all_items:
        M_tt = compute_mtt_abs(updated_trans_prob.cuda(), abs_node_id)
        F_ = compute_F(M_tt,precision,device)
        F[abs_node_id] = F_

    accepted_rewirings_threshold.setdefault(threshold, list())
    # exploration of best rewiring from the possible rewirings
    j = 1
    #batch_size = 50
    print(f'Number of splits = {batch_size}')
    permitted_rewirings_chunked = [split for split in chunks(permitted_rewirings,batch_size)]
    n_rewirings = 50
    t_i = time.time()
    for m in range(int(n_rewirings/batch_size)):
        for split in permitted_rewirings_chunked:
            rewiring_increment = list()

            for rewiring in split:
                source_node, original_target, new_target = rewiring
                prob_diff = updated_trans_prob[source_node][original_target].item()-updated_trans_prob[source_node][new_target].item()
                indices = [get_index(source_node, abs_node_id) for abs_node_id in all_items]

                # Precompute sigma_ and tau_ tensors
                sigma_ = torch.stack([
                    F[abs_node_id][:, indices[abs_node_id]] if abs_node_id != source_node else torch.zeros((n_items-1,), device=device)
                    for abs_node_id in all_items
                ], dim=1)

                tau_ = prob_diff * torch.stack([
                    get_row_diff(F[abs_node_id], abs_node_id, original_target, new_target)
                    for abs_node_id in all_items
                ], dim=1)
                tmp_ = tau_.T
                # Compute sigma, tau, and rho
                sigma = torch.sum(sigma_, dim=0)
                tau = torch.sum(tau_, dim=0)
                rho = 1.0 - torch.stack([
                    tmp_[abs_node_id][indices[abs_node_id]]
                    if abs_node_id != source_node else torch.tensor(0.0, device=device)
                    for abs_node_id in all_items
                ])
                rewiring_increment.append(torch.sum(sigma*tau/rho).item())



            # exploration of rewirings has finished --> need to find the best one
            rewiring_index = np.argmin(rewiring_increment)
            rewiring_to_implement = split[rewiring_index]
            print(f'\t \t {j}. ', rewiring_to_implement)
            j += 1
            accepted_rewirings_threshold[threshold].append(rewiring_to_implement)
            source_node, original_target, new_target = rewiring_to_implement
            prob_diff = updated_trans_prob[source_node][original_target].item()-updated_trans_prob[source_node][new_target].item()

            # recompute F
            numerator_1 = torch.stack([
                F[abs_node_id][:, get_index(source_node, abs_node_id)] if abs_node_id != source_node else torch.zeros((n_items-1,), device=device)
                for abs_node_id in all_items
            ], dim=0)

            numerator_2 = prob_diff * torch.stack([
                get_row_diff(F[abs_node_id], abs_node_id, original_target, new_target)
                for abs_node_id in all_items
            ], dim=0)

            denominator = 1.0 - torch.stack([
                    numerator_2[abs_node_id][get_index(source_node, abs_node_id)]
                    if abs_node_id != source_node else torch.tensor(0.0, device=device)
                    for abs_node_id in all_items
                ])
    
            F += torch.div(torch.matmul(numerator_1.reshape(n_items,n_items-1,1),numerator_2.reshape(n_items,1,n_items-1)),denominator.reshape(n_items,1,1))
            
            numerator_1 = numerator_1.cpu().detach()
            numerator_2 = numerator_2.cpu().detach()
            denominator = denominator.cpu().detach()
            del numerator_1, numerator_2, denominator
            torch.cuda.empty_cache() 
            # update permitted rewires
            del split[rewiring_index]

            # update transition probabilities
            value_1 = updated_trans_prob[source_node][original_target].item()
            value_2 = updated_trans_prob[source_node][new_target].item()

            updated_trans_prob[source_node][new_target] = value_1
            updated_trans_prob[source_node][original_target] = value_2

            # save
            with open(f'results/BGS/rewirings_v2/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_BGS_budget_threshold_{threshold}_batch_size_{batch_size}_damping_factor_{damping_factor}', 'wb') as f:
                pickle.dump(accepted_rewirings_threshold, f, protocol=pickle.HIGHEST_PROTOCOL)

    t_f = time.time() 
    print(f'It took {t_f-t_i} seconds')
    # save: in the name of the files we need to specify the threshold for the ranking metric (to filter possible rewirings)
    with open(f'results/BGS/rewirings_v2/accepted_rewirings_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_BGS_budget_threshold_{threshold}_batch_size_{batch_size}_damping_factor_{damping_factor}', 'wb') as f:
        pickle.dump(accepted_rewirings_threshold, f, protocol=pickle.HIGHEST_PROTOCOL)





    