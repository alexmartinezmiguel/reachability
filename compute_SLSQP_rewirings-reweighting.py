# import dependencies
import random
import pickle
import argparse
import time
import tqdm
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

from utils import compute_F, compute_mtt_abs, torch_delete, compute_mean_expected_path, \
    compute_At, compute_Bt, compute_gradient, get_top_k, compute_dcg, subgraph_k, select_optimization_variables, \
    compute_coverage, list_similarity


iter_count = 0
reach_k_list = list()
ndcg_list = list()
coverage_list = list()
diversity_list = list()
def fun(x0,x_indices,y_indices):
    """Function to compute Mean Expected Path and its gradient. Variables to optimized are passed as a 
    vector for scipy.minimize routine"""

    # define global variables
    global iter_count
    global reach_k_list
    global ndcg_list
    global coverage_list
    global diversity_list

    # x is (n,)-shaped numpy array -> need to convert it to (n,n)-shaped torch tensor (cuda)
    tmp = np.zeros((n_items,n_items))
    tmp[x_indices,y_indices] = x0
    # save transition probabilities
    with open(f'results/SLSQP/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}_iter_count_{iter_count}', 'wb') as f:
        pickle.dump(tmp,f)
    iter_count += 1

    x_matrix = torch.tensor(tmp, device=device, dtype=precision)
    damping_factor = 0.015
    if damping_factor:
        updated_x_matrix = (1 - damping_factor) * x_matrix + damping_factor / x_matrix.shape[0]

    # compute F and mean expected path
    single_node_reachability = list()
    F = torch.zeros((len(all_items),len(all_items)-1,len(all_items)-1), dtype=precision, device=device)
    for abs_node_id in all_items:
        M_tt = compute_mtt_abs(updated_x_matrix, abs_node_id)
        F_ = compute_F(M_tt,precision, device)
        F[abs_node_id] = F_
        single_node_reachability.append(torch.sum(F_))
    reach = 1/(n_items**2*(n_items-1))*torch.sum(torch.stack(single_node_reachability)).item()

    # compute gradient
    grad = (1.0/(n_items**2*(n_items-1)))*torch.sum(compute_gradient(A_t,B_t,torch.transpose(F,1,2),I), axis=0)[x_indices,y_indices]

    # get top k subgraph to compute metrics
    trans_prob_k = torch.tensor(subgraph_k(x_matrix.cpu().numpy(),k), device=device, dtype=precision)
    if damping_factor:
        updated_trans_prob_k = (1 - damping_factor) * trans_prob_k + damping_factor / trans_prob_k.shape[0]

    # compute reachability of top k subgraph
    single_node_reachability = list()
    F = torch.zeros((len(all_items),len(all_items)-1,len(all_items)-1), dtype=precision, device=device)
    for abs_node_id in all_items:
        M_tt = compute_mtt_abs(updated_trans_prob_k, abs_node_id)
        F_ = compute_F(M_tt,precision, device)
        F[abs_node_id] = F_
        single_node_reachability.append(torch.sum(F_))
    reach_k = 1/(n_items**2*(n_items-1))*torch.sum(torch.stack(single_node_reachability)).item()

    # compute ndcg of top k subgraph
    modified_top_k = dict()
    tmp = x_matrix.cpu().numpy()
    for i in all_items:
        modified_top_k.setdefault(i, get_top_k(tmp[i],k))
    modified_dcg = compute_dcg(topk=modified_top_k, similarities=similarity_matrix, k=k)
    ndcg = np.mean((np.array(modified_dcg)/np.array(ideal_dcg)))

    # compute coverage of top k subgraph
    tmp = trans_prob_k.cpu().numpy()
    coverage = compute_coverage(tmp)

    # compute list dissimilarity of top k subgraph
    similarities = list()    
    for i in range(len(tmp)):
        targets = np.argsort(-tmp[i])[:k]
        similarities.append(list_similarity(targets, similarity_matrix))
    global_similarity = np.mean(similarities)

    # display information
    print(f'{reach_k}, \t {ndcg}, \t {coverage}, \t {global_similarity}')
    

    # save information
    reach_k_list.append(reach_k)
    ndcg_list.append(ndcg)
    coverage_list.append(coverage)
    diversity_list.append(global_similarity)

    with open(f'results/SLSQP/mean_expected_path/mean_expected_path_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}', 'wb') as f:
        pickle.dump(reach_k_list,f)
    with open(f'results/SLSQP/ndcg/ndcg_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}', 'wb') as f:
        pickle.dump(ndcg_list,f)
    with open(f'results/SLSQP/coverage/coverage_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}', 'wb') as f:
        pickle.dump(coverage_list,f)
    with open(f'results/SLSQP/diversity/diversity_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}', 'wb') as f:
        pickle.dump(diversity_list,f)

    return reach, grad.cpu().detach().numpy().flatten()


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
# load arguments 
dataset = 'nelagt'
#newspaper = 'theguardian'
newspaper = args.newspaper
start_date = args.start_date
end_date = args.end_date
k = args.out_degree

# track computational time
t_i = time.time()

# load similarity matrix
with open(f'data/{dataset}//similarity_matrices/similarity_matrix_{newspaper}_from_{start_date}_to_{end_date}','rb') as f:
    similarity_matrix = pickle.load(f)
all_items = set(range(len(similarity_matrix)))
n_items = len(all_items)


# graph parameters
with open(f'data/{dataset}/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}','rb') as f:
    trans_prob = pickle.load(f)

# compute initial ndcg (ideal, ndcg=1)
ideal_top_k = dict()
for i in all_items:
    ideal_top_k.setdefault(i, get_top_k(trans_prob[i],k))
ideal_dcg = compute_dcg(topk=ideal_top_k, similarities=similarity_matrix, k=k)

# compute A_t B_t and I matrices (for the gradient)
A_t = torch.zeros((len(all_items),len(all_items)-1,len(all_items)), dtype=precision, device=device)
B_t = torch.zeros((len(all_items),len(all_items),len(all_items)-1), dtype=precision, device=device)
for abs_node_id in all_items:    
    # second way
    A_t[abs_node_id] = torch.tensor(compute_At(abs_node_id,n_items),dtype=precision, device=device)
    B_t[abs_node_id] = torch.tensor(compute_Bt(abs_node_id,n_items),dtype=precision, device=device)

A_t= torch.transpose(A_t,1,2)
B_t = torch.transpose(B_t,1,2)

I_tmp = torch.ones((n_items-1,1), dtype=precision, device=device)
I_T = I_tmp.T
I = torch.matmul(I_tmp, I_T)
I = I.repeat(n_items,1,1)

# set threshold for rewirings
for threshold, iter in [(0.90,50)]:

    #input of our function
    indices,x0 = select_optimization_variables(trans_prob, similarity_matrix,threshold)

    # define the constraint bounds
    b = 0
    a = 0
    constraint_bounds = list()
    for i in range(len(trans_prob)):
        a = len(np.where(indices[0]==i)[0])
        constraint_bounds.append((b,b+a))
        b += a

    # bounds --> input data (original non zero values) must be between lower_bound (defined by the number of variables) and 1
    lowerBounds = list()
    for cons_bound in constraint_bounds:
        number_variables = cons_bound[1]-cons_bound[0]
        lowerBounds.extend(np.repeat(1./(10*number_variables),number_variables))
    upperBounds=1*np.ones(len(x0))
    boundData=Bounds(np.array(lowerBounds),upperBounds)

    # constraint
    cons = list()
    ub = 0.0
    lb = 0.0
    for cons_bound in constraint_bounds:
        cons.append(NonlinearConstraint(lambda x,bnd=cons_bound: np.sum(x[bnd[0]:bnd[1]]) - 1.0,lb,ub))

    # start minimization loop
    print(f'Running minimization using scipy SLSQP method for {dataset}-{newspaper}-from-{start_date}-to-{end_date} dataset for k={k}')
    print('Mean Expected Path, \t NDCG, \t Coverage, \t Diversity')
    print(f'Threshold = {threshold}, number of variables to optimize = {len(x0)}')
    # minimize
    res = minimize(fun=fun, args=(indices), x0=x0, method='SLSQP', jac=True, bounds=boundData, constraints=cons,options={'disp':False, 'maxiter':iter})
    # build new transition probability matrix
    solution_matrix = np.zeros((n_items,n_items))
    solution_matrix[indices[0],indices[1]] = res.x

    t_f = time.time()
    print(f'It took {t_f-t_i} seconds. Summary:')
    print(res)
    # save result object
    with open(f'results/SLSQP/minimization_objects/result_object_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}','wb') as f:
        pickle.dump(res, f)

    # build and save solution matrix
    with open(f'results/SLSQP/transition_probabilities/transition_probabilities_{newspaper}_from_{start_date}_to_{end_date}_k_{k}_SLSQP_budget_threshold_{threshold}_iter_count_{iter_count}','wb') as f:
        pickle.dump(solution_matrix, f)
