import torch
import random
import numpy as np
import networkx as nx


def get_out_degree_distribution(probs):
    """
    Calculate the out-degree distribution of a transition probability matrix.
    
    Parameters:
    probs (np.ndarray): Transition probability matrix.
    
    Returns:
    list: Out-degree distribution.
    """
    out_degree_distribution = list()
    for i in range(len(probs)):
        out_degree_distribution.append(len(np.where(probs[i])[0]))
    return out_degree_distribution

def get_in_degree_distribution(probs):
    """
    Calculate the in-degree distribution of a transition probability matrix.
    
    Parameters:
    probs (np.ndarray): Transition probability matrix.
    
    Returns:
    list: In-degree distribution.
    """
    in_degree_distribution = list()
    for i in range(len(probs)):
        in_degree_distribution.append(len(np.where(probs[:,i])[0]))
    return in_degree_distribution


def compute_F(m_tt, precision, cuda):
    """
    Compute the fundamental matrix F for a given matrix m_tt.
    
    Parameters:
    m_tt (torch.Tensor): Matrix to compute F from.
    precision (torch.dtype): Data type for the computation.
    cuda (torch.device): Device to perform the computation on.
    
    Returns:
    torch.Tensor: Fundamental matrix F.
    """
    m_tt = m_tt * -1.

    for ix in range(m_tt.shape[0]):
        m_tt[ix, ix] += 1.
    f = torch.linalg.solve(m_tt,torch.eye(len(m_tt), dtype=precision, device=cuda))
    return f


# https://gist.github.com/velikodniy/6efef837e67aee2e7152eb5900eb0258
def torch_delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    """
    Delete a specific index from a tensor along a given dimension.
    
    Parameters:
    arr (torch.Tensor): Input tensor.
    ind (int): Index to delete.
    dim (int): Dimension along which to delete the index.
    
    Returns:
    torch.Tensor: Tensor with the specified index removed.
    """
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


def compute_mtt_abs(trans_probs, abs_item):
    """
    Compute the sub-matrix of transient nodes from a transition probability matrix.
    
    Parameters:
    trans_probs (torch.Tensor): Transition probability matrix.
    abs_item (int): Index of the absorbing node.
    
    Returns:
    torch.Tensor: Sub-matrix of transient nodes.
    """
    M_tt = torch_delete(torch_delete(trans_probs,abs_item,1),abs_item,0)
    return M_tt

# given a matrix, normalize its rows such that they add up to 1
def row_normalize(M):
    """
    Normalize the rows of a matrix so that each row sums to 1.
    
    Parameters:
    M (np.ndarray): Input matrix.
    
    Returns:
    np.ndarray: Row-normalized matrix.
    """
    return (M.T/M.sum(axis=1)).T

def select_permitted_rewirings(probabilities,similarities,threshold):
    """
    Select permitted rewirings based on similarity threshold.
    
    Parameters:
    probabilities (np.ndarray): Transition probability matrix.
    similarities (np.ndarray): Similarity matrix.
    threshold (float): Similarity threshold.
    
    Returns:
    list: List of permitted rewirings as tuples (source, original_target, new_target).
    """
    permitted_rewirings = list()
    for source_node in range(len(probabilities)):
        original_target_nodes = np.where(probabilities[source_node]!=0.0)[0]
        last_original_target_node = np.argsort(-probabilities[source_node])[len(original_target_nodes)-1].item()
        last_original_target_similarity = similarities[source_node][last_original_target_node].item()
        full_list_target_nodes = np.where(similarities[source_node]!=0.0)[0]
        new_possible_target_nodes = set(full_list_target_nodes).difference(original_target_nodes).difference([source_node])
        new_permitted_target_nodes = list()
        for node in new_possible_target_nodes:
            if similarities[source_node][node].item()/last_original_target_similarity >= threshold:
                new_permitted_target_nodes.append(node)
        for original_target in original_target_nodes:
            for new_target in new_permitted_target_nodes:
                permitted_rewirings.append((source_node, original_target,new_target))
    return permitted_rewirings

def select_permitted_rewirings_shufflik(probabilities):
    """
    Select permitted rewirings within the same list of recommended nodes
    
    Parameters:
    probabilities (np.ndarray): Transition probability matrix.
    
    Returns:
    list: List of permitted rewirings as tuples (source, original_target, new_target).
    """
    permitted_rewirings = list()
    for source_node in range(len(probabilities)):
        original_target_nodes = np.where(probabilities[source_node]!=0.0)[0]
        possible_pairs = list(combinations(original_target_nodes, 2))
        permitted_rewirings.extend([(source_node, pair[0],pair[1]) for pair in possible_pairs])
    return permitted_rewirings


def chunks(l, n):
    """
    Yield n number of striped chunks from a list.
    
    Parameters:
    l (list): Input list.
    n (int): Number of chunks.
    
    Yields:
    list: Chunks of the input list.
    """
    l_copy = l.copy()
    random.seed(42)
    random.shuffle(l_copy)
    for i in range(0, n):
        yield l_copy[i::n]

def compute_mean_expected_path(F, n_items):
    """
    Compute the mean expected path length from fundamental matrices.
    
    Parameters:
    F (list of torch.Tensor): List of fundamental matrices.
    n_items (int): Number of items.
    
    Returns:
    float: Mean expected path length.
    """
    single_node_reachability = list()
    for f in F:
        single_node_reachability.append(torch.sum(f))
    return 1/(n_items**2*(n_items-1))*torch.sum( torch.stack(single_node_reachability)).item()

def compute_pairwise_shortest_path(G):
    """
    Compute pairwise shortest path lengths in a graph.
    
    Parameters:
    G (networkx.Graph): Input graph.
    
    Returns:
    np.ndarray: Matrix of pairwise shortest path lengths.
    """
    pairwise_shortest_path = np.zeros((len(G),len(G)))
    for item in range(len(G)):
        item_shortest_paths = nx.shortest_path_length(G,item)
        for item2 in range(len(G)):
            pairwise_shortest_path[item,item2] = item_shortest_paths.get(item2, np.NAN)
    return pairwise_shortest_path

def compute_mean_shortest_path(pairwise_shortest_path, n_items):
    """
    Compute the mean shortest path length, excluding self-loops.
    
    Parameters:
    pairwise_shortest_path (np.ndarray): Matrix of pairwise shortest path lengths.
    n_items (int): Number of items.
    
    Returns:
    float: Mean shortest path length.
    """
    return 1/(n_items*(n_items-1))*np.sum(pairwise_shortest_path)

def list_similarity(items,similarities):
    """
    Compute the mean similarity between pairs of items.
    
    Parameters:
    items (list): List of item indices.
    similarities (np.ndarray): Similarity matrix.
    
    Returns:
    float: Mean similarity.
    """
    ils = list()
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            ils.append(similarities[items[i],items[j]].item())
    return np.mean(ils)

def compute_coverage(probs):
    """
    Compute the coverage of a transition probability matrix.
    
    Parameters:
    probs (np.ndarray): Transition probability matrix.
    
    Returns:
    int: Number of unique non-zero elements in the matrix.
    """
    return len(set(np.nonzero(probs)[1]))


def compute_At(t,n_items):
    """
    Compute matrix A_t for a given t and number of items.
    
    Parameters:
    t (int): Index t.
    n_items (int): Number of items.
    
    Returns:
    np.ndarray: Matrix A_t.
    """
    A_t = np.zeros((n_items-1,n_items))
    for i in range(n_items-1):
        for j in range(n_items):
            if i==j and j<t:
                A_t[i,j] = 1.0
            elif j == i+1 and j>t:
                A_t[i,j] = 1.0
    return A_t


def compute_Bt(t,n_items):
    """
    Compute matrix B_t for a given t and number of items.
    
    Parameters:
    t (int): Index t.
    n_items (int): Number of items.
    
    Returns:
    np.ndarray: Matrix B_t.
    """
    B_t = np.zeros((n_items,n_items-1))
    for i in range(n_items):
        for j in range(n_items-1):
            if i==j and i<t:
                B_t[i,j] = 1.0
            elif i == j+1 and i>t:
                B_t[i,j] = 1.0
    return B_t



def compute_gradient(A, B, F, I):
    """
    Compute the gradient of a function with respect to matrices A, B, F, and I.
    
    Parameters:
    A (torch.Tensor): Matrix A.
    B (torch.Tensor): Matrix B.
    F (torch.Tensor): Fundamental matrix F.
    I (torch.Tensor): Identity vector I.
    
    Returns:
    torch.Tensor: Computed gradient.
    """
    return torch.matmul(A,torch.matmul(F,torch.matmul(I,torch.matmul(F,B))))


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def get_top_k(probs,k):
    """
    Get the indices of the top k elements in a probability distribution.
    
    Parameters:
    probs (np.ndarray): Probability distribution.
    k (int): Number of top elements to select.
    
    Returns:
    list: Indices of the top k elements.
    """
    return list(np.argsort(-probs)[:k])

def compute_dcg(topk, similarities,k):
    """
    Compute the Discounted Cumulative Gain (DCG) for a set of top-k elements.
    
    Parameters:
    topk (list): Indices of top-k elements.
    similarities (np.ndarray): Similarity matrix.
    k (int): Number of top elements.
    
    Returns:
    list: DCG values for each set of top-k elements.
    """
    dcg = list()
    for i in range(len(similarities)):
        relevance_scores = similarities[i][topk[i]]
        dcg.append(np.sum(relevance_scores/(1+np.log2(1+np.arange(1,k+1)))))
    return dcg


def subgraph_k(probs,k):
    """
    Create a subgraph with top-k connections for each node based on probabilities.
    
    Parameters:
    probs (np.ndarray): Transition probability matrix.
    k (int): Number of top connections to retain.
    
    Returns:
    np.ndarray: Subgraph with top-k connections.
    """
    subgraph_k = np.zeros((len(probs),len(probs)))
    for i in range(len(probs)):
        top_k_indices_i = np.argsort(-probs[i])[:k]
        subgraph_k[i][top_k_indices_i] = probs[i][top_k_indices_i]/np.sum(probs[i][top_k_indices_i])
    return subgraph_k

def select_optimization_variables(probabilities,similarities,threshold):
    """
    Select optimization variables based on similarity threshold.
    
    Parameters:
    probabilities (np.ndarray): Transition probability matrix.
    similarities (np.ndarray): Similarity matrix.
    threshold (float): Similarity threshold.
    
    Returns:
    tuple: Arrays of source nodes, target nodes, and their values.
    """
    source_nodes = list()
    target_nodes = list()
    values = list()
    for source_node in range(len(probabilities)):
        original_target_nodes = np.where(probabilities[source_node]!=0.0)[0]
        last_original_target_node = np.argsort(-probabilities[source_node])[len(original_target_nodes)-1].item()
        last_original_target_similarity = similarities[source_node][last_original_target_node].item()
        full_list_target_nodes = np.where(similarities[source_node]!=0.0)[0]
        new_possible_target_nodes = set(full_list_target_nodes).difference([source_node])
        values_tmp = list()
        for node in new_possible_target_nodes:
            if similarities[source_node][node].item()/last_original_target_similarity >= threshold:
                source_nodes.append(source_node)
                target_nodes.append(node)
                values_tmp.append(similarities[source_node][node].item())
        values.extend(list(np.array(values_tmp)/np.sum(values_tmp)))
    return (np.array(source_nodes), np.array(target_nodes)), np.array(values)


def get_index(source_node, abs_node_id):
    """
    Get the adjusted index of a source node relative to an absorbing node.
    
    Parameters:
    source_node (int): Index of the source node.
    abs_node_id (int): Index of the absorbing node.
    
    Returns:
    int: Adjusted index of the source node.
    """
    return source_node - 1 if source_node > abs_node_id else source_node
    
def get_row_diff(F,abs_node_id,original_target,new_target):
    """
    Compute the difference between rows in a fundamental matrix F.
    
    Parameters:
    F (torch.Tensor): Fundamental matrix.
    abs_node_id (int): Index of the absorbing node.
    original_target (int): Index of the original target node.
    new_target (int): Index of the new target node.
    
    Returns:
    torch.Tensor: Difference between the specified rows.
    """
    original_target_index = original_target - (1 if abs_node_id < original_target else 0)
    new_target_index = new_target - (1 if abs_node_id < new_target else 0)
    
    # row differences
    if abs_node_id != original_target and abs_node_id != new_target:
        return F[new_target_index]-F[original_target_index]
    elif abs_node_id == original_target:
        return F[new_target_index]
    elif abs_node_id == new_target:
        return -F[original_target_index]
