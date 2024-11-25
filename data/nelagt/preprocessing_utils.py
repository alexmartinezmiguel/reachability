import sqlite3
import numpy as np
import pandas as pd
import random

def execute_query(path, query):
    """
    Execute a given SQL query on the database and return the results.
    
    Parameters:
    path (str): Path to the SQLite database file.
    query (str): SQL query to execute.
    
    Returns:
    list: Results of the query as a list of tuples.
    """
    conn = sqlite3.connect(path)
    results = conn.cursor().execute(query).fetchall()
    return results

def execute_query_pandas(path, query):
    """
    Execute a SQL query and load the results into a pandas DataFrame.
    
    Parameters:
    path (str): Path to the SQLite database file.
    query (str): SQL query to execute.
    
    Returns:
    pd.DataFrame: Results of the query as a DataFrame.
    """
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(query, conn)
    return df

def zero_diagonal(matrix):
    """
    Set the diagonal elements of a matrix to zero.
    
    Parameters:
    matrix (np.ndarray): Input matrix.
    
    Returns:
    np.ndarray: Matrix with zeroed diagonal.
    """
    for i in range(len(matrix)):
        matrix[i, i] = 0.0
    return matrix

def cut_links(similarity_matrix, k):
    """
    Modify a similarity matrix to retain only the top-k items for each row.
    
    Parameters:
    similarity_matrix (np.ndarray): Item similarity matrix.
    k (int): Number of top items to retain.
    
    Returns:
    np.ndarray: Modified similarity matrix with top-k items retained.
    """
    tmp = similarity_matrix.copy()
    for source_node in range(len(tmp)):
        non_zero_tragets = np.nonzero(tmp[source_node])[0]
        targets_to_keep = np.argsort(tmp[source_node])[-k:]
        targets_to_delete = set(non_zero_tragets).difference(targets_to_keep)
        for target_to_delete in targets_to_delete:
            tmp[source_node][target_to_delete] = 0
        # check if source node has the desired out-degree, otherwise sample random
        existing_links = np.nonzero(tmp[source_node])[0]
        if len(existing_links)<k:
            print(source_node)
            all_items = set(range(len(similarity_matrix)))
            n_targets_to_sample = k-len(existing_links)
            # avoid creating a link to itself
            targets_to_sample = all_items.difference(list(existing_links))
            targets_to_sample = targets_to_sample.difference([source_node])
            new_targets = random.sample(list(targets_to_sample), n_targets_to_sample)
            for new_target in new_targets:
                tmp[source_node][new_target] = 1
    return tmp

def row_normalize(M):
    """
    Normalize the rows of a matrix so that each row sums to 1.
    
    Parameters:
    M (np.ndarray): Input matrix.
    
    Returns:
    np.ndarray: Row-normalized matrix.
    """
    return (M.T/M.sum(axis=1)).T