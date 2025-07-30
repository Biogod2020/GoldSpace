# spaglam_preproc/core/graph_builder.py

from collections import deque
from scipy.sparse import csr_matrix

def get_k_hop_neighborhood(
    adjacency_matrix: csr_matrix, start_node_idx: int, k: int
) -> list[int]:
    """
    Finds all unique nodes within k hops of a starting node using Breadth-First Search (BFS).
    This is the most efficient method for this task.

    Args:
        adjacency_matrix: The sparse adjacency matrix (e.g., from adata.obsp['spatial_connectivities']).
        start_node_idx: The integer index of the starting node.
        k: The number of hops (e.g., 1 for immediate neighbors, 2 for neighbors of neighbors).

    Returns:
        A list of unique integer indices of all nodes in the k-hop neighborhood, including the start node.
    """
    if k == 0:
        return [start_node_idx]

    visited = {start_node_idx}
    # queue stores tuples of (node_index, current_hop_level)
    queue = deque([(start_node_idx, 0)])
    
    # We add the start node to the final list
    neighborhood = [start_node_idx]

    while queue:
        current_node, level = queue.popleft()

        if level >= k:
            continue

        # Get neighbors using the efficient .indices attribute of CSR matrices
        # adjacency_matrix.indices[start:end] slices the column indices for a given row
        start_ptr = adjacency_matrix.indptr[current_node]
        end_ptr = adjacency_matrix.indptr[current_node + 1]
        neighbors = adjacency_matrix.indices[start_ptr:end_ptr]

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                neighborhood.append(neighbor)
                queue.append((neighbor, level + 1))
                
    return neighborhood
