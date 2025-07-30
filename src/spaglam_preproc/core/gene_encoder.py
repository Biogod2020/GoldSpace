# spaglam_preproc/core/gene_encoder.py

import numpy as np

def generate_gene_sentence(
    expression_vector: np.ndarray,
    gene_names: np.ndarray,
    n_top_genes: int
) -> str:
    """
    Generates a gene sentence string from a single spot's expression vector.
    This function operates entirely in memory.

    Args:
        expression_vector: A 1D numpy array of gene expression values.
        gene_names: A 1D numpy array of corresponding gene names for the expression vector.
        n_top_genes: The number of top genes to include in the sentence.

    Returns:
        A space-separated string of the top N gene names.
    """
    # np.argsort is highly optimized for this task. We reverse it to get descending order.
    sorted_indices = np.argsort(expression_vector)[-1::-1]
    
    # Slice the top N indices and corresponding names
    top_n_indices = sorted_indices[:n_top_genes]
    top_genes = gene_names[top_n_indices]
    
    return " ".join(top_genes)
