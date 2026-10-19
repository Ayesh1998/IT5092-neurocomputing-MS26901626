import numpy as np

# -----------------------------------------
# Entropy Computation
# -----------------------------------------

def compute_entropy(probabilities):
    """
    Compute Shannon entropy H(X) in bits.

    Parameters:
        probabilities (array-like): probability distribution (must sum to 1)

    Returns:
        float: entropy in bits
    """

    # TODO 1:
    # Remove zero probabilities to avoid log(0)
    p = np.array(probabilities, dtype=float)
    p = p[p > 0]

    # TODO 2:
    # Compute entropy using:
    # H = -sum(p * log2(p))
    H = -np.sum(p * np.log2(p))

    # avoid printing -0.0 when there is no uncertainty
    if H == 0:
        H = 0.0

    return H


def normalize_distribution(counts):
    """
    Convert raw counts into probability distribution.
    """

    # TODO 3:
    # Normalize counts so they sum to 1
    counts = np.array(counts, dtype=float)
    probabilities = counts / np.sum(counts)

    return probabilities