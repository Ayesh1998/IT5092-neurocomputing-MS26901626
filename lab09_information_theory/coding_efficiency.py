import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Coding Efficiency
# -----------------------------------------

def compare_distributions():
    """
    Compare entropy of:
    - Uniform distribution
    - Skewed distribution
    """

    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    skewed = np.array([0.7, 0.1, 0.1, 0.1])

    # TODO:
    # Compute entropy of both
    H_uniform = compute_entropy(uniform)
    H_skewed = compute_entropy(skewed)

    # Print comparison
    print("Uniform distribution:", uniform, "-> Entropy =", round(H_uniform, 4), "bits")
    print("Skewed distribution :", skewed, "-> Entropy =", round(H_skewed, 4), "bits")
    print("Maximum possible entropy (4 states) = log2(4) =", np.log2(4), "bits")

    if H_uniform > H_skewed:
        print("Uniform distribution has higher entropy, so it is more efficient.")
    else:
        print("Skewed distribution has higher entropy, so it is more efficient.")

    return H_uniform, H_skewed