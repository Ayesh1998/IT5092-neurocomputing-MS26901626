import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Mutual Information
# -----------------------------------------

def compute_joint_distribution(stimulus, response):
    """
    Compute empirical joint probability P(S, R)
    """

    # TODO 1:
    # Find unique stimulus and response values
    stimulus = np.array(stimulus)
    response = np.array(response)
    s_values = np.unique(stimulus)
    r_values = np.unique(response)

    # Count co-occurrences
    joint = np.zeros((len(s_values), len(r_values)))
    for i in range(len(s_values)):
        for j in range(len(r_values)):
            joint[i, j] = np.sum((stimulus == s_values[i]) & (response == r_values[j]))

    # Normalize to probabilities
    joint = joint / np.sum(joint)

    return joint


def compute_mutual_information(stimulus, response):
    """
    Compute I(S ; R)
    """

    # TODO 2:
    # Compute joint distribution
    joint = compute_joint_distribution(stimulus, response)

    # TODO 3:
    # Compute marginal P(S) and P(R)
    p_s = np.sum(joint, axis=1)
    p_r = np.sum(joint, axis=0)

    # TODO 4:
    # Use formula:
    # I = sum P(s,r) * log2( P(s,r) / (P(s)P(r)) )
    I = 0
    for i in range(len(p_s)):
        for j in range(len(p_r)):
            if joint[i, j] > 0:
                I = I + joint[i, j] * np.log2(joint[i, j] / (p_s[i] * p_r[j]))

    return I