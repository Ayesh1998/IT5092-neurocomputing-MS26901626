import numpy as np

def train_associative_memory(patterns):
    """
    Store bipolar patterns using outer-product Hebbian learning
    """
    n_features = patterns.shape[1]
    W = np.zeros((n_features, n_features))

    for p in patterns:
        # TODO 4:
        # Add outer product of p with itself
        W = W + np.outer(p, p)

    np.fill_diagonal(W, 0)
    return W

def recall_pattern(W, cue, steps=5):
    x = cue.copy()

    for _ in range(steps):
        # TODO 5:
        # Update pattern using sign(W @ x)
        x = np.sign(W @ x)

    return x