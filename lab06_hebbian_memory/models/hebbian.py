import numpy as np

from config import EPOCHS, LEARNING_RATE

def compute_output(x, w):
    return np.dot(w, x)

def hebbian_update(w, x, y, lr):
    """
    Hebbian learning rule:
    w_new = w + lr * x * y
    """
    # TODO 1:
    # Implement the Hebbian update rule
    w_new = w + lr * x * y
    return w_new

def train_hebbian(X, epochs=EPOCHS, lr=LEARNING_RATE):
    n_features = X.shape[1]
    # start with small weights (if w = 0 then y = 0 and w never changes)
    w = np.ones(n_features) * 0.1
    history = []

    for epoch in range(epochs):
        for x in X:
            y = compute_output(x, w)
            w = hebbian_update(w, x, y, lr)
        history.append(w.copy())

    return w, np.array(history)